"""Gemma 4 second judge: gold validation, resumable coding, SDT analysis.

    python scripts/second_judge/run_gemma_judge.py --gold
    python scripts/second_judge/run_gemma_judge.py --code --priority 1 --workers 2
    python scripts/second_judge/run_gemma_judge.py --code --priority 2 --workers 2
    python scripts/second_judge/run_gemma_judge.py --analyze
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge.anthropic_judge import load_gold  # noqa: E402
from judge.codebook_v2 import prompt_sha256_v2, system_prompt_v2  # noqa: E402
from scripts.agreement import atomic_csv  # noqa: E402
from scripts.analyze_full import B, load_items, role, sdt  # noqa: E402
from scripts.code_replies import CLARIFICATION, judge_case, parse_reply_judgment  # noqa: E402
from scripts.judge_robustness import corrected_sdt  # noqa: E402
from scripts.rater_metrics import (  # noqa: E402
    binary_pairs,
    kappa_with_interval,
    rogan_gladen,
    sensitivity_specificity,
)
from scripts.run_qwen_judge_scoped import SKIP_LISTENERS  # noqa: E402
from scripts.scoped_judge import scoped_tasks as _scoped_tasks  # noqa: E402
from scripts.second_judge.gemma_common import (  # noqa: E402
    AGREE_CSV,
    CACHE,
    CODINGS,
    CONTRAST_CSV,
    FAIL_LOG,
    GOLD_CSV,
    MODEL,
    OUT,
    PINNED_HASH,
    SDT_CSV,
    SIDE_CSV,
    TABLE3_CSV,
    append_jsonl,
    cache_key,
    call_gemma,
    load_done_ids,
    reply_id,
)

LOCK = threading.Lock()
MANIFEST = ROOT / "results" / "qwen_judge_expand6k_manifest.csv"
QWEN_CODINGS = ROOT / "outputs" / "codings_qwen_expand6k.jsonl"
QWEN_GOLD_VAL = ROOT / "results" / "qwen_expand6k_gold_validation.csv"
KAPPA_STOP = 0.40
PRIORITY1_LISTENERS = {"qwen/qwen3.8-27b", "human"}
PRIORITY2_LISTENERS = {
    "mistral-small-3.1-24b-instruct-2503",
    "unsloth/gpt-oss-20b",
    "llama-3.2-3b-instruct",
}
GEMMA_SELF = "google/gemma-4-26b-a4b-qat"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_pool() -> dict[str, dict]:
    tasks, _counts = _scoped_tasks(ROOT)
    pool = {}
    for task in tasks:
        if task["listener"] in SKIP_LISTENERS:
            continue
        pool[reply_id(task["listener"], task["condition"], task["item_id"])] = task
    return pool


def _strip_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_text(text: str):
    return parse_reply_judgment(_strip_fences(text)) if text else None


def judge_one(system: str, task: dict, *, retries: int = 3) -> dict:
    rid = reply_id(task["listener"], task["condition"], task["item_id"])
    key = cache_key(task["listener"], task["condition"], task["item_id"], task["context"], task["reply"])
    cache_path = CACHE / f"{key}.json"
    text = ""
    usage: dict = {}
    error = ""
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        text = cached.get("text") or ""
        usage = cached.get("usage") or {}
        if _parse_text(text) is None:
            cache_path.unlink(missing_ok=True)
            text, usage = "", {}
    if not text or _parse_text(text) is None:
        from scripts.code_replies import CATEGORIES

        user = judge_case(task["context"], task["reply"], qwen=True)
        last_err = ""
        text = ""
        for attempt in range(retries):
            prompt = user
            if attempt > 0:
                prompt = (
                    user
                    + "\n\nReminder: category must be exactly one of: "
                    + ", ".join(CATEGORIES)
                    + ". Do not use wrong_role as a category; wrong_role is only a boolean field. "
                    "Output raw JSON only, with no markdown fences."
                )
            try:
                text, usage = call_gemma(system, prompt)
                if _parse_text(text) is not None:
                    last_err = ""
                    break
                last_err = "parse_failed"
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                text, usage = "", {}
            time.sleep(1.5 * (attempt + 1))
        error = last_err
        if _parse_text(text) is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"text": text, "usage": usage, "error": ""}, ensure_ascii=False),
                encoding="utf-8",
            )
    parsed = _parse_text(text)
    self_judged = task["listener"] == GEMMA_SELF
    return {
        "reply_id": rid,
        "item_id": task["item_id"],
        "listener": task["listener"],
        "condition": task["condition"],
        "judge_model": MODEL,
        "prompt_version": "precedence_v2",
        "prompt_sha256": PINNED_HASH,
        "failed": parsed is None,
        "error": error or None,
        "raw": text,
        "category": None if parsed is None else parsed["category"],
        "wrong_role": None if parsed is None else parsed["wrong_role"],
        "corrects_error": None if parsed is None else parsed["corrects_error"],
        "self_judged": self_judged,
        "exclude_from_conclusions": self_judged,
    }


def run_gold(*, workers: int = 1) -> float:
    if prompt_sha256_v2() != PINNED_HASH:
        raise SystemExit("precedence_v2 hash mismatch")
    system = system_prompt_v2()
    gold = {row["annotation_id"]: row["category"] for row in _read_csv(ROOT / "annotation" / "gold_final.csv")}
    splits = {row["annotation_id"]: row["split"] for row in _read_csv(ROOT / "annotation" / "gold_split.csv")}
    a1 = {row["annotation_id"]: row["category"] for row in _read_csv(ROOT / "annotation" / "labels_annotator1.csv")}
    a2 = {row["annotation_id"]: row["category"] for row in _read_csv(ROOT / "annotation" / "labels_annotator2.csv")}
    items = load_gold(ROOT / "annotation")
    if len(items) != 400:
        raise SystemExit(f"expected 400 gold items, found {len(items)}")

    def one(item: dict) -> dict:
        task = {
            "listener": "gold",
            "condition": item["condition"] if item["source"] != "human" else "human",
            "item_id": item["annotation_id"],
            "context": item["context"],
            "reply": item["reply"],
        }
        # Cache under annotation id as item_id.
        row = judge_one(system, task)
        return {
            "annotation_id": item["annotation_id"],
            "split": splits[item["annotation_id"]],
            "source": item["source"],
            "gold_category": gold[item["annotation_id"]],
            "category": row.get("category"),
            "failed": row.get("failed"),
            "raw": row.get("raw"),
        }

    t0 = time.perf_counter()
    scored: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(one, item) for item in items]
        for i, fut in enumerate(as_completed(futures), start=1):
            scored.append(fut.result())
            if i % 20 == 0 or i == len(items):
                print(f"gold {i}/{len(items)}", flush=True)
    # Persist gold codings separately (do not mix with expand reply_ids).
    gold_path = OUT / "gemma_gold_codes.jsonl"
    if gold_path.exists():
        gold_path.unlink()
    for row in scored:
        append_jsonl(gold_path, row)

    def metrics(subset: list[dict], label: str) -> dict:
        pairs = [(row["gold_category"], row["category"]) for row in subset if row.get("category")]
        clar = binary_pairs(pairs)
        k_clar = kappa_with_interval(clar)
        k_cat = kappa_with_interval(pairs)
        se, sp = sensitivity_specificity(clar)
        return {
            "row": label,
            "n": str(len(pairs)),
            "failed": str(sum(1 for row in subset if row.get("failed"))),
            "kappa_clarification": "" if k_clar["value"] is None else f"{k_clar['value']:.4f}",
            "kappa_clarification_lo": "" if k_clar["lo"] is None else f"{k_clar['lo']:.4f}",
            "kappa_clarification_hi": "" if k_clar["hi"] is None else f"{k_clar['hi']:.4f}",
            "kappa_category": "" if k_cat["value"] is None else f"{k_cat['value']:.4f}",
            "kappa_category_lo": "" if k_cat["lo"] is None else f"{k_cat['lo']:.4f}",
            "kappa_category_hi": "" if k_cat["hi"] is None else f"{k_cat['hi']:.4f}",
            "sensitivity": "" if se is None else f"{se:.4f}",
            "specificity": "" if sp is None else f"{sp:.4f}",
            "seconds": f"{time.perf_counter() - t0:.1f}",
        }

    test = [row for row in scored if row["split"] == "test"]
    all_rows = [metrics(test, "gemma_vs_gold_test")]
    # Confusion on test.
    confusion = Counter()
    for row in test:
        if not row.get("category"):
            continue
        g = "clar" if row["gold_category"] in CLARIFICATION else "other"
        p = "clar" if row["category"] in CLARIFICATION else "other"
        confusion[(g, p)] += 1
    for (g, p), n in sorted(confusion.items()):
        all_rows.append(
            {
                "row": "confusion",
                "n": str(n),
                "failed": "",
                "kappa_clarification": "",
                "kappa_clarification_lo": "",
                "kappa_clarification_hi": "",
                "kappa_category": "",
                "kappa_category_lo": "",
                "kappa_category_hi": "",
                "sensitivity": "",
                "specificity": "",
                "seconds": "",
                "gold": g,
                "pred": p,
            }
        )
    fields = (
        "row",
        "n",
        "failed",
        "kappa_clarification",
        "kappa_clarification_lo",
        "kappa_clarification_hi",
        "kappa_category",
        "kappa_category_lo",
        "kappa_category_hi",
        "sensitivity",
        "specificity",
        "seconds",
        "gold",
        "pred",
    )
    atomic_csv(GOLD_CSV, fields, [{key: row.get(key, "") for key in fields} for row in all_rows])

    # Table 3 style: human-human, Qwen vs gold, Gemma vs gold (test half).
    test_ids = [aid for aid, split in splits.items() if split == "test"]
    hh_pairs = [(a1[aid], a2[aid]) for aid in test_ids]
    hh_clar = binary_pairs(hh_pairs)
    hh_k = kappa_with_interval(hh_clar)
    hh_cat = kappa_with_interval(hh_pairs)
    hh_se, hh_sp = sensitivity_specificity(hh_clar)

    qwen_val = _read_csv(QWEN_GOLD_VAL)
    qwen_main = next(row for row in qwen_val if row.get("split") == "test")

    gemma = all_rows[0]
    table = [
        {
            "comparison": "human_vs_human",
            "n": str(len(hh_pairs)),
            "kappa_clarification": f"{hh_k['value']:.4f}",
            "kappa_clarification_lo": f"{hh_k['lo']:.4f}",
            "kappa_clarification_hi": f"{hh_k['hi']:.4f}",
            "kappa_category": f"{hh_cat['value']:.4f}",
            "kappa_category_lo": f"{hh_cat['lo']:.4f}",
            "kappa_category_hi": f"{hh_cat['hi']:.4f}",
            "sensitivity": f"{hh_se:.4f}",
            "specificity": f"{hh_sp:.4f}",
            "note": "annotator1 as reference vs annotator2 on test half",
        },
        {
            "comparison": "qwen_judge_vs_gold",
            "n": qwen_main["n"],
            "kappa_clarification": qwen_main["kappa_clarification"],
            "kappa_clarification_lo": qwen_main["kappa_clarification_lo"],
            "kappa_clarification_hi": qwen_main["kappa_clarification_hi"],
            "kappa_category": qwen_main["kappa_category"],
            "kappa_category_lo": qwen_main["kappa_category_lo"],
            "kappa_category_hi": qwen_main["kappa_category_hi"],
            "sensitivity": qwen_main["sensitivity"],
            "specificity": qwen_main["specificity"],
            "note": "from results/qwen_expand6k_gold_validation.csv",
        },
        {
            "comparison": "gemma_judge_vs_gold",
            "n": gemma["n"],
            "kappa_clarification": gemma["kappa_clarification"],
            "kappa_clarification_lo": gemma["kappa_clarification_lo"],
            "kappa_clarification_hi": gemma["kappa_clarification_hi"],
            "kappa_category": gemma["kappa_category"],
            "kappa_category_lo": gemma["kappa_category_lo"],
            "kappa_category_hi": gemma["kappa_category_hi"],
            "sensitivity": gemma["sensitivity"],
            "specificity": gemma["specificity"],
            "note": "this run",
        },
    ]

    # Qwen vs Gemma on held-out 200 (both vs same items).
    qwen_gold_codes = ROOT / "outputs" / "codings_qwen_gold.jsonl"
    qwen_by_ann: dict[str, str] = {}
    if qwen_gold_codes.exists():
        with qwen_gold_codes.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("category") and splits.get(row.get("annotation_id")) == "test":
                    qwen_by_ann[row["annotation_id"]] = row["category"]
    gemma_by_ann = {row["annotation_id"]: row["category"] for row in test if row.get("category")}
    both_ids = sorted(set(qwen_by_ann) & set(gemma_by_ann))
    if both_ids:
        qg_pairs = [(qwen_by_ann[aid], gemma_by_ann[aid]) for aid in both_ids]
        qg_clar = binary_pairs(qg_pairs)
        qg_k = kappa_with_interval(qg_clar)
        qg_cat = kappa_with_interval(qg_pairs)
        table.append(
            {
                "comparison": "qwen_judge_vs_gemma_judge",
                "n": str(len(both_ids)),
                "kappa_clarification": f"{qg_k['value']:.4f}",
                "kappa_clarification_lo": f"{qg_k['lo']:.4f}",
                "kappa_clarification_hi": f"{qg_k['hi']:.4f}",
                "kappa_category": f"{qg_cat['value']:.4f}",
                "kappa_category_lo": f"{qg_cat['lo']:.4f}",
                "kappa_category_hi": f"{qg_cat['hi']:.4f}",
                "sensitivity": "",
                "specificity": "",
                "note": "inter-judge on test half where both parsed",
            }
        )
    else:
        # Fall back: re-use Gemma gold codes only and note Qwen gold file missing overlap.
        table.append(
            {
                "comparison": "qwen_judge_vs_gemma_judge",
                "n": "0",
                "kappa_clarification": "",
                "kappa_clarification_lo": "",
                "kappa_clarification_hi": "",
                "kappa_category": "",
                "kappa_category_lo": "",
                "kappa_category_hi": "",
                "sensitivity": "",
                "specificity": "",
                "note": "no overlapping Qwen gold test codes found in outputs/codings_qwen_gold.jsonl",
            }
        )

    atomic_csv(TABLE3_CSV, tuple(table[0].keys()), table)
    clar_kappa = float(gemma["kappa_clarification"])
    print(
        f"wrote {GOLD_CSV}; wrote {TABLE3_CSV}; "
        f"gemma test clar kappa {clar_kappa:.4f} gate {KAPPA_STOP}",
        flush=True,
    )
    return clar_kappa


def manifest_tasks(priority: int, pool: dict[str, dict]) -> list[dict]:
    rows = _read_csv(MANIFEST)
    selected = []
    for row in rows:
        listener = row["listener"]
        if priority == 1 and listener not in PRIORITY1_LISTENERS:
            continue
        if priority == 2 and listener not in PRIORITY2_LISTENERS:
            continue
        if priority == 0:
            # all non-self for completeness optional; priority 0 = 1+2+gemma self
            pass
        rid = reply_id(listener, row["condition"], row["item_id"])
        task = pool.get(rid)
        if task is None:
            raise SystemExit(f"manifest reply missing from pool: {rid}")
        selected.append(task)
    return selected


def run_code(*, priority: int, workers: int) -> None:
    if prompt_sha256_v2() != PINNED_HASH:
        raise SystemExit("precedence_v2 hash mismatch")
    system = system_prompt_v2()
    pool = load_pool()
    if priority not in (1, 2):
        raise SystemExit("priority must be 1 or 2")
    tasks = manifest_tasks(priority, pool)
    done = load_done_ids()
    pending = [task for task in tasks if reply_id(task["listener"], task["condition"], task["item_id"]) not in done]
    print(
        f"priority {priority} tasks {len(tasks)} pending {len(pending)} workers {workers}",
        flush=True,
    )
    if not pending:
        print("nothing pending", flush=True)
        return

    t0 = time.perf_counter()
    ok = 0
    failed = 0
    finished = 0

    def work(task: dict) -> dict:
        return judge_one(system, task)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool_ex:
        # Process in waves; after first 50 print throughput.
        index = 0
        wave = max(workers * 4, 8)
        while index < len(pending):
            batch = pending[index : index + wave]
            index += len(batch)
            futures = [pool_ex.submit(work, task) for task in batch]
            for fut in as_completed(futures):
                row = fut.result()
                with LOCK:
                    if row.get("category") and not row.get("failed"):
                        append_jsonl(CODINGS, row)
                        ok += 1
                    else:
                        append_jsonl(FAIL_LOG, row)
                        failed += 1
                    finished += 1
                    if finished == 50:
                        rate = (time.perf_counter() - t0) / 50
                        p1_n = sum(
                            1
                            for task in manifest_tasks(1, pool)
                            if reply_id(task["listener"], task["condition"], task["item_id"]) not in done
                        )
                        p2_n = sum(
                            1
                            for task in manifest_tasks(2, pool)
                            if reply_id(task["listener"], task["condition"], task["item_id"]) not in done
                        )
                        # Adjust pending remaining after these 50.
                        rem_p1 = max(0, p1_n - (50 if priority == 1 else 0))
                        rem_p2 = max(0, p2_n - (50 if priority == 2 else 0))
                        print(
                            json.dumps(
                                {
                                    "throughput_s_per_call": rate,
                                    "est_priority1_h": rem_p1 * rate / max(1, workers) / 3600,
                                    "est_priority2_h": rem_p2 * rate / max(1, workers) / 3600,
                                    "workers": workers,
                                }
                            ),
                            flush=True,
                        )
                    if finished % 20 == 0 or finished == len(pending):
                        left = (len(pending) - finished) * ((time.perf_counter() - t0) / max(1, finished))
                        print(
                            f"code {finished}/{len(pending)} ok {ok} fail {failed} eta_h {left/3600:.2f}",
                            flush=True,
                        )
    print(f"priority {priority} done ok {ok} fail {failed}", flush=True)


def load_qwen_codes() -> dict[str, dict]:
    out = {}
    with QWEN_CODINGS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("category") or row.get("failed"):
                continue
            rid = reply_id(row["listener"], row["condition"], row["item_id"])
            out[rid] = row
    return out


def load_gemma_codes() -> dict[str, dict]:
    out = {}
    if not CODINGS.exists():
        return out
    with CODINGS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("category") or row.get("failed"):
                continue
            out[row["reply_id"]] = row
    return out


def cell_array(
    codes: dict[str, dict],
    items: dict[str, dict],
    children: list[str],
    index: dict[str, int],
    listener: str,
    condition: str,
) -> tuple[np.ndarray, int]:
    arr = np.zeros((len(children), 4))
    n_coded = 0
    for rid, row in codes.items():
        if row["listener"] != listener or row["condition"] != condition:
            continue
        if row.get("exclude_from_conclusions"):
            continue
        item = items.get(row["item_id"])
        if item is None:
            continue
        n_coded += 1
        clar = row["category"] in CLARIFICATION
        kind = role(item)
        ci = index[item["child_id"] + "|" + item["corpus"]]
        if kind == "hit":
            arr[ci, 1] += 1
            arr[ci, 0] += bool(clar)
        elif kind == "fa":
            arr[ci, 3] += 1
            arr[ci, 2] += bool(clar)
    return arr, n_coded


def bootstrap_sdt(arr: np.ndarray, draws: list[np.ndarray]) -> dict | None:
    kh, nh, kf, nf = (int(x) for x in arr.sum(0))
    if nh == 0 or nf == 0:
        return None
    hit, fa, d_prime, c = sdt(kh, nh, kf, nf)
    hs, fs, ds, cs = [], [], [], []
    for idx in draws:
        total = arr[idx].sum(0)
        if total[1] == 0 or total[3] == 0:
            continue
        bh, bf, bd, bc = sdt(int(total[0]), int(total[1]), int(total[2]), int(total[3]))
        hs.append(bh)
        fs.append(bf)
        ds.append(bd)
        cs.append(bc)

    def ci(values: list[float]) -> tuple[float, float]:
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    return {
        "hit": hit,
        "fa": fa,
        "d": d_prime,
        "c": c,
        "hit_ci": ci(hs),
        "fa_ci": ci(fs),
        "d_ci": ci(ds),
        "c_ci": ci(cs),
        "n_unint": nh,
        "n_quiet": nf,
    }


def run_analyze() -> None:
    items = load_items()
    children = sorted({it["child_id"] + "|" + it["corpus"] for it in items.values()})
    index = {child: i for i, child in enumerate(children)}
    rng = np.random.default_rng(42)
    draws = [rng.integers(0, len(children), len(children)) for _ in range(B)]
    gemma_codes = load_gemma_codes()
    qwen_codes = load_qwen_codes()

    gold_val = _read_csv(GOLD_CSV)
    gemma_main = next(row for row in gold_val if row["row"] == "gemma_vs_gold_test")
    se = float(gemma_main["sensitivity"])
    sp = float(gemma_main["specificity"])

    # Cells present in gemma codes, excluding self-judged from conclusions tables.
    cells = sorted(
        {
            (row["listener"], row["condition"])
            for row in gemma_codes.values()
            if not row.get("exclude_from_conclusions")
        }
    )
    sdt_rows = []
    side_rows = []
    for listener, condition in cells:
        arr, n_coded = cell_array(gemma_codes, items, children, index, listener, condition)
        raw = bootstrap_sdt(arr, draws)
        if raw is None:
            continue
        # Rogan-Gladen with same Se/Sp for H and FA classes (held-out overall).
        corr = corrected_sdt(raw["hit"], raw["n_unint"], raw["fa"], raw["n_quiet"], se, sp, se, sp)
        hit_corr = rogan_gladen(raw["hit"], se, sp)
        fa_corr = rogan_gladen(raw["fa"], se, sp)
        dagger_h = hit_corr is not None and (hit_corr <= 0.0 or hit_corr >= 1.0)
        dagger_f = fa_corr is not None and (fa_corr <= 0.0 or fa_corr >= 1.0)
        withhold = dagger_h or dagger_f or corr is None
        sdt_rows.append(
            {
                "listener": listener,
                "condition": condition,
                "judge": MODEL,
                "n_coded": str(n_coded),
                "hit": f"{raw['hit']:.6f}",
                "fa": f"{raw['fa']:.6f}",
                "d": f"{raw['d']:.6f}",
                "c": f"{raw['c']:.6f}",
                "hit_lo": f"{raw['hit_ci'][0]:.6f}",
                "hit_hi": f"{raw['hit_ci'][1]:.6f}",
                "fa_lo": f"{raw['fa_ci'][0]:.6f}",
                "fa_hi": f"{raw['fa_ci'][1]:.6f}",
                "d_lo": f"{raw['d_ci'][0]:.6f}",
                "d_hi": f"{raw['d_ci'][1]:.6f}",
                "c_lo": f"{raw['c_ci'][0]:.6f}",
                "c_hi": f"{raw['c_ci'][1]:.6f}",
                "hit_corrected": "" if hit_corr is None else f"{hit_corr:.6f}",
                "fa_corrected": "" if fa_corr is None else f"{fa_corr:.6f}",
                "d_corrected": "" if withhold or corr is None else f"{corr[2]:.6f}",
                "c_corrected": "" if withhold or corr is None else f"{corr[3]:.6f}",
                "dagger": "yes" if withhold else "no",
                "n_unint": str(raw["n_unint"]),
                "n_quiet": str(raw["n_quiet"]),
            }
        )
        q_arr, q_n = cell_array(qwen_codes, items, children, index, listener, condition)
        q_raw = bootstrap_sdt(q_arr, draws)
        side_rows.append(
            {
                "listener": listener,
                "condition": condition,
                "n_gemma": str(n_coded),
                "n_qwen_overlap_cell": str(q_n),
                "qwen_hit": "" if q_raw is None else f"{q_raw['hit']:.6f}",
                "qwen_fa": "" if q_raw is None else f"{q_raw['fa']:.6f}",
                "qwen_d": "" if q_raw is None else f"{q_raw['d']:.6f}",
                "gemma_hit": f"{raw['hit']:.6f}",
                "gemma_fa": f"{raw['fa']:.6f}",
                "gemma_d": f"{raw['d']:.6f}",
            }
        )
    atomic_csv(SDT_CSV, tuple(sdt_rows[0].keys()) if sdt_rows else ("listener",), sdt_rows)
    atomic_csv(SIDE_CSV, tuple(side_rows[0].keys()) if side_rows else ("listener",), side_rows)

    # Contrasts with child bootstrap.
    def arr_of(listener: str, condition: str) -> np.ndarray:
        arr, _n = cell_array(gemma_codes, items, children, index, listener, condition)
        return arr

    contrast_rows = []

    def add_delta(name: str, a: np.ndarray, b: np.ndarray, metric_pair: str = "d_and_c") -> None:
        dd, dc = [], []
        for idx in draws:
            sa = a[idx].sum(0)
            sb = b[idx].sum(0)
            if sa[1] == 0 or sa[3] == 0 or sb[1] == 0 or sb[3] == 0:
                continue
            _, _, d1, c1 = sdt(int(sa[0]), int(sa[1]), int(sa[2]), int(sa[3]))
            _, _, d2, c2 = sdt(int(sb[0]), int(sb[1]), int(sb[2]), int(sb[3]))
            dd.append(d2 - d1)
            dc.append(c2 - c1)
        if not dd:
            return
        # Point from full arrays.
        sa = a.sum(0)
        sb = b.sum(0)
        _, _, d1, c1 = sdt(int(sa[0]), int(sa[1]), int(sa[2]), int(sa[3]))
        _, _, d2, c2 = sdt(int(sb[0]), int(sb[1]), int(sb[2]), int(sb[3]))
        contrast_rows.append(
            {
                "contrast": name,
                "delta_d": f"{(d2 - d1):.6f}",
                "delta_d_lo": f"{float(np.percentile(dd, 2.5)):.6f}",
                "delta_d_hi": f"{float(np.percentile(dd, 97.5)):.6f}",
                "delta_c": f"{(c2 - c1):.6f}",
                "delta_c_lo": f"{float(np.percentile(dc, 2.5)):.6f}",
                "delta_c_hi": f"{float(np.percentile(dc, 97.5)):.6f}",
            }
        )

    q_plain = arr_of("qwen/qwen3.8-27b", "plain")
    q_repair = arr_of("qwen/qwen3.8-27b", "repair")
    if q_plain.sum() > 0 and q_repair.sum() > 0:
        add_delta("qwen_repair_minus_plain", q_plain, q_repair)
    human = arr_of("human", "human")
    for cond in ("plain", "child", "repair"):
        q = arr_of("qwen/qwen3.8-27b", cond)
        if q.sum() > 0 and human.sum() > 0:
            add_delta(f"qwen_{cond}_minus_adults", human, q)
    for listener in sorted(PRIORITY2_LISTENERS):
        for cond in ("plain", "child", "repair"):
            arr = arr_of(listener, cond)
            if arr.sum() > 0 and human.sum() > 0:
                add_delta(f"{listener}|{cond}_minus_adults", human, arr)

    # Self-preference style: clar rate Qwen-judge minus Gemma-judge by source.
    sources = sorted({row["listener"] for row in gemma_codes.values() if not row.get("exclude_from_conclusions")})
    agree_rows = []
    for source in sources:
        pairs = []
        q_clar_flags = []
        g_clar_flags = []
        for rid, grow in gemma_codes.items():
            if grow["listener"] != source:
                continue
            qrow = qwen_codes.get(rid)
            if qrow is None:
                continue
            pairs.append((qrow["category"], grow["category"]))
            q_clar_flags.append(qrow["category"] in CLARIFICATION)
            g_clar_flags.append(grow["category"] in CLARIFICATION)
        if not pairs:
            continue
        clar_pairs = binary_pairs(pairs)
        k = kappa_with_interval(clar_pairs)
        agree = sum(a == b for a, b in clar_pairs) / len(clar_pairs)
        q_rate = sum(q_clar_flags) / len(q_clar_flags)
        g_rate = sum(g_clar_flags) / len(g_clar_flags)
        # Bootstrap difference in clarification rates (item level).
        rng2 = np.random.default_rng(42)
        diffs = []
        n = len(q_clar_flags)
        for _ in range(B):
            idx = rng2.integers(0, n, n)
            diffs.append(
                sum(q_clar_flags[i] for i in idx) / n - sum(g_clar_flags[i] for i in idx) / n
            )
        agree_rows.append(
            {
                "source": source,
                "n": str(len(pairs)),
                "kappa_clarification": f"{k['value']:.4f}",
                "kappa_lo": f"{k['lo']:.4f}",
                "kappa_hi": f"{k['hi']:.4f}",
                "pct_agree_clarification": f"{agree:.4f}",
                "qwen_judge_clar_rate": f"{q_rate:.4f}",
                "gemma_judge_clar_rate": f"{g_rate:.4f}",
                "qwen_minus_gemma_clar_rate": f"{(q_rate - g_rate):.4f}",
                "diff_lo": f"{float(np.percentile(diffs, 2.5)):.4f}",
                "diff_hi": f"{float(np.percentile(diffs, 97.5)):.4f}",
            }
        )
    atomic_csv(AGREE_CSV, tuple(agree_rows[0].keys()) if agree_rows else ("source",), agree_rows)
    atomic_csv(CONTRAST_CSV, tuple(contrast_rows[0].keys()) if contrast_rows else ("contrast",), contrast_rows)
    print(f"wrote {SDT_CSV} {SIDE_CSV} {AGREE_CSV} {CONTRAST_CSV}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", action="store_true")
    parser.add_argument("--code", action="store_true")
    parser.add_argument("--priority", type=int, default=1, choices=[1, 2])
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--force-gate",
        action="store_true",
        help="Continue coding even if held-out clarification kappa is below 0.40",
    )
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.gold:
        kappa = run_gold(workers=max(1, args.workers))
        if kappa < KAPPA_STOP and not args.force_gate:
            print(f"STOP: Gemma clarification kappa {kappa:.4f} < {KAPPA_STOP}", flush=True)
            return 3
        if kappa < KAPPA_STOP and args.force_gate:
            print(
                f"GATE_OVERRIDE kappa {kappa:.4f} < {KAPPA_STOP}; continuing by request",
                flush=True,
            )
        else:
            print("GATE_PASS", flush=True)
    if args.code:
        # Require gate pass file existence via gold csv kappa.
        if not GOLD_CSV.exists():
            raise SystemExit("run --gold first")
        gemma_main = next(row for row in _read_csv(GOLD_CSV) if row["row"] == "gemma_vs_gold_test")
        if float(gemma_main["kappa_clarification"]) < KAPPA_STOP and not args.force_gate:
            print("STOP: gate failed; not coding", flush=True)
            return 3
        if float(gemma_main["kappa_clarification"]) < KAPPA_STOP and args.force_gate:
            print(
                f"GATE_OVERRIDE kappa {gemma_main['kappa_clarification']} < {KAPPA_STOP}; coding by request",
                flush=True,
            )
        run_code(priority=args.priority, workers=max(1, args.workers))
    if args.analyze:
        run_analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
