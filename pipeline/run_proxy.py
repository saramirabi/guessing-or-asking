"""Step 1: judge-free explicit-repair proxy for five listeners + adults.

No LLM calls. Imports sdt / bootstrap definitions from scripts.analyze_full.
Writes results/second_judge/proxy_table.csv and proxy_table.md only.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agreement import atomic_csv  # noqa: E402
from scripts.analyze_full import B, load_items, role, sdt  # noqa: E402
from scripts.repair_proxy import REPAIR_RE  # noqa: E402

OUT_DIR = ROOT / "results" / "second_judge"
CSV_PATH = OUT_DIR / "proxy_table.csv"
MD_PATH = OUT_DIR / "proxy_table.md"
GEN_FILE = ROOT / "outputs" / "generations_full_lmstudio.jsonl"

MODELS = (
    "google/gemma-4-26b-a4b-qat",
    "qwen/qwen3.8-27b",
    "mistral-small-3.1-24b-instruct-2503",
    "unsloth/gpt-oss-20b",
    "llama-3.2-3b-instruct",
)
CONDITIONS = ("plain", "child", "repair")


def load_responses(items: dict[str, dict]) -> dict[tuple[str, str], dict[str, str]]:
    responses: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for item in items.values():
        responses[("human", "human")][item["item_id"]] = item.get("human_next_turn") or ""
    with GEN_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = row.get("item_id")
            if item_id not in items:
                continue
            model = row.get("model")
            if model not in MODELS:
                continue
            if row.get("outcome") == "blocked":
                continue
            cond = row.get("prompt_condition")
            if cond not in CONDITIONS:
                continue
            responses[(model, cond)][item_id] = row.get("reply") or ""
    return responses


def per_child_proxy(
    items: dict[str, dict],
    replies: dict[str, str],
    children: list[str],
    child_index: dict[str, int],
) -> np.ndarray:
    arr = np.zeros((len(children), 4))
    for item_id, reply in replies.items():
        item = items.get(item_id)
        if item is None:
            continue
        kind = role(item)
        if kind is None:
            continue
        ci = child_index[item["child_id"] + "|" + item["corpus"]]
        hit = bool(REPAIR_RE.search(reply or ""))
        if kind == "hit":
            arr[ci, 1] += 1
            arr[ci, 0] += hit
        else:
            arr[ci, 3] += 1
            arr[ci, 2] += hit
    return arr


def summarise(arr: np.ndarray, boot: list[np.ndarray]) -> dict:
    kh, nh, kf, nf = (int(x) for x in arr.sum(0))
    hit, fa, d_prime, c = sdt(kh, nh, kf, nf)
    hs, fs, ds, cs = [], [], [], []
    for idx in boot:
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
        "n_unint": nh,
        "n_quiet": nf,
        "hit": hit,
        "hit_ci": ci(hs),
        "fa": fa,
        "fa_ci": ci(fs),
        "d": d_prime,
        "d_ci": ci(ds),
        "c": c,
        "c_ci": ci(cs),
        "arr": arr,
    }


def fmt_pct(rate: float, lo: float, hi: float) -> str:
    return f"{100 * rate:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def fmt_num(value: float, lo: float, hi: float) -> str:
    return f"{value:.2f} [{lo:.2f}, {hi:.2f}]"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_items()
    responses = load_responses(items)
    children = sorted({it["child_id"] + "|" + it["corpus"] for it in items.values()})
    child_index = {child: i for i, child in enumerate(children)}
    rng = np.random.default_rng(42)
    boot = [rng.integers(0, len(children), len(children)) for _ in range(B)]

    keys = [("human", "human")] + [(model, cond) for model in MODELS for cond in CONDITIONS]
    rows_out = []
    stats: dict[tuple[str, str], dict] = {}
    for key in keys:
        replies = responses.get(key) or {}
        arr = per_child_proxy(items, replies, children, child_index)
        block = summarise(arr, boot)
        stats[key] = block
        listener, condition = key
        rows_out.append(
            {
                "record": "cell",
                "listener": listener,
                "condition": condition,
                "n_replies": str(len(replies)),
                "n_unint": str(block["n_unint"]),
                "n_quiet": str(block["n_quiet"]),
                "hit": f"{block['hit']:.6f}",
                "hit_lo": f"{block['hit_ci'][0]:.6f}",
                "hit_hi": f"{block['hit_ci'][1]:.6f}",
                "fa": f"{block['fa']:.6f}",
                "fa_lo": f"{block['fa_ci'][0]:.6f}",
                "fa_hi": f"{block['fa_ci'][1]:.6f}",
                "d": f"{block['d']:.6f}",
                "d_lo": f"{block['d_ci'][0]:.6f}",
                "d_hi": f"{block['d_ci'][1]:.6f}",
                "c": f"{block['c']:.6f}",
                "c_lo": f"{block['c_ci'][0]:.6f}",
                "c_hi": f"{block['c_ci'][1]:.6f}",
                "delta_d": "",
                "delta_d_lo": "",
                "delta_d_hi": "",
                "delta_c": "",
                "delta_c_lo": "",
                "delta_c_hi": "",
                "note": "proxy misses bare candidate checks; lower bound on clarification",
            }
        )

    for model in MODELS:
        plain = stats[(model, "plain")]
        repair = stats[(model, "repair")]
        dd, dc = [], []
        for idx in boot:
            p = plain["arr"][idx].sum(0)
            r = repair["arr"][idx].sum(0)
            if p[1] == 0 or p[3] == 0 or r[1] == 0 or r[3] == 0:
                continue
            _, _, d1, c1 = sdt(int(p[0]), int(p[1]), int(p[2]), int(p[3]))
            _, _, d2, c2 = sdt(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
            dd.append(d2 - d1)
            dc.append(c2 - c1)
        delta_d = repair["d"] - plain["d"]
        delta_c = repair["c"] - plain["c"]
        rows_out.append(
            {
                "record": "repair_minus_plain",
                "listener": model,
                "condition": "repair_minus_plain",
                "n_replies": "",
                "n_unint": "",
                "n_quiet": "",
                "hit": "",
                "hit_lo": "",
                "hit_hi": "",
                "fa": "",
                "fa_lo": "",
                "fa_hi": "",
                "d": "",
                "d_lo": "",
                "d_hi": "",
                "c": "",
                "c_lo": "",
                "c_hi": "",
                "delta_d": f"{delta_d:.6f}",
                "delta_d_lo": f"{float(np.percentile(dd, 2.5)):.6f}",
                "delta_d_hi": f"{float(np.percentile(dd, 97.5)):.6f}",
                "delta_c": f"{delta_c:.6f}",
                "delta_c_lo": f"{float(np.percentile(dc, 2.5)):.6f}",
                "delta_c_hi": f"{float(np.percentile(dc, 97.5)):.6f}",
                "note": "key contrast for Qwen3.8 when listener is qwen/qwen3.8-27b",
            }
        )

    fields = list(rows_out[0].keys())
    atomic_csv(CSV_PATH, fields, rows_out)

    lines = [
        "# Judge-free proxy (Step 1)",
        "",
        "Explicit repair wording regex from `scripts/repair_proxy.py`. "
        "Hit = proxy at unintelligible turns; FA = proxy at quiet clear baseline. "
        f"Log-linear correction at 0/1 rates. Child bootstrap B={B}, seed 42. "
        "The proxy misses bare candidate checks, so it is a lower bound on clarification.",
        "",
        f"Items: {len(items)}. Children: {len(children)}.",
        "",
        "## Cells",
        "",
        "| Listener | Condition | n replies | H % [CI] | F % [CI] | d' [CI] | c [CI] |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in keys:
        block = stats[key]
        lines.append(
            "| {listener} | {cond} | {n} | {h} | {f} | {d} | {c} |".format(
                listener=key[0],
                cond=key[1],
                n=len(responses.get(key) or {}),
                h=fmt_pct(block["hit"], *block["hit_ci"]),
                f=fmt_pct(block["fa"], *block["fa_ci"]),
                d=fmt_num(block["d"], *block["d_ci"]),
                c=fmt_num(block["c"], *block["c_ci"]),
            )
        )
    lines.extend(
        [
            "",
            "## Repair minus plain",
            "",
            "| Model | Δd' [CI] | Δc [CI] |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in rows_out:
        if row["record"] != "repair_minus_plain":
            continue
        lines.append(
            f"| {row['listener']} | {float(row['delta_d']):.2f} "
            f"[{float(row['delta_d_lo']):.2f}, {float(row['delta_d_hi']):.2f}] | "
            f"{float(row['delta_c']):.2f} "
            f"[{float(row['delta_c_lo']):.2f}, {float(row['delta_c_hi']):.2f}] |"
        )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {CSV_PATH}", flush=True)
    print(f"wrote {MD_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
