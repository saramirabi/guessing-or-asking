"""Analyse self-preference pack after two annotators + adjudication.

Does not invent labels. Expects:
  results/second_judge/self_preference_labels_annotator1.csv
  results/second_judge/self_preference_labels_annotator2.csv
  results/second_judge/self_preference_gold_final.csv
each with annotation_id, category (and optional corrects_error).

    python scripts/second_judge/analyse_self_preference.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agreement import atomic_csv  # noqa: E402
from scripts.code_replies import CLARIFICATION  # noqa: E402
from scripts.rater_metrics import sensitivity_specificity  # noqa: E402
from scripts.second_judge.gemma_common import CODINGS, OUT, reply_id  # noqa: E402

KEY = OUT / "self_preference_key.csv"
A1 = OUT / "self_preference_labels_annotator1.csv"
A2 = OUT / "self_preference_labels_annotator2.csv"
GOLD = OUT / "self_preference_gold_final.csv"
QWEN_CODINGS = ROOT / "outputs" / "codings_qwen_expand6k.jsonl"
REPORT = OUT / "self_preference_analysis.csv"
B = 1000
SEED = 42


def _read(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return {
        row["annotation_id"]: row["category"]
        for row in csv.DictReader(path.open(encoding="utf-8-sig", newline=""))
        if row.get("category")
    }


def _load_judge(path: Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("category") and not row.get("failed"):
                rid = row.get("reply_id") or reply_id(row["listener"], row["condition"], row["item_id"])
                out[rid] = row["category"]
    return out


def se_sp(gold: list[str], pred: list[str]) -> tuple[float | None, float | None]:
    pairs = [
        ("yes" if g in CLARIFICATION else "no", "yes" if p in CLARIFICATION else "no")
        for g, p in zip(gold, pred)
    ]
    return sensitivity_specificity(pairs)


def main() -> int:
    for path in (KEY, A1, A2, GOLD):
        if not path.exists():
            raise SystemExit(
                "Labels not ready. Add annotator CSVs and adjudicated gold under "
                f"{OUT}, then re-run. Missing: {path.name}"
            )
    key_rows = list(csv.DictReader(KEY.open(encoding="utf-8", newline="")))
    gold = _read(GOLD)
    qwen_judge = _load_judge(QWEN_CODINGS)
    gemma_judge = _load_judge(CODINGS)

    qwen_items = [r for r in key_rows if r["listener"] == "qwen/qwen3.8-27b" and r["annotation_id"] in gold]
    other_items = [r for r in key_rows if r["listener"] != "qwen/qwen3.8-27b" and r["annotation_id"] in gold]

    def block(rows: list[dict], judge_map: dict[str, str], label: str) -> dict:
        g_list, p_list = [], []
        for row in rows:
            pred = judge_map.get(row["reply_id"])
            if pred is None:
                continue
            g_list.append(gold[row["annotation_id"]])
            p_list.append(pred)
        se, sp = se_sp(g_list, p_list)
        return {"subset": label, "n": str(len(g_list)), "sensitivity": se, "specificity": sp, "g": g_list, "p": p_list}

    rows_out = []
    for judge_name, judge_map in (("qwen_judge", qwen_judge), ("gemma_judge", gemma_judge)):
        q = block(qwen_items, judge_map, f"{judge_name}_on_qwen_replies")
        o = block(other_items, judge_map, f"{judge_name}_on_non_qwen_replies")
        # Bootstrap Se difference (Qwen replies minus non-Qwen).
        rng = np.random.default_rng(SEED)
        se_diffs, sp_diffs = [], []
        if q["g"] and o["g"]:
            for _ in range(B):
                qi = rng.integers(0, len(q["g"]), len(q["g"]))
                oi = rng.integers(0, len(o["g"]), len(o["g"]))
                qg = [q["g"][i] for i in qi]
                qp = [q["p"][i] for i in qi]
                og = [o["g"][i] for i in oi]
                op = [o["p"][i] for i in oi]
                qse, qsp = se_sp(qg, qp)
                ose, osp = se_sp(og, op)
                if qse is not None and ose is not None:
                    se_diffs.append(qse - ose)
                if qsp is not None and osp is not None:
                    sp_diffs.append(qsp - osp)
        for block_row in (q, o):
            rows_out.append(
                {
                    "judge": judge_name,
                    "subset": block_row["subset"],
                    "n": block_row["n"],
                    "sensitivity": "" if block_row["sensitivity"] is None else f"{block_row['sensitivity']:.4f}",
                    "specificity": "" if block_row["specificity"] is None else f"{block_row['specificity']:.4f}",
                    "se_diff_qwen_minus_other": "",
                    "se_diff_lo": "",
                    "se_diff_hi": "",
                    "sp_diff_qwen_minus_other": "",
                    "sp_diff_lo": "",
                    "sp_diff_hi": "",
                }
            )
        rows_out.append(
            {
                "judge": judge_name,
                "subset": "difference_qwen_minus_other",
                "n": "",
                "sensitivity": "",
                "specificity": "",
                "se_diff_qwen_minus_other": ""
                if not se_diffs
                else f"{(q['sensitivity'] or 0) - (o['sensitivity'] or 0):.4f}",
                "se_diff_lo": "" if not se_diffs else f"{float(np.percentile(se_diffs, 2.5)):.4f}",
                "se_diff_hi": "" if not se_diffs else f"{float(np.percentile(se_diffs, 97.5)):.4f}",
                "sp_diff_qwen_minus_other": ""
                if not sp_diffs
                else f"{(q['specificity'] or 0) - (o['specificity'] or 0):.4f}",
                "sp_diff_lo": "" if not sp_diffs else f"{float(np.percentile(sp_diffs, 2.5)):.4f}",
                "sp_diff_hi": "" if not sp_diffs else f"{float(np.percentile(sp_diffs, 97.5)):.4f}",
            }
        )
    atomic_csv(REPORT, tuple(rows_out[0].keys()), rows_out)
    print(f"wrote {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
