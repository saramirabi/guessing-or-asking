"""Step 3: build self-preference annotation pack (no labels filled).

    python scripts/second_judge/build_self_preference_pack.py
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
from scripts.scoped_judge import scoped_tasks  # noqa: E402
from scripts.second_judge.gemma_common import OUT, reply_id  # noqa: E402

MANIFEST = ROOT / "results" / "qwen_judge_expand6k_manifest.csv"
GOLD_FINAL = ROOT / "annotation" / "gold_final.csv"
GOLD_ITEMS = ROOT / "annotation" / "gold_items.csv"
SHEET = OUT / "self_preference_items.csv"
KEY = OUT / "self_preference_key.csv"
SEED = 42


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gold_pairs = {
        (row["source"] if row["source"] != "human" else "human",
         row["condition"] if row["source"] != "human" else "human",
         row["item_id"])
        for row in csv.DictReader(GOLD_FINAL.open(encoding="utf-8-sig", newline=""))
    }
    # Also block by annotation reply text match via gold items? key by source+condition+item is enough.
    meta = {}
    with (ROOT / "data" / "processed" / "items_v2.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            meta[row["item_id"]] = row

    pool, _counts = scoped_tasks(ROOT)
    by_rid = {reply_id(t["listener"], t["condition"], t["item_id"]): t for t in pool}

    manifest = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    qwen_cand = []
    other_cand = []
    for row in manifest:
        listener = row["listener"]
        condition = row["condition"]
        item_id = row["item_id"]
        if (listener, condition, item_id) in gold_pairs:
            continue
        rid = reply_id(listener, condition, item_id)
        task = by_rid.get(rid)
        if task is None:
            continue
        item = meta.get(item_id) or {}
        trouble = item.get("trouble_class") or "none"
        if trouble == "none" and item.get("recent_trouble") is False:
            turn = "quiet_baseline"
        elif trouble == "unintelligible":
            turn = "unintelligible"
        elif trouble == "error":
            turn = "error"
        else:
            turn = "other"
        rec = {
            "reply_id": rid,
            "listener": listener,
            "condition": condition,
            "item_id": item_id,
            "turn_type": turn,
            "context": task["context"],
            "reply": task["reply"],
        }
        if listener == "qwen/qwen3.8-27b":
            qwen_cand.append(rec)
        elif listener != "google/gemma-4-26b-a4b-qat":
            other_cand.append(rec)

    rng = np.random.default_rng(SEED)
    # Target ~34 unintelligible, 33 quiet, 33 error, spread over plain/child/repair.
    selected_qwen = []
    for turn, n_take in (("unintelligible", 34), ("quiet_baseline", 33), ("error", 33)):
        pool_t = [r for r in qwen_cand if r["turn_type"] == turn]
        rng.shuffle(pool_t)
        # balance conditions
        by_cond = {c: [r for r in pool_t if r["condition"] == c] for c in ("plain", "child", "repair")}
        take = []
        i = 0
        while len(take) < n_take:
            progressed = False
            for cond in ("plain", "child", "repair"):
                if len(take) >= n_take:
                    break
                if i < len(by_cond[cond]):
                    take.append(by_cond[cond][i])
                    progressed = True
            if not progressed:
                break
            i += 1
        selected_qwen.extend(take[:n_take])

    rng.shuffle(other_cand)
    selected_other = other_cand[:50]
    if len(selected_qwen) < 100:
        raise SystemExit(f"only {len(selected_qwen)} Qwen non-gold candidates available")
    if len(selected_other) < 50:
        raise SystemExit(f"only {len(selected_other)} other non-gold candidates available")

    pack = selected_qwen[:100] + selected_other
    rng.shuffle(pack)

    # Sheet in gold_items format + empty label columns.
    sheet_rows = []
    key_rows = []
    for i, rec in enumerate(pack, start=1):
        ann_id = f"sp{i:03d}"
        sheet_rows.append(
            {
                "annotation_id": ann_id,
                "context": rec["context"],
                "reply": rec["reply"],
                "category": "",
                "corrects_error": "",
                "note": "",
            }
        )
        key_rows.append(
            {
                "annotation_id": ann_id,
                "reply_id": rec["reply_id"],
                "listener": rec["listener"],
                "condition": rec["condition"],
                "item_id": rec["item_id"],
                "turn_type": rec["turn_type"],
            }
        )
    atomic_csv(
        SHEET,
        ("annotation_id", "context", "reply", "category", "corrects_error", "note"),
        sheet_rows,
    )
    atomic_csv(
        KEY,
        ("annotation_id", "reply_id", "listener", "condition", "item_id", "turn_type"),
        key_rows,
    )
    print(f"wrote {SHEET} n={len(sheet_rows)}", flush=True)
    print(f"wrote {KEY}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
