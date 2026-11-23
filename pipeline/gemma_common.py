"""Shared paths and LM Studio call for the Gemma 4 second judge."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "second_judge"
MODEL = "google/gemma-4-26b-a4b-qat"
PINNED_HASH = "26166c9693f14222254ef5bf58d59eba19060a6791e5f95154ccb57256ba8826"
MAX_TOKENS = 1024
CACHE = OUT / "gemma_judge_cache"
CODINGS = OUT / "gemma_judge_codes.jsonl"
FAIL_LOG = OUT / "gemma_judge_failures.jsonl"
GOLD_CSV = OUT / "gemma_gold_validation.csv"
TABLE3_CSV = OUT / "table3_judge_validation.csv"
SDT_CSV = OUT / "gemma_sdt.csv"
SIDE_CSV = OUT / "qwen_vs_gemma_sdt.csv"
AGREE_CSV = OUT / "judge_agreement_by_source.csv"
CONTRAST_CSV = OUT / "gemma_contrasts.csv"

from judge.codebook_v2 import PROMPT_VERSION_V2  # noqa: E402
from scripts.code_replies import parse_reply_judgment  # noqa: E402
from scripts.generate import load_config  # noqa: E402


def reply_id(listener: str, condition: str, item_id: str) -> str:
    return f"{listener}|{condition}|{item_id}"


def cache_key(listener: str, condition: str, item_id: str, context: str, reply: str) -> str:
    raw = "|".join(
        [
            MODEL,
            PROMPT_VERSION_V2,
            PINNED_HASH,
            reply_id(listener, condition, item_id),
            context,
            reply,
            "0",
            str(MAX_TOKENS),
            "42",
            "reasoning_none",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def call_gemma(system: str, user: str, *, allow_extra: bool = True) -> tuple[str, dict]:
    config = load_config(ROOT / "config.yaml")
    base = str(config.get("lmstudio_base_url", "http://localhost:1234/v1")).rstrip("/")
    body: dict = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if allow_extra:
        # Same LM Studio thinking-off settings as the Qwen judge run.
        body["reasoning_effort"] = "none"
        body["enable_thinking"] = False
        body["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if allow_extra and exc.code == 400:
            return call_gemma(system, user, allow_extra=False)
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = (message.get("content") or "").strip()
    if not text:
        reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
        start = reasoning.rfind("{")
        end = reasoning.rfind("}")
        if start >= 0 and end > start:
            candidate = reasoning[start : end + 1].strip()
            if parse_reply_judgment(candidate) is not None:
                text = candidate
    return text, payload.get("usage") or {}


def load_done_ids(path: Path = CODINGS) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("category") and not row.get("failed"):
                done.add(row["reply_id"])
    return done


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
