"""Recall-based reward for SearchAgent smoke task.

Parses the finalize_ranking tool_call from the final assistant turn and
scores predicted chunk_ids against ground truth from the dataset.

Score = 0.9 * recall + 0.1 * precision  (precision is a light tiebreaker
to discourage dumping the whole corpus into the ranking).
"""

import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FN_CALL_RE = re.compile(r"<function=finalize_ranking>\s*(.*?)\s*</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _extract_chunks(text: str) -> list[str]:
    """Return chunk_ids from the finalize_ranking call (qwen3_coder XML format)."""
    text = _THINK_RE.sub("", text)
    for m in _FN_CALL_RE.finditer(text):
        for pname, pval in _PARAM_RE.findall(m.group(1)):
            if pname != "ranking":
                continue
            try:
                ranking = json.loads(pval)
            except json.JSONDecodeError:
                continue
            if isinstance(ranking, list):
                return [r["chunk_id"] for r in ranking if isinstance(r, dict) and "chunk_id" in r]
    return []

def _normalize_gt(gt: Any) -> set[str]:
    """Coerce ground_truth (list / np.ndarray / JSON string / single id) to a set of ids."""
    if hasattr(gt, "tolist"):
        gt = gt.tolist()
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except json.JSONDecodeError:
            gt = [gt]
    return set(gt or [])


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    predicted = set(_extract_chunks(solution_str))
    gt = _normalize_gt(ground_truth)

    tp = len(predicted & gt)
    recall = tp / len(gt) if gt else 0.0
    precision = tp / len(predicted) if predicted else 0.0
    score = 0.9 * recall + 0.1 * precision

    return {
        "score": float(score),
        "recall": float(recall),
        "precision": float(precision),
        "finalized": float(bool(predicted)),
        "num_predicted": float(len(predicted)),
        "num_ground_truth": float(len(gt)),
        "num_turns": float((extra_info or {}).get("num_turns", 0)),
    }