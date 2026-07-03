"""Strict OVEN reward for responses ending in ``\\boxed{}``.

This reward intentionally avoids taxonomy hP/hR/hF as a training signal.  It is
boxed-format gated and gives full credit only to exact ground-truth labels or
known aliases.  Optional alias support is loaded from ``OVEN_TAXONOMY_INDEX``.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

BOXED_WRONG_REWARD = 0.05
CORRECT_REWARD = 1.0


def _strip_latex_answer(answer: str) -> str:
    answer = answer.strip().strip("$").strip().strip(".").strip()
    wrappers = (r"\text", r"\mathrm", r"\operatorname", r"\mathbf")
    changed = True
    while changed:
        changed = False
        for wrapper in wrappers:
            prefix = wrapper + "{"
            if answer.startswith(prefix) and answer.endswith("}"):
                answer = answer[len(prefix):-1].strip()
                changed = True
    return answer.strip().strip("$").strip().strip(".").strip()


def extract_boxed_answer(text: str) -> tuple[str, bool]:
    matches: list[str] = []
    start = 0
    needle = r"\boxed{"
    while True:
        box_start = text.find(needle, start)
        if box_start < 0:
            break
        content_start = box_start + len(needle)
        depth = 1
        idx = content_start
        while idx < len(text) and depth > 0:
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            answer = _strip_latex_answer(text[content_start:idx - 1])
            if answer:
                matches.append(answer)
            start = idx
        else:
            break
    if matches:
        return matches[-1], True
    return text.strip(), False


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _load_alias_index() -> tuple[dict[str, set[str]], dict[str, str]]:
    path = os.environ.get("OVEN_TAXONOMY_INDEX")
    if not path:
        return {}, {}

    index_path = Path(path)
    if not index_path.exists():
        raise FileNotFoundError(f"OVEN_TAXONOMY_INDEX does not exist: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    aliases_by_canonical: dict[str, set[str]] = {}
    canonical_by_alias: dict[str, str] = {}
    for alias, canonical in index.get("aliases", {}).items():
        alias_norm = normalize_answer(str(alias))
        canonical_norm = normalize_answer(str(canonical))
        if not alias_norm or not canonical_norm:
            continue
        aliases_by_canonical.setdefault(canonical_norm, set()).add(alias_norm)
        canonical_by_alias.setdefault(alias_norm, canonical_norm)
    return aliases_by_canonical, canonical_by_alias


def _extra_value(extra_info: dict | None, key: str) -> str:
    if not extra_info:
        return ""
    value = extra_info.get(key)
    return str(value or "")


def _taxonomy_leaf(extra_info: dict | None) -> str:
    if not extra_info:
        return ""
    labels = extra_info.get("taxonomy_labels") or []
    if isinstance(labels, (list, tuple)) and labels:
        return str(labels[0] or "")
    return ""


def valid_answer_norms(ground_truth: str, extra_info: dict | None = None) -> set[str]:
    answers = {
        normalize_answer(value)
        for value in (
            ground_truth,
            _extra_value(extra_info, "answer"),
            _extra_value(extra_info, "entity_text"),
            _taxonomy_leaf(extra_info),
        )
        if str(value or "").strip()
    }
    answers.discard("")

    aliases_by_canonical, canonical_by_alias = _load_alias_index()

    expanded = set(answers)
    for answer in list(answers):
        canonical = canonical_by_alias.get(answer)
        if canonical:
            expanded.add(canonical)

    for answer in list(expanded):
        expanded.update(aliases_by_canonical.get(answer, set()))

    expanded.discard("")
    return expanded


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    del data_source
    prediction, parse_ok = extract_boxed_answer(solution_str)
    if not parse_ok:
        return 0.0

    if normalize_answer(prediction) in valid_answer_norms(ground_truth, extra_info):
        return CORRECT_REWARD
    return BOXED_WRONG_REWARD
