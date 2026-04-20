# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary-answer classification reward with structured think-block scoring.

v3 reward components (max 1.0 per response):
  format   0.30  - gated nested ``<think>`` structure plus quality checks
  answer   0.70  - whole-token substring match after ``</think>``

Expected response format:
  <think>
    <HasProperty>tag1, tag2</HasProperty>
    <HasA>tag1, tag2</HasA>
    <AtLocation>tag1, tag2</AtLocation>
  </think>
  label or natural-language continuation

No per-class metadata is used. The reward reads only ``ground_truth``.
spaCy is required at reward time for the format-quality POS sanity check.
"""

from __future__ import annotations

import re
from functools import lru_cache

FORMAT_MAX = 0.30
ANSWER_MAX = 0.70
RELATIONS = ("HasProperty", "HasA", "AtLocation")
MIN_ENTRIES_PER_TAG = 2
MAX_TOKENS_PER_ENTRY = 3

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_GATE_PATTERN = re.compile(
    r"<think>\s*"
    r".*?<HasProperty>(.+?)</HasProperty>\s*"
    r".*?<HasA>(.+?)</HasA>\s*"
    r".*?<AtLocation>(.+?)</AtLocation>\s*"
    r".*?</think>",
    re.DOTALL,
)
_RELATION_PATTERNS = {
    rel: re.compile(rf"<{rel}>(.*?)</{rel}>", re.DOTALL) for rel in RELATIONS
}


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def _whole_token_substring(needle: str, haystack: str) -> bool:
    if not needle:
        return False
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])")
    return bool(pattern.search(haystack))


def _split_entries(block: str) -> list[str]:
    return [_normalise(part) for part in block.split(",") if _normalise(part)]


def _extract_relation_entries(think_body: str, relation: str) -> list[str]:
    match = _RELATION_PATTERNS[relation].search(think_body)
    if not match:
        return []
    return _split_entries(match.group(1))


def _extract_post_think_content(solution_str: str) -> str:
    match = _THINK_PATTERN.search(solution_str)
    if not match:
        return solution_str.strip()
    return solution_str[match.end() :].strip()


def _format_gate_passes(solution_str: str) -> bool:
    return bool(_THINK_GATE_PATTERN.search(solution_str))


def _min_content_score(entries_by_relation: dict[str, list[str]]) -> float:
    scores = []
    for relation in RELATIONS:
        entries = entries_by_relation[relation]
        ok = len(entries) >= MIN_ENTRIES_PER_TAG and all(len(entry) >= 2 for entry in entries)
        scores.append(1.0 if ok else 0.0)
    return sum(scores) / len(scores)


def _dedup_score(entries_by_relation: dict[str, list[str]]) -> float:
    scores = []
    for relation in RELATIONS:
        entries = entries_by_relation[relation]
        scores.append(1.0 if len(entries) == len(set(entries)) else 0.0)
    return sum(scores) / len(scores)


@lru_cache(maxsize=1)
def _get_nlp():
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "classification.py reward v3 requires spaCy at reward time. "
            "Install `spacy` and the `en_core_web_sm` model."
        ) from exc

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise OSError(
            "classification.py reward v3 requires the spaCy model `en_core_web_sm`. "
            "Install it with `python -m spacy download en_core_web_sm`."
        ) from exc


def _entry_passes_pos(entry: str, relation: str) -> float:
    nlp = _get_nlp()
    doc = nlp(entry)
    tokens = [tok for tok in doc if not tok.is_space and not tok.is_punct]
    if not tokens or len(tokens) > MAX_TOKENS_PER_ENTRY:
        return 0.0

    root = next((tok for tok in tokens if tok.dep_ == "ROOT"), tokens[0])
    if any(tok.pos_ in {"VERB", "AUX"} for tok in tokens):
        return 0.0

    if relation == "HasProperty":
        return 1.0 if root.pos_ in {"ADJ", "NOUN", "PROPN"} else 0.0
    return 1.0 if root.pos_ in {"NOUN", "PROPN"} else 0.0


def _pos_sanity_score(entries_by_relation: dict[str, list[str]]) -> float:
    relation_scores = []
    for relation in RELATIONS:
        entries = entries_by_relation[relation]
        if not entries:
            relation_scores.append(0.0)
            continue
        entry_scores = [_entry_passes_pos(entry, relation) for entry in entries]
        relation_scores.append(sum(entry_scores) / len(entry_scores))
    return sum(relation_scores) / len(relation_scores)


def _format_score(solution_str: str) -> float:
    if not _format_gate_passes(solution_str):
        return 0.0

    think_match = _THINK_PATTERN.search(solution_str)
    if not think_match:
        return 0.0
    think_body = think_match.group(1)

    entries_by_relation = {
        relation: _extract_relation_entries(think_body, relation) for relation in RELATIONS
    }
    quality = (
        _min_content_score(entries_by_relation)
        + _pos_sanity_score(entries_by_relation)
        + _dedup_score(entries_by_relation)
    ) / 3.0
    return FORMAT_MAX * quality


def _answer_score(solution_str: str, ground_truth: str) -> float:
    pred = _normalise(_extract_post_think_content(solution_str))
    gt = _normalise(ground_truth)
    return ANSWER_MAX if _whole_token_substring(gt, pred) else 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """Compute the v3 reward in [0, 1] for a single response."""
    del data_source, extra_info
    return _format_score(solution_str) + _answer_score(solution_str, ground_truth)
