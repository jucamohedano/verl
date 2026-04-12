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

"""Graded classification reward for verl GRPO with attribute extraction.

Reward components (max 1.0 per response):
  format       0.10  - structural validity of <redacted_thinking>/<attrs>/<answer>
  attributes   0.20  - ConceptNet-verified visual attributes (gated on answer)
  answer       0.70  - graded by ConceptNet-derived label metadata

Answer tiers (multiplied by 0.70):
  Specific       1.00  exact match or synonym
  Less Specific  0.60  parent (single-hop IsA)
  Generic        0.30  grandparent (two-hop IsA)
  Sibling        0.15  shares a parent with the ground truth
  Abstain        0.15  honest refusal
  Wrong          0.00

Attribute reward is GATED: zero unless the answer tier is at least Generic.
This prevents good attributes from compensating for wrong labels.

Expected response format:
  <redacted_thinking>...</redacted_thinking>
  <HasProperty>tag1, tag2, tag3</HasProperty>
  <HasA>tag1, tag2</HasA>
  <AtLocation>tag1</AtLocation>
  <answer>label</answer>

Metadata: one JSON file per lm-eval task name in ``metadata/<task_name>.json``,
keyed by normalised ground-truth labels (see ``reward_design_v2.md`` in lmms-ocw).

verl launch (example)::
    custom_reward_function.path=.../classification.py
    custom_reward_function.name=compute_score
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Tunable weights
# ---------------------------------------------------------------------------

FORMAT_SCORE = 0.10
ATTRIBUTE_MAX = 0.20
ANSWER_MAX = 0.70

TIER_WEIGHTS = {
    "specific": 1.00,
    "less_specific": 0.60,
    "generic": 0.30,
    "sibling": 0.15,
    "abstain": 0.15,
    "wrong": 0.00,
}

# Tiers at or above which the attribute reward unlocks
GATING_TIERS = {"specific", "less_specific", "generic"}

RELATIONS = ("HasProperty", "HasA", "AtLocation")
TARGET_HITS_PER_RELATION = 3
MAX_TAGS_PER_RELATION = 5  # hard cap before scoring to prevent spam

ABSTAIN_TOKENS = {"none", "n/a", "unknown", "i don't know", "unsure", "cannot tell"}

METADATA_DIR = Path(__file__).resolve().parent / "metadata"

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_FORMAT_PATTERN = re.compile(
    r"<redacted_thinking>.+?</redacted_thinking>.*?<answer>.+?</answer>",
    re.DOTALL,
)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_RELATION_PATTERNS = {rel: re.compile(rf"<{rel}>(.*?)</{rel}>", re.DOTALL) for rel in RELATIONS}

# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def _load_all_metadata() -> dict[str, dict]:
    tables: dict[str, dict] = {}
    if not METADATA_DIR.is_dir():
        return tables
    for path in METADATA_DIR.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            tables[path.stem] = json.load(f)
    return tables


_METADATA = _load_all_metadata()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def _extract_tags(solution_str: str, relation: str) -> set[str]:
    """Pull tags from a relation block, normalised, deduped, capped (comma order)."""
    match = _RELATION_PATTERNS[relation].search(solution_str)
    if not match:
        return set()
    raw = match.group(1).split(",")
    ordered: list[str] = []
    seen: set[str] = set()
    for t in raw:
        if not t.strip():
            continue
        n = _normalise(t)
        if n in seen:
            continue
        seen.add(n)
        ordered.append(n)
        if len(ordered) >= MAX_TAGS_PER_RELATION:
            break
    return set(ordered)


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------


def _categorise(prediction: str, gt: str, table: dict) -> str:
    pred = _normalise(prediction)
    gt_norm = _normalise(gt)

    if pred in ABSTAIN_TOKENS:
        return "abstain"

    info = table.get(gt_norm, {})
    parents = {_normalise(s) for s in info.get("parents", [])}
    grandparents = {_normalise(s) for s in info.get("grandparents", [])}
    synonyms = {_normalise(s) for s in info.get("synonyms", [])}
    siblings = {_normalise(s) for s in info.get("siblings", [])}

    if pred == gt_norm or pred in synonyms:
        return "specific"
    if pred in parents:
        return "less_specific"
    if pred in grandparents:
        return "generic"
    if pred in siblings:
        return "sibling"
    return "wrong"


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------


def _format_score(solution_str: str) -> float:
    return FORMAT_SCORE if _FORMAT_PATTERN.search(solution_str) else 0.0


def _answer_score_and_tier(solution_str: str, gt: str, table: dict) -> tuple[float, str]:
    match = _ANSWER_PATTERN.search(solution_str)
    if not match:
        return 0.0, "wrong"
    tier = _categorise(match.group(1).strip(), gt, table)
    return ANSWER_MAX * TIER_WEIGHTS[tier], tier


def _attribute_score(solution_str: str, gt: str, table: dict, tier: str) -> float:
    """Mean per-relation hit ratio, scaled to ATTRIBUTE_MAX. Gated on tier."""
    if tier not in GATING_TIERS:
        return 0.0

    gt_attrs = table.get(_normalise(gt), {}).get("attributes", {})

    per_relation: list[float] = []
    for relation in RELATIONS:
        valid = {_normalise(a) for a in gt_attrs.get(relation, [])}
        if not valid:
            continue  # no metadata for this relation, skip
        hits = len(_extract_tags(solution_str, relation) & valid)
        per_relation.append(min(1.0, hits / TARGET_HITS_PER_RELATION))

    if not per_relation:
        return 0.0
    return ATTRIBUTE_MAX * (sum(per_relation) / len(per_relation))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """Total reward in [0.0, 1.0] for a single response."""
    if data_source not in _METADATA:
        raise KeyError(
            f"No metadata JSON for data_source={data_source!r} under {METADATA_DIR}. "
            f"Add {data_source}.json (see lmms-ocw docs/reward_design_v2.md). "
            f"Available: {sorted(_METADATA)}"
        )
    table = _METADATA[data_source]

    fmt = _format_score(solution_str)
    answer, tier = _answer_score_and_tier(solution_str, ground_truth, table)
    attrs = _attribute_score(solution_str, ground_truth, table, tier)

    return fmt + answer + attrs
