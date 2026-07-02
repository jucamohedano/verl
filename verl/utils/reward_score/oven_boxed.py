"""Structured OVEN reward for GRPO training.

Primary signal: exact/alias match on the boxed answer (dominant, 70%).
Auxiliary shaping signals (30% total):
  - specific_hF: taxonomy specificity of the boxed answer via conservative text-only linker
  - path_match: overlap between <traversal> nodes and the GT taxonomy path
  - aggregation_improvement: whether the final answer improves over RSA candidates

All shaping signals are gated: no taxonomy credit unless the answer is taxonomically plausible.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Reward weights (sum to at most 1.0)
# ---------------------------------------------------------------------------
FORMAT_REWARD = 0.05
EXACT_REWARD = 0.70
SPECIFIC_HF_WEIGHT = 0.15
PATH_MATCH_WEIGHT = 0.05
AGGREGATION_WEIGHT = 0.05
BOXED_WRONG_REWARD = 0.0        # fallback when boxed answer doesn't match anything

# ---------------------------------------------------------------------------
# Boxed-answer extraction — mirrors oven_boxed.py original
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Alias index (from taxonomy index)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Taxonomy index loader — cached, used by linker + specific_hF
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_taxonomy_data():
    """Load the full taxonomy index and pre-build lookup structures.

    Returns None if OVEN_TAXONOMY_INDEX is not set or the file is missing.
    """
    path = os.environ.get("OVEN_TAXONOMY_INDEX")
    if not path:
        return None
    index_path = Path(path)
    if not index_path.exists():
        return None

    index = json.loads(index_path.read_text(encoding="utf-8"))
    all_nodes: list[str] = index.get("all_nodes", [])
    node_to_path: dict[str, list[str]] = index.get("node_to_path", {})
    label_to_paths: dict[str, list[list[str]]] = index.get("label_to_paths", {})
    aliases: dict[str, str] = index.get("aliases", {})

    # Pre-normalise all node labels for fast scoring
    norm_nodes = [normalize_answer(n) for n in all_nodes]
    norm_to_original: dict[str, str] = {}
    for node in all_nodes:
        norm_to_original[normalize_answer(node)] = node
    for alias, canonical in aliases.items():
        n = normalize_answer(alias)
        if n not in norm_to_original:
            norm_to_original[n] = canonical

    # Build entity_id → path map
    entity_to_path: dict[str, list[str]] = {}
    for eid, path in index.get("entity_id_to_path", {}).items():
        entity_to_path[str(eid)] = [str(p) for p in path]

    return {
        "all_nodes": all_nodes,
        "norm_nodes": norm_nodes,
        "node_to_path": node_to_path,
        "label_to_paths": label_to_paths,
        "norm_to_original": norm_to_original,
        "entity_to_path": entity_to_path,
    }


# ---------------------------------------------------------------------------
# N-gram Jaccard — for conservative text-only linking
# ---------------------------------------------------------------------------

def _get_n_grams(text: str, n: int = 2) -> set[str]:
    words = text.split()
    if len(words) < n:
        return set()
    return {" ".join(words[i: i + n]) for i in range(len(words) - n + 1)}


def _ngram_jaccard(pred_norm: str, label_norm: str, n: int = 2) -> float:
    pred_ngrams = _get_n_grams(pred_norm, n)
    label_ngrams = _get_n_grams(label_norm, n)
    if not label_ngrams:
        return 0.0
    return len(pred_ngrams & label_ngrams) / len(label_ngrams)


# ---------------------------------------------------------------------------
# Conservative linker: maps a boxed answer to a taxonomy path (or None)
# ---------------------------------------------------------------------------

def _link_prediction(prediction_text: str):
    """Map a boxed answer string to a (predicted_node, predicted_path) tuple.

    Returns None if the prediction cannot be conservatively linked to any
    taxonomy node.  The linker is text-only: no CLIP, no embeddings, no
    top-score fallback that always returns a node.
    """
    idx = _load_taxonomy_data()
    if idx is None:
        return None

    norm_pred = normalize_answer(prediction_text)
    if not norm_pred:
        return None

    norm_to_original = idx["norm_to_original"]
    node_to_path = idx["node_to_path"]
    label_to_paths = idx["label_to_paths"]
    norm_nodes = idx["norm_nodes"]
    all_nodes = idx["all_nodes"]

    # 1. Exact match against labels + aliases
    original = norm_to_original.get(norm_pred)
    if original is not None:
        path = node_to_path.get(original)
        if path is None:
            paths = label_to_paths.get(normalize_answer(original), [])
            path = paths[0] if paths else None
        if path:
            return (original, path)

    # 2. N-gram Jaccard against all node labels (conservative: require threshold)
    best_score = 0.0
    best_idx = -1
    for i, norm_label in enumerate(norm_nodes):
        score = _ngram_jaccard(norm_pred, norm_label)
        if score > best_score:
            best_score = score
            best_idx = i

    # Conservative threshold: require at least 0.5 Jaccard to accept the match
    if best_score >= 0.5 and best_idx >= 0:
        node = all_nodes[best_idx]
        path = node_to_path.get(node)
        if path is None:
            paths = label_to_paths.get(normalize_answer(node), [])
            path = paths[0] if paths else None
        if path:
            return (node, path)

    return None


# ---------------------------------------------------------------------------
# Specificity-aware hierarchical F1 — inlined from scores.py
# ---------------------------------------------------------------------------

def _suffix_weights(path: list[str], decay: float = 0.5) -> dict[tuple[str, ...], float]:
    """Weight deeper/specific suffixes more than broad/root suffixes.

    Paths are leaf→root.  The broadest suffix is ("root",) and the most
    specific is the full leaf→root path.  With decay=0.5, each step toward
    the leaf is worth twice as much as the broader ancestor before it.
    """
    total_depth = len(path)
    return {
        tuple(path[-(i + 1):]): decay ** (total_depth - (i + 1))
        for i in range(total_depth)
    }


def _is_strict_ancestor_path(pred_path: list[str], ref_path: list[str]) -> bool:
    """Whether pred_path names a broader ancestor of ref_path."""
    return (
        len(pred_path) < len(ref_path)
        and len(pred_path) > 0
        and tuple(ref_path[-len(pred_path):]) == tuple(pred_path)
    )


def _compute_specific_hF(
    pred_path: list[str],
    ref_path: list[str],
    decay: float = 0.5,
    under_specific_penalty: float = 0.5,
) -> float:
    """Compute specificity-weighted hierarchical F1 for a single prediction.

    Returns 0.0 for empty paths.  Under-specific predictions (strict ancestors
    of the reference) receive an additional penalty.
    """
    if not pred_path or not ref_path:
        return 0.0

    pred_weights = _suffix_weights(pred_path, decay=decay)
    ref_weights = _suffix_weights(ref_path, decay=decay)
    common = set(pred_weights) & set(ref_weights)

    pred_total = sum(pred_weights.values())
    ref_total = sum(ref_weights.values())
    hP = sum(pred_weights[suffix] for suffix in common) / pred_total if pred_total > 0 else 0.0
    hR = sum(ref_weights[suffix] for suffix in common) / ref_total if ref_total > 0 else 0.0
    hF = (2 * hP * hR / (hP + hR)) if (hP + hR) > 0 else 0.0

    if _is_strict_ancestor_path(pred_path, ref_path):
        hF *= under_specific_penalty

    return hF


# ---------------------------------------------------------------------------
# Traversal path parsing
# ---------------------------------------------------------------------------

def _extract_traversal_nodes(text: str) -> list[str]:
    """Parse ``<traversal>...</traversal>`` section and extract node mentions.

    Supports arrow-separated (``A → B → C``) and comma-separated formats.
    Returns normalized node names.
    """
    # Find <traversal>...</traversal>
    m = re.search(r"<traversal>\s*(.*?)\s*</traversal>", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return []

    content = m.group(1).strip()
    if not content:
        return []

    # Split on arrows (→, ->, >) or commas
    parts = re.split(r"\s*(?:→|->|>|,)\s*", content)
    nodes = []
    for part in parts:
        part = part.strip().strip(".-•*").strip()
        if part and len(part) > 1:  # skip single chars and empty
            nodes.append(normalize_answer(part))

    # Deduplicate preserving order
    seen: set[str] = set()
    unique = []
    for node in nodes:
        if node and node not in seen:
            seen.add(node)
            unique.append(node)
    return unique


def _path_match_score(traversal_nodes: list[str], gt_path: list[str]) -> float:
    """Jaccard overlap between traversal-mentioned nodes and GT taxonomy path nodes."""
    if not traversal_nodes or not gt_path:
        return 0.0

    gt_norms = {normalize_answer(n) for n in gt_path if n}
    traversal_set = set(traversal_nodes)

    intersection = traversal_set & gt_norms
    union = traversal_set | gt_norms

    if not union:
        return 0.0
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Aggregation improvement
# ---------------------------------------------------------------------------

def _aggregation_improvement(
    prediction_norm: str,
    extra_info: dict | None,
) -> float:
    """Score whether the final answer improves over the RSA candidate answers.

    Returns 0.0–1.0 where:
      0.0 = worse than all candidates
      0.5 = selected the best candidate (correctly identified the right answer)
      1.0 = improved beyond the best candidate (synthesised a better answer)

    Only applies when prompt_type == "aggregation".
    """
    if not extra_info or extra_info.get("prompt_type") != "aggregation":
        return 0.0

    candidate_answers = extra_info.get("candidate_final_answers") or []
    if not candidate_answers:
        return 0.0

    gt_norms = valid_answer_norms(
        extra_info.get("answer", ""),
        extra_info,
    )

    # Score each candidate: 1.0 if correct, 0.0 otherwise
    candidate_scores = [
        1.0 if normalize_answer(str(c)) in gt_norms else 0.0
        for c in candidate_answers
    ]
    best_candidate = max(candidate_scores) if candidate_scores else 0.0
    final_score = 1.0 if prediction_norm in gt_norms else 0.0

    if final_score > best_candidate:
        # The model improved beyond any candidate — full aggregation credit
        return 1.0
    elif final_score == 1.0 and best_candidate == 1.0:
        # At least one candidate was already correct, model correctly selected it
        return 0.5
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    del data_source

    prediction, parse_ok = extract_boxed_answer(solution_str)
    if not parse_ok:
        return 0.0

    R = FORMAT_REWARD
    pred_norm = normalize_answer(prediction)
    gt_norms = valid_answer_norms(ground_truth, extra_info)
    exact_match = pred_norm in gt_norms

    # --- 1. Exact / alias match (dominant) ---
    if exact_match:
        R += EXACT_REWARD

    # --- 2. Conservative taxonomy shaping (specific_hF) ---
    linked = _link_prediction(prediction)
    if linked is not None:
        pred_node, pred_path = linked

        # Look up GT path from node_to_path (canonical format).
        # Prefer taxonomy_labels[0] (the leaf label), then ground_truth.
        idx = _load_taxonomy_data()
        gt_path = None
        if idx:
            # Best: leaf label from extra_info, looked up in node_to_path
            leaf = _taxonomy_leaf(extra_info)
            if leaf:
                gt_path = idx["node_to_path"].get(leaf)
            # Fallback: ground_truth label
            if gt_path is None:
                gt_path = idx["node_to_path"].get(ground_truth)
            # Last resort: label_to_paths
            if gt_path is None:
                paths = idx["label_to_paths"].get(normalize_answer(ground_truth), [])
                gt_path = paths[0] if paths else None

        if gt_path:
            shF = _compute_specific_hF(pred_path, gt_path)
            R += SPECIFIC_HF_WEIGHT * shF

            # --- 3. Path match (double-gated: only if answer is taxonomically plausible) ---
            if shF >= 0.3:
                traversal_nodes = _extract_traversal_nodes(solution_str)
                if traversal_nodes:
                    R += PATH_MATCH_WEIGHT * _path_match_score(traversal_nodes, gt_path)

    # --- 4. Aggregation improvement ---
    if extra_info and extra_info.get("prompt_type") == "aggregation":
        R += AGGREGATION_WEIGHT * _aggregation_improvement(pred_norm, extra_info)

    return min(R, 1.0)
