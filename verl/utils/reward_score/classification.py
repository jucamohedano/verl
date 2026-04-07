"""TTW Classification Reward Function for verl GRPO.

Implements a two-component reward mirroring the GSM8K approach:

  1. FORMAT reward  — did the model use <think>...</think><answer>...</answer>?
  2. CORRECTNESS reward — does the extracted label match ground truth?

Total score per response: 0.0 / 0.5 / 1.0 / 1.5 / 2.0

verl calls ``compute_score`` once per response with the signature::

    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

To use this file, add to your verl launch script:
    custom_reward_function.path=/path/to/classification_reward.py
    custom_reward_function.name=compute_score   # optional, this is the default
"""

import re

# ---------------------------------------------------------------------------
# Tag patterns
# ---------------------------------------------------------------------------

# Requires both tags to be present with content, in the right order.
# re.DOTALL lets . match newlines inside the tags.
_FORMAT_PATTERN = re.compile(
    r"<think>(.+?)</think>\s*<answer>(.+?)</answer>",
    re.DOTALL,
)

_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# ---------------------------------------------------------------------------
# Reward weights — adjust here to change the total scale
# ---------------------------------------------------------------------------

FORMAT_SCORE = 0.5       # awarded for correct tag structure
CORRECTNESS_SCORE = 1.0  # awarded for a matching label

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace/punctuation variants.

    Handles common mismatches:
      - "Yorkshire_Terrier" -> "yorkshire terrier"
      - "baby-crawling"     -> "baby crawling"
      - "  striped  "       -> "striped"
    """
    text = text.lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _labels_match(predicted: str, ground_truth: str) -> bool:
    """Normalised exact match (Option A from the reward design doc).

    This is the strictest matching strategy and the cleanest baseline —
    it mirrors the binary behaviour of the GSM8K reward and makes training
    dynamics easy to interpret.

    To switch to softer matching (e.g. for dtd or ucf101), replace the body
    with one of the alternatives below:

    Option B — substring containment (good for oxford_pets breed names):
        return gt in pred or pred in gt

    Option C — token overlap (good for ucf101 multi-word actions):
        pred_tokens = set(pred.split())
        gt_tokens   = set(gt.split())
        return len(pred_tokens & gt_tokens) / len(gt_tokens) >= 0.8
    """
    pred = _normalise(predicted)
    gt   = _normalise(ground_truth)
    return pred == gt


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def _format_score(solution_str: str) -> float:
    """Return FORMAT_SCORE if the response uses both required tags, else 0.0."""
    return FORMAT_SCORE if _FORMAT_PATTERN.search(solution_str) else 0.0


def _correctness_score(solution_str: str, ground_truth: str) -> float:
    """Return CORRECTNESS_SCORE if the extracted label matches ground truth.

    Returns 0.0 if:
      - no <answer> tag is present
      - the extracted label does not match after normalisation
    """
    match = _ANSWER_PATTERN.search(solution_str)
    if not match:
        return 0.0
    predicted = match.group(1).strip()
    return CORRECTNESS_SCORE if _labels_match(predicted, ground_truth) else 0.0


# ---------------------------------------------------------------------------
# Public entry point — called by verl per response
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """Compute the total classification reward for a single model response.

    Args:
        data_source:  Task name from the parquet, e.g. "oxford_pets".
                      Can be used to apply different matching strategies
                      per dataset (see comments below).
        solution_str: The model's raw response string (detokenised).
        ground_truth: The gold label string from reward_model.ground_truth
                      in the parquet, e.g. "yorkshire terrier".
        extra_info:   Dict from the parquet's extra_info column.
                      Contains "gt_label", "split", "index".
                      Not needed here but available for debugging.

    Returns:
        Float in [0.0, FORMAT_SCORE + CORRECTNESS_SCORE].
        Currently: 0.0, 0.5, 1.0, or 1.5.

    Per-dataset notes
    -----------------
    caltech101  : normalised exact match works well (unambiguous nouns).
    oxford_pets : normalised exact match works well (breed names).
    dtd         : consider Option B (substring) for texture adjectives.
    ucf101      : consider Option C (token overlap) for action phrases.

    To route per dataset, replace the _correctness_score call with:

        if data_source in ("dtd",):
            correct = _correctness_score_substring(solution_str, ground_truth)
        elif data_source in ("ucf101",):
            correct = _correctness_score_token_overlap(solution_str, ground_truth)
        else:
            correct = _correctness_score(solution_str, ground_truth)
    """
    fmt     = _format_score(solution_str)
    correct = _correctness_score(solution_str, ground_truth)
    return fmt + correct
