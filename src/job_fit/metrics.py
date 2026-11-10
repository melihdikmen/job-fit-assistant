"""Evaluation metrics over pairs of actual and predicted labels.

Everything here is a pure function: labels in, numbers out, no I/O and no LLM.
A prediction of ``None`` means the model produced no valid decision for that posting.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from job_fit.schema import DecisionLabel

Prediction = DecisionLabel | None

LABELS: tuple[DecisionLabel, ...] = ("APPLY", "PREPARE", "SKIP")

# Business cost of predicting the second label when the first is true. Missing a good
# job (APPLY predicted as SKIP) is the most expensive mistake; a wasted application
# (SKIP predicted as APPLY) is cheap.
ERROR_COSTS: dict[tuple[DecisionLabel, DecisionLabel], int] = {
    ("APPLY", "APPLY"): 0,
    ("APPLY", "PREPARE"): 1,
    ("APPLY", "SKIP"): 3,
    ("PREPARE", "APPLY"): 1,
    ("PREPARE", "PREPARE"): 0,
    ("PREPARE", "SKIP"): 2,
    ("SKIP", "APPLY"): 1,
    ("SKIP", "PREPARE"): 1,
    ("SKIP", "SKIP"): 0,
}


@dataclass(frozen=True)
class Scores:
    """Metrics for one set of predictions.

    Attributes:
        accuracy: Share of postings predicted correctly, between 0 and 1. A failed
            posting counts as wrong.
        weighted_error_total: Sum of the error costs over all postings.
        weighted_error_mean: Error cost per posting; lower is better.
        confusion: 3x3 counts, rows are actual labels and columns are predicted
            labels, both in ``LABELS`` order. Failed postings are not in it.
        failed: Number of postings without a valid prediction.
    """

    accuracy: float
    weighted_error_total: int
    weighted_error_mean: float
    confusion: tuple[tuple[int, ...], ...]
    failed: int


def _check_inputs(actual: Sequence[DecisionLabel], predicted: Sequence[Prediction]) -> None:
    """Reject inputs that cannot be scored.

    Args:
        actual: True labels.
        predicted: Predictions, in the same order as ``actual``.

    Raises:
        ValueError: If the sequences are empty or differ in length.
    """
    if not actual:
        raise ValueError("Cannot compute metrics on an empty set of postings.")
    if len(actual) != len(predicted):
        raise ValueError(
            f"Got {len(actual)} actual labels but {len(predicted)} predictions; they must match."
        )


def error_cost(actual: DecisionLabel, predicted: Prediction) -> int:
    """Return the business cost of one prediction.

    Args:
        actual: True label.
        predicted: Predicted label, or None if the model produced no valid decision.

    Returns:
        The cost from ``ERROR_COSTS``. A missing prediction is charged the worst cost
        in the row of the actual label.
    """
    if predicted is None:
        return max(ERROR_COSTS[(actual, label)] for label in LABELS)
    return ERROR_COSTS[(actual, predicted)]


def confusion_matrix(
    actual: Sequence[DecisionLabel], predicted: Sequence[Prediction]
) -> tuple[tuple[int, ...], ...]:
    """Count predictions per pair of actual and predicted label.

    Args:
        actual: True labels.
        predicted: Predictions, in the same order as ``actual``.

    Returns:
        A 3x3 table: rows are actual labels, columns are predicted labels, both in
        ``LABELS`` order. Missing predictions are left out.
    """
    counts = Counter(zip(actual, predicted, strict=True))
    return tuple(tuple(counts[(row, column)] for column in LABELS) for row in LABELS)


def score(actual: Sequence[DecisionLabel], predicted: Sequence[Prediction]) -> Scores:
    """Compute accuracy, weighted error and the confusion matrix.

    Args:
        actual: True labels.
        predicted: Predictions, in the same order as ``actual``. None marks a posting
            for which the model produced no valid decision.

    Returns:
        The metrics for these predictions.

    Raises:
        ValueError: If the sequences are empty or differ in length.
    """
    _check_inputs(actual, predicted)
    pairs = list(zip(actual, predicted, strict=True))
    correct = sum(1 for truth, guess in pairs if truth == guess)
    total_cost = sum(error_cost(truth, guess) for truth, guess in pairs)
    return Scores(
        accuracy=correct / len(pairs),
        weighted_error_total=total_cost,
        weighted_error_mean=total_cost / len(pairs),
        confusion=confusion_matrix(actual, predicted),
        failed=sum(1 for guess in predicted if guess is None),
    )


def majority_label(actual: Sequence[DecisionLabel]) -> DecisionLabel:
    """Find the most common label.

    Args:
        actual: True labels.

    Returns:
        The label that occurs most often. A tie goes to the label that comes first in
        ``LABELS``, so the result does not depend on the order of the postings.

    Raises:
        ValueError: If there are no labels.
    """
    if not actual:
        raise ValueError("Cannot find the majority label of an empty set of postings.")
    counts = Counter(actual)
    return max(LABELS, key=lambda label: counts[label])


def constant_baseline(actual: Sequence[DecisionLabel], label: DecisionLabel) -> Scores:
    """Score a "dumb model" that predicts the same label for every posting.

    Args:
        actual: True labels.
        label: The label the dumb model always predicts.

    Returns:
        The metrics of that constant predictor.

    Raises:
        ValueError: If there are no labels.
    """
    return score(actual, [label] * len(actual))
