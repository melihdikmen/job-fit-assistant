"""Evaluation metrics over pairs of actual and predicted labels.

Everything here is a pure function: labels in, numbers out, no I/O and no LLM.
A prediction of ``None`` means the model produced no valid decision for that posting.
"""

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from job_fit.schema import DecisionLabel

Prediction = DecisionLabel | None
CostMatrix = Mapping[tuple[DecisionLabel, DecisionLabel], int]

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

MISSED_APPLY: tuple[DecisionLabel, DecisionLabel] = ("APPLY", "SKIP")

# The cost matrix is a judgment, not a measurement. These two variants change its most
# debatable entry, the price of a missed good job, to see whether a conclusion drawn
# from the weighted error depends on that one number.
ALTERNATIVE_COSTS: dict[str, CostMatrix] = {
    "missed APPLY = 2": {**ERROR_COSTS, MISSED_APPLY: 2},
    "missed APPLY = 5": {**ERROR_COSTS, MISSED_APPLY: 5},
}


@dataclass(frozen=True)
class ClassScores:
    """Precision and recall of one label.

    Attributes:
        precision: Of the postings predicted as this label, the share that truly have
            it. None if the label was never predicted: there is nothing to be right
            or wrong about.
        recall: Of the postings that truly have this label, the share predicted as
            it. A failed posting counts as missed. None if no posting has the label.
        support: Number of postings that truly have this label.
        predicted: Number of postings predicted as this label.
    """

    precision: float | None
    recall: float | None
    support: int
    predicted: int


@dataclass(frozen=True)
class Spread:
    """How one number varied over repeated runs of the same configuration.

    Attributes:
        mean: Mean over the runs.
        stdev: Sample standard deviation; 0 for a single run.
        minimum: Smallest value seen.
        maximum: Largest value seen.
        runs: Number of runs.
    """

    mean: float
    stdev: float
    minimum: float
    maximum: float
    runs: int


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
        kappa: Cohen's kappa against the labels: agreement beyond what the label mix
            alone would produce. None in run files written before the field existed.
        per_class: Precision and recall per label; None in older run files.
        alternative_weighted_error_mean: Mean weighted error under each matrix of
            ``ALTERNATIVE_COSTS``; None in older run files.
    """

    accuracy: float
    weighted_error_total: int
    weighted_error_mean: float
    confusion: tuple[tuple[int, ...], ...]
    failed: int
    kappa: float | None = None
    per_class: dict[DecisionLabel, ClassScores] | None = None
    alternative_weighted_error_mean: dict[str, float] | None = None


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


def error_cost(
    actual: DecisionLabel, predicted: Prediction, costs: CostMatrix = ERROR_COSTS
) -> int:
    """Return the business cost of one prediction.

    Args:
        actual: True label.
        predicted: Predicted label, or None if the model produced no valid decision.
        costs: Cost of every pair of actual and predicted label.

    Returns:
        The cost from the matrix. A missing prediction is charged the worst cost in
        the row of the actual label.
    """
    if predicted is None:
        return max(costs[(actual, label)] for label in LABELS)
    return costs[(actual, predicted)]


def weighted_error_total(
    actual: Sequence[DecisionLabel],
    predicted: Sequence[Prediction],
    costs: CostMatrix = ERROR_COSTS,
) -> int:
    """Sum the business cost of all predictions.

    Args:
        actual: True labels.
        predicted: Predictions, in the same order as ``actual``.
        costs: Cost of every pair of actual and predicted label.

    Returns:
        The total cost; lower is better.

    Raises:
        ValueError: If the sequences are empty or differ in length.
    """
    _check_inputs(actual, predicted)
    return sum(
        error_cost(truth, guess, costs) for truth, guess in zip(actual, predicted, strict=True)
    )


def precision_recall(
    actual: Sequence[DecisionLabel], predicted: Sequence[Prediction]
) -> dict[DecisionLabel, ClassScores]:
    """Compute precision and recall for every label.

    Args:
        actual: True labels.
        predicted: Predictions, in the same order as ``actual``. A failed posting
            (None) lowers the recall of its true label and is no prediction of any.

    Returns:
        The scores per label, in ``LABELS`` order.

    Raises:
        ValueError: If the sequences are empty or differ in length.
    """
    _check_inputs(actual, predicted)
    pairs = list(zip(actual, predicted, strict=True))
    scores: dict[DecisionLabel, ClassScores] = {}
    for label in LABELS:
        support = sum(1 for truth, _ in pairs if truth == label)
        predicted_count = sum(1 for _, guess in pairs if guess == label)
        hits = sum(1 for truth, guess in pairs if truth == label and guess == label)
        scores[label] = ClassScores(
            precision=hits / predicted_count if predicted_count else None,
            recall=hits / support if support else None,
            support=support,
            predicted=predicted_count,
        )
    return scores


def cohen_kappa(actual: Sequence[DecisionLabel], predicted: Sequence[Prediction]) -> float:
    """Measure agreement with the labels beyond what chance would produce.

    Accuracy rewards a model for guessing the common label. Kappa subtracts the
    agreement expected from the label mix alone: 1 is perfect agreement, 0 is what a
    predictor that ignores the posting gets, below 0 is worse than that.

    Args:
        actual: True labels.
        predicted: Predictions, in the same order as ``actual``. A failed posting
            (None) is an outcome of its own and therefore always a disagreement.

    Returns:
        Cohen's kappa. 0 when both sides use a single, identical label: there is no
        variation to agree on.

    Raises:
        ValueError: If the sequences are empty or differ in length.
    """
    _check_inputs(actual, predicted)
    total = len(actual)
    observed = sum(1 for truth, guess in zip(actual, predicted, strict=True) if truth == guess)
    actual_counts = Counter(actual)
    predicted_counts = Counter(predicted)
    expected = sum(actual_counts[label] * predicted_counts[label] for label in LABELS) / total**2
    if expected == 1:
        return 0.0
    return (observed / total - expected) / (1 - expected)


def spread(values: Sequence[float]) -> Spread:
    """Summarize one number over repeated runs.

    Args:
        values: The number as measured in each run.

    Returns:
        Mean, sample standard deviation, minimum and maximum.

    Raises:
        ValueError: If there are no values.
    """
    if not values:
        raise ValueError("Cannot summarize an empty list of runs.")
    return Spread(
        mean=statistics.fmean(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
        runs=len(values),
    )


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
    total_cost = weighted_error_total(actual, predicted)
    return Scores(
        accuracy=correct / len(pairs),
        weighted_error_total=total_cost,
        weighted_error_mean=total_cost / len(pairs),
        confusion=confusion_matrix(actual, predicted),
        failed=sum(1 for guess in predicted if guess is None),
        kappa=cohen_kappa(actual, predicted),
        per_class=precision_recall(actual, predicted),
        alternative_weighted_error_mean={
            name: weighted_error_total(actual, predicted, costs) / len(pairs)
            for name, costs in ALTERNATIVE_COSTS.items()
        },
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
