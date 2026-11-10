"""Tests for the evaluation metrics, on hand-made inputs with known answers."""

import pytest

from job_fit.metrics import (
    ERROR_COSTS,
    LABELS,
    confusion_matrix,
    constant_baseline,
    error_cost,
    majority_label,
    score,
)


def test_perfect_predictions_score_full_accuracy_and_zero_error() -> None:
    """Predicting every label correctly gives accuracy 1 and no weighted error."""
    actual = ["APPLY", "PREPARE", "SKIP", "SKIP"]

    scores = score(actual, list(actual))

    assert scores.accuracy == 1.0
    assert scores.weighted_error_total == 0
    assert scores.weighted_error_mean == 0.0
    assert scores.failed == 0


@pytest.mark.parametrize(
    ("actual", "predicted", "cost"),
    [
        ("APPLY", "APPLY", 0),
        ("APPLY", "PREPARE", 1),
        ("APPLY", "SKIP", 3),
        ("PREPARE", "APPLY", 1),
        ("PREPARE", "PREPARE", 0),
        ("PREPARE", "SKIP", 2),
        ("SKIP", "APPLY", 1),
        ("SKIP", "PREPARE", 1),
        ("SKIP", "SKIP", 0),
    ],
)
def test_error_cost_matches_the_process_table(actual: str, predicted: str, cost: int) -> None:
    """Each cell of the cost matrix has the value given in PROCESS.md."""
    assert error_cost(actual, predicted) == cost


def test_cost_matrix_covers_every_label_pair() -> None:
    """The cost matrix has an entry for all nine combinations."""
    assert set(ERROR_COSTS) == {(row, column) for row in LABELS for column in LABELS}


@pytest.mark.parametrize(("actual", "cost"), [("APPLY", 3), ("PREPARE", 2), ("SKIP", 1)])
def test_missing_prediction_is_charged_the_worst_cost_of_its_row(actual: str, cost: int) -> None:
    """A failed posting costs as much as the worst mistake for its label."""
    assert error_cost(actual, None) == cost


def test_score_mixes_right_and_wrong_predictions() -> None:
    """Accuracy and weighted error follow from the individual predictions."""
    actual = ["APPLY", "APPLY", "PREPARE", "SKIP"]
    predicted = ["APPLY", "SKIP", "APPLY", "SKIP"]

    scores = score(actual, predicted)

    assert scores.accuracy == 0.5
    assert scores.weighted_error_total == 4
    assert scores.weighted_error_mean == 1.0


def test_confusion_matrix_rows_are_actual_and_columns_are_predicted() -> None:
    """An APPLY predicted as SKIP lands in row APPLY, column SKIP."""
    matrix = confusion_matrix(["APPLY", "SKIP", "SKIP"], ["SKIP", "SKIP", "PREPARE"])

    assert matrix == ((0, 0, 1), (0, 0, 0), (0, 1, 1))


def test_failed_posting_counts_as_wrong_and_stays_out_of_the_matrix() -> None:
    """A missing prediction lowers accuracy, adds cost and is counted separately."""
    scores = score(["APPLY", "SKIP"], [None, "SKIP"])

    assert scores.accuracy == 0.5
    assert scores.weighted_error_total == 3
    assert scores.failed == 1
    assert sum(sum(row) for row in scores.confusion) == 1


def test_majority_label_is_the_most_common_one() -> None:
    """The label that occurs most often wins."""
    assert majority_label(["APPLY", "SKIP", "SKIP", "PREPARE"]) == "SKIP"


def test_majority_label_tie_goes_to_the_first_label_in_order() -> None:
    """A tie is broken by label order, not by the order of the postings."""
    assert majority_label(["SKIP", "PREPARE", "SKIP", "PREPARE"]) == "PREPARE"


def test_constant_baseline_scores_always_predicting_one_label() -> None:
    """Always predicting SKIP is right for SKIPs and costly for the rest."""
    scores = constant_baseline(["APPLY", "PREPARE", "SKIP", "SKIP"], "SKIP")

    assert scores.accuracy == 0.5
    assert scores.weighted_error_total == 5


def test_always_apply_is_cheaper_than_always_skip_on_a_balanced_set() -> None:
    """The cost matrix punishes missed jobs more than wasted applications."""
    actual = ["APPLY", "PREPARE", "SKIP"]

    always_apply = constant_baseline(actual, "APPLY")
    always_skip = constant_baseline(actual, "SKIP")

    assert always_apply.weighted_error_total < always_skip.weighted_error_total


def test_score_rejects_empty_input() -> None:
    """Metrics on no postings are an error, not zero."""
    with pytest.raises(ValueError, match="empty"):
        score([], [])


def test_score_rejects_inputs_of_different_length() -> None:
    """Every label needs exactly one prediction."""
    with pytest.raises(ValueError, match="must match"):
        score(["APPLY", "SKIP"], ["APPLY"])


def test_majority_label_rejects_empty_input() -> None:
    """There is no majority label without postings."""
    with pytest.raises(ValueError, match="empty"):
        majority_label([])
