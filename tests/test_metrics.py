"""Tests for the evaluation metrics, on hand-made inputs with known answers."""

import pytest

from job_fit.metrics import (
    ALTERNATIVE_COSTS,
    ERROR_COSTS,
    LABELS,
    MISSED_APPLY,
    ClassScores,
    Spread,
    cohen_kappa,
    confusion_matrix,
    constant_baseline,
    error_cost,
    majority_label,
    precision_recall,
    score,
    spread,
    weighted_error_total,
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


def test_precision_and_recall_are_computed_per_label() -> None:
    """Precision looks down a column of the confusion matrix, recall along a row."""
    actual = ["APPLY", "APPLY", "PREPARE", "SKIP", "SKIP", "SKIP"]
    predicted = ["APPLY", "SKIP", "APPLY", "SKIP", "SKIP", "APPLY"]

    scores = precision_recall(actual, predicted)

    # APPLY: predicted 3 times, 1 of them right; 2 postings are APPLY, 1 was found.
    assert scores["APPLY"] == ClassScores(precision=1 / 3, recall=1 / 2, support=2, predicted=3)
    # SKIP: predicted 3 times, 2 right; 3 postings are SKIP, 2 were found.
    assert scores["SKIP"] == ClassScores(precision=2 / 3, recall=2 / 3, support=3, predicted=3)


def test_precision_is_undefined_for_a_label_that_was_never_predicted() -> None:
    """Zero predictions of a label give no precision, not a precision of zero."""
    scores = precision_recall(["APPLY", "PREPARE"], ["APPLY", "APPLY"])

    assert scores["PREPARE"].precision is None
    assert scores["PREPARE"].recall == 0.0


def test_recall_is_undefined_for_a_label_without_postings() -> None:
    """A label that does not occur has nothing to recall."""
    scores = precision_recall(["APPLY"], ["APPLY"])

    assert scores["SKIP"].recall is None
    assert scores["SKIP"].support == 0


def test_failed_posting_lowers_recall_and_is_no_prediction() -> None:
    """A posting without a decision is a miss for its label and counts for no label."""
    scores = precision_recall(["APPLY", "APPLY"], ["APPLY", None])

    assert scores["APPLY"].recall == 0.5
    assert scores["APPLY"].precision == 1.0
    assert sum(row.predicted for row in scores.values()) == 1


def test_kappa_is_one_for_perfect_agreement() -> None:
    """Identical labels on a mixed set agree completely."""
    actual = ["APPLY", "PREPARE", "SKIP", "SKIP"]

    assert cohen_kappa(actual, list(actual)) == 1.0


@pytest.mark.parametrize(
    "actual",
    [
        ["SKIP", "SKIP", "SKIP", "APPLY"],
        ["APPLY", "PREPARE", "SKIP"],
        ["APPLY"] * 9 + ["PREPARE"] * 9 + ["SKIP"] * 22,
    ],
    ids=["skewed", "uniform", "like-the-real-set"],
)
@pytest.mark.parametrize("constant", LABELS)
def test_kappa_of_a_constant_predictor_is_zero(actual: list[str], constant: str) -> None:
    """Always answering one label scores 0, whatever the label mix and however often it is right."""
    predicted = [constant] * len(actual)

    assert cohen_kappa(actual, predicted) == pytest.approx(0.0)
    assert constant_baseline(actual, constant).kappa == pytest.approx(0.0)


def test_kappa_subtracts_the_agreement_expected_by_chance() -> None:
    """Kappa is (observed - expected) / (1 - expected), with expected from the label mix."""
    actual = ["APPLY", "APPLY", "SKIP", "SKIP"]
    predicted = ["APPLY", "SKIP", "SKIP", "SKIP"]
    observed = 3 / 4
    # APPLY: 2 of 4 actual, 1 of 4 predicted; SKIP: 2 of 4 actual, 3 of 4 predicted.
    expected = (2 / 4) * (1 / 4) + (2 / 4) * (3 / 4)

    assert cohen_kappa(actual, predicted) == pytest.approx((observed - expected) / (1 - expected))


def test_kappa_can_be_negative() -> None:
    """Systematic disagreement is worse than ignoring the posting."""
    assert cohen_kappa(["APPLY", "SKIP"], ["SKIP", "APPLY"]) < 0


def test_kappa_counts_a_failed_posting_as_disagreement() -> None:
    """A missing decision lowers kappa below perfect agreement."""
    actual = ["APPLY", "SKIP", "SKIP"]

    assert cohen_kappa(actual, ["APPLY", "SKIP", None]) < 1.0


def test_kappa_is_zero_when_there_is_no_variation_to_agree_on() -> None:
    """One label on both sides gives no evidence of skill."""
    assert cohen_kappa(["SKIP", "SKIP"], ["SKIP", "SKIP"]) == 0.0


@pytest.mark.parametrize("name", ALTERNATIVE_COSTS)
def test_alternative_matrices_change_only_the_missed_apply_cost(name: str) -> None:
    """The variants differ from the default matrix in exactly one entry."""
    costs = ALTERNATIVE_COSTS[name]
    changed = {pair for pair in ERROR_COSTS if costs[pair] != ERROR_COSTS[pair]}

    assert changed == {MISSED_APPLY}


@pytest.mark.parametrize("name", ALTERNATIVE_COSTS)
def test_weighted_error_uses_the_given_cost_matrix(name: str) -> None:
    """A missed APPLY is charged what the chosen matrix says; other errors are unchanged."""
    costs = ALTERNATIVE_COSTS[name]
    actual = ["APPLY", "SKIP"]
    predicted = ["SKIP", "APPLY"]

    total = weighted_error_total(actual, predicted, costs)

    assert total == costs[MISSED_APPLY] + ERROR_COSTS[("SKIP", "APPLY")]
    assert score(actual, predicted).alternative_weighted_error_mean[name] == total / len(actual)


def test_failed_posting_is_charged_the_worst_cost_of_the_given_matrix() -> None:
    """Under a matrix with a dearer missed APPLY, a failed APPLY posting costs that much."""
    costs = ALTERNATIVE_COSTS["missed APPLY = 5"]

    assert error_cost("APPLY", None, costs) == max(costs[("APPLY", label)] for label in LABELS)


def test_score_carries_kappa_and_per_class_numbers() -> None:
    """The combined score includes the numbers added for the Week 2 report."""
    actual = ["APPLY", "PREPARE", "SKIP", "SKIP"]

    scores = score(actual, list(actual))

    assert scores.kappa == 1.0
    assert scores.per_class == precision_recall(actual, list(actual))
    assert set(scores.alternative_weighted_error_mean) == set(ALTERNATIVE_COSTS)


def test_spread_reports_mean_sample_stdev_and_range() -> None:
    """Three runs give a mean, the sample standard deviation and the extremes."""
    result = spread([1.0, 2.0, 3.0])

    assert result == Spread(mean=2.0, stdev=1.0, minimum=1.0, maximum=3.0, runs=3)


def test_spread_of_a_single_run_has_no_deviation() -> None:
    """One run cannot show how much runs differ."""
    assert spread([0.4]) == Spread(mean=0.4, stdev=0.0, minimum=0.4, maximum=0.4, runs=1)


def test_spread_of_no_runs_is_rejected() -> None:
    """There is nothing to summarize without a run."""
    with pytest.raises(ValueError, match="empty"):
        spread([])
