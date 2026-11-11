"""Tests for the comparison table built from saved eval runs. No network, no LLM."""

from datetime import datetime
from pathlib import Path

import pytest

from job_fit.compare import (
    CompareError,
    Row,
    format_markdown_table,
    format_version_table,
    load_runs,
    main,
    pareto_front,
    rows_from_reports,
)
from job_fit.eval import EvalReport, PostingResult, build_report, save_report
from job_fit.workflow import ToolCallRecord

STARTED_AT = datetime(2026, 11, 10, 15, 30, 5)
LABELS = ["APPLY", "SKIP", "SKIP", "PREPARE"]


def _row(name: str, error: float, cost: float | None, latency: float) -> Row:
    """Build a table row from the three measures the Pareto front looks at."""
    return Row(
        name=name,
        accuracy=0.5,
        weighted_error_mean=error,
        cost_per_100_usd=cost,
        mean_latency_s=latency,
    )


def _report(
    model: str,
    predictions: list[str | None],
    *,
    provider: str = "vertex",
    reasoning: str = "default",
    dataset_sha256: str = "same-set",
) -> EvalReport:
    """Build a report for a run that predicted the given labels for ``LABELS``."""
    results = [
        PostingResult(
            id=f"p-{index}",
            company="Acme",
            role="Engineer",
            label=label,
            note="",
            predicted=predicted,
            llm_decision=predicted,
            input_tokens=1000,
            output_tokens=200,
            latency_s=2.0,
            attempts=1,
        )
        for index, (label, predicted) in enumerate(zip(LABELS, predictions, strict=True))
    ]
    return build_report(
        results,
        model=model,
        provider=provider,
        started_at=STARTED_AT,
        dataset_path=Path("data/postings.jsonl"),
        dataset_sha256=dataset_sha256,
        coverage_problems=[],
        reasoning=reasoning,
    )


def _save(report: EvalReport, folder: Path, name: str) -> Path:
    """Write a report into its own sub-folder and return the file path."""
    return save_report(report, folder / name, STARTED_AT)


def test_dominated_row_is_not_on_the_front() -> None:
    """A row that is worse on every measure than another is dropped."""
    good = _row("good", error=0.3, cost=0.10, latency=2.0)
    bad = _row("bad", error=0.5, cost=0.20, latency=3.0)

    assert pareto_front([good, bad]) == [good]


def test_rows_that_trade_off_are_all_on_the_front() -> None:
    """Accurate-but-expensive and cheap-but-weaker rows both stay."""
    accurate = _row("accurate", error=0.2, cost=1.00, latency=10.0)
    cheap = _row("cheap", error=0.6, cost=0.02, latency=1.0)
    middle = _row("middle", error=0.4, cost=0.10, latency=3.0)

    assert pareto_front([accurate, cheap, middle]) == [accurate, cheap, middle]


def test_row_equal_on_two_measures_and_worse_on_one_is_dominated() -> None:
    """Being strictly worse on a single measure is enough to be dominated."""
    fast = _row("fast", error=0.4, cost=0.10, latency=2.0)
    slow = _row("slow", error=0.4, cost=0.10, latency=9.0)

    assert pareto_front([slow, fast]) == [fast]


def test_identical_rows_are_both_kept() -> None:
    """Ties do not dominate each other."""
    first = _row("first", error=0.4, cost=0.10, latency=2.0)
    second = _row("second", error=0.4, cost=0.10, latency=2.0)

    assert pareto_front([first, second]) == [first, second]


def test_row_with_unknown_cost_is_left_out_of_the_front() -> None:
    """A row without a price can neither be on the front nor push others off it."""
    unpriced = _row("unpriced", error=0.0, cost=None, latency=0.1)
    priced = _row("priced", error=0.5, cost=0.10, latency=2.0)

    assert pareto_front([unpriced, priced]) == [priced]


def test_rows_start_with_the_majority_baseline() -> None:
    """The baseline row costs nothing and names the label it always predicts."""
    report = _report("gemini-2.5-flash", ["APPLY", "SKIP", "SKIP", "PREPARE"])

    rows = rows_from_reports([report])

    assert rows[0].name == "Baseline"
    assert rows[0].accuracy == 0.5
    assert rows[0].cost_per_100_usd == 0.0
    assert rows[0].notes == ("always SKIP",)
    assert rows[1].name == "gemini-2.5-flash"
    assert rows[1].accuracy == 1.0


def test_row_name_shows_reasoning_and_notes_show_provider_and_failures() -> None:
    """Everything needed to tell two runs apart is visible in the row."""
    on_vertex = _report("gemini-2.5-flash", ["APPLY", "SKIP", "SKIP", "SKIP"], reasoning="off")
    elsewhere = _report("vendor/open:free", ["APPLY", None, "SKIP", "SKIP"], provider="openrouter")

    rows = rows_from_reports([on_vertex, elsewhere])

    assert rows[1].name == "gemini-2.5-flash (reasoning off)"
    assert rows[1].notes == ()
    assert rows[2].name == "vendor/open:free"
    assert rows[2].notes == ("via openrouter", "1 failed")


def test_repeated_run_is_numbered() -> None:
    """A second run of the same configuration gets a run number instead of a duplicate name."""
    report = _report("gemini-2.5-flash", ["APPLY", "SKIP", "SKIP", "PREPARE"])

    rows = rows_from_reports([report, report])

    assert [row.name for row in rows[1:]] == ["gemini-2.5-flash", "gemini-2.5-flash, run 2"]


def test_table_marks_the_front_and_prints_unknown_cost() -> None:
    """The Markdown table has one line per row, a Pareto mark and no invented price."""
    rows = [
        Row("Baseline", 0.5, 1.0, 0.0, 0.0, ("always SKIP",)),
        Row("model-a", 0.75, 0.25, 0.631, 14.84),
        Row("model-b", 0.5, 0.5, 0.9, 20.0),
        Row("model-c", 0.5, 0.5, None, 3.0, ("via openrouter",)),
    ]

    lines = format_markdown_table(rows).splitlines()

    assert lines[0].startswith("| Model | Accuracy | Weighted error (mean) | Cost per 100 postings")
    assert lines[2] == "| Baseline | 50.0% | 1.00 | $0.00 | 0.0 s | always SKIP |"
    assert lines[3] == "| model-a | 75.0% | 0.25 | $0.63 | 14.8 s | Pareto front |"
    assert lines[4] == "| model-b | 50.0% | 0.50 | $0.90 | 20.0 s |  |"
    assert lines[5] == "| model-c | 50.0% | 0.50 | unknown | 3.0 s | via openrouter |"


def test_load_runs_reads_saved_reports(tmp_path: Path) -> None:
    """Run files written by eval load back as reports, in the order given."""
    first = _save(_report("model-a", ["APPLY", "SKIP", "SKIP", "SKIP"]), tmp_path, "a")
    second = _save(_report("model-b", ["SKIP", "SKIP", "SKIP", "SKIP"]), tmp_path, "b")

    reports = load_runs([first, second])

    assert [report.model for report in reports] == ["model-a", "model-b"]


def test_load_runs_reads_a_run_file_from_before_the_reasoning_field(tmp_path: Path) -> None:
    """Older run files without reasoning and location still load, with defaults."""
    report = _report("model-a", ["APPLY", "SKIP", "SKIP", "SKIP"])
    path = tmp_path / "old.json"
    path.write_text(report.model_dump_json(exclude={"reasoning", "location"}), encoding="utf-8")

    loaded = load_runs([path])[0]

    assert (loaded.reasoning, loaded.location) == ("default", None)


def test_load_runs_rejects_runs_on_different_test_sets(tmp_path: Path) -> None:
    """Runs made on different dataset versions must not share a table."""
    first = _save(_report("model-a", ["APPLY", "SKIP", "SKIP", "SKIP"]), tmp_path, "a")
    other_set = _report("model-b", ["APPLY", "SKIP", "SKIP", "SKIP"], dataset_sha256="other-set")
    second = _save(other_set, tmp_path, "b")

    with pytest.raises(CompareError, match="not comparable"):
        load_runs([first, second])


def test_load_runs_rejects_a_file_that_is_not_a_run(tmp_path: Path) -> None:
    """A JSON file of another shape is reported by name."""
    path = tmp_path / "other.json"
    path.write_text('{"hello": "world"}', encoding="utf-8")

    with pytest.raises(CompareError, match="other.json"):
        load_runs([path])


def test_load_runs_needs_at_least_one_file() -> None:
    """An empty list of run files is a usage error."""
    with pytest.raises(CompareError, match="at least one"):
        load_runs([])


def test_main_prints_the_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command prints the Markdown table and exits 0."""
    path = _save(_report("gemini-2.5-flash", ["APPLY", "SKIP", "SKIP", "SKIP"]), tmp_path, "a")

    exit_code = main([str(path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "| Baseline | 50.0% |" in output
    assert "| gemini-2.5-flash | 75.0% |" in output


def test_main_reports_a_missing_run_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing file gives exit code 1 and an error on stderr."""
    exit_code = main([str(tmp_path / "missing.json")])

    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


def _agent_report(predictions: list[str | None], prompt_sha256: str = "agent-prompt") -> EvalReport:
    """Build an agent report in which every posting took 3 model calls and 2 tool calls."""
    base = _report("gemini-2.5-flash", predictions, reasoning="on")
    calls = (
        ToolCallRecord(name="get_cv_section", arguments={"topic": "skills"}),
        ToolCallRecord(name="check_rules", arguments={}),
    )
    results = [
        result.model_copy(update={"attempts": 3, "tool_calls": calls}) for result in base.results
    ]
    return build_report(
        results,
        model=base.model,
        provider=base.provider,
        started_at=STARTED_AT,
        dataset_path=Path("data/postings.jsonl"),
        dataset_sha256=base.dataset_sha256,
        coverage_problems=[],
        reasoning="on",
        version="agent",
        prompt_sha256=prompt_sha256,
    )


def test_load_runs_accepts_a_workflow_run_and_an_agent_run_with_different_prompts(
    tmp_path: Path,
) -> None:
    """The two versions have different prompts by design and still share a table."""
    workflow = _save(_report("gemini-2.5-flash", ["APPLY", "SKIP", "SKIP", "SKIP"]), tmp_path, "w")
    agent = _save(_agent_report(["APPLY", "SKIP", "SKIP", "SKIP"]), tmp_path, "a")

    reports = load_runs([workflow, agent])

    assert [report.version for report in reports] == ["workflow", "agent"]
    assert reports[0].prompt_sha256 != reports[1].prompt_sha256


def test_load_runs_rejects_two_agent_runs_with_different_prompts(tmp_path: Path) -> None:
    """Within one version a changed prompt makes runs incomparable."""
    first = _save(_agent_report(["APPLY", "SKIP", "SKIP", "SKIP"]), tmp_path, "a")
    changed = _agent_report(["APPLY", "SKIP", "SKIP", "SKIP"], prompt_sha256="other-prompt")
    second = _save(changed, tmp_path, "b")

    with pytest.raises(CompareError, match="not comparable"):
        load_runs([first, second])


def test_agent_run_is_named_as_such_in_the_model_table() -> None:
    """A row of the model table says when the agent produced it."""
    rows = rows_from_reports([_agent_report(["APPLY", "SKIP", "SKIP", "SKIP"])])

    assert rows[1].name == "gemini-2.5-flash (agent, reasoning on)"


def test_version_table_shows_model_and_tool_calls_per_posting() -> None:
    """Each run is one row; repeated runs of a version are numbered."""
    workflow = _report("gemini-2.5-flash", ["APPLY", "SKIP", "SKIP", "SKIP"])
    agent = _agent_report(["APPLY", "SKIP", "SKIP", "PREPARE"])

    table = format_version_table([workflow, agent, agent])

    lines = table.splitlines()
    assert lines[0].endswith("| Avg. model calls | Avg. tool calls |")
    assert lines[2] == "| Workflow, run 1 | 75.0% | 0.50 | $0.08 | 2.0 s | 1.00 | 0.00 |"
    assert lines[3] == "| Agent, run 1 | 100.0% | 0.00 | $0.08 | 2.0 s | 3.00 | 2.00 |"
    assert lines[4].startswith("| Agent, run 2 |")


def test_version_table_treats_an_old_run_file_as_one_model_call(tmp_path: Path) -> None:
    """Run files from before Phase 4 load as workflow runs with one call per posting."""
    report = _report("gemini-2.5-flash", ["APPLY", "SKIP", "SKIP", "SKIP"])
    new_fields = {"version", "mean_model_calls", "mean_tool_calls", "tool_call_counts"}
    path = tmp_path / "old.json"
    path.write_text(report.model_dump_json(exclude=new_fields), encoding="utf-8")

    loaded = load_runs([path])[0]

    assert (loaded.version, loaded.mean_model_calls) == ("workflow", None)
    assert "| Workflow, run 1 | 75.0% | 0.50 | $0.08 | 2.0 s | 1.00 | 0.00 |" in (
        format_version_table([loaded])
    )


def test_main_prints_the_version_table_on_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--table versions`` switches to the workflow-vs-agent table."""
    workflow = _save(_report("gemini-2.5-flash", ["APPLY", "SKIP", "SKIP", "SKIP"]), tmp_path, "w")
    agent = _save(_agent_report(["APPLY", "SKIP", "SKIP", "SKIP"]), tmp_path, "a")

    exit_code = main([str(workflow), str(agent), "--table", "versions"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "| Agent, run 1 |" in output
    assert "Baseline" not in output
