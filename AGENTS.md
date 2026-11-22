# AGENTS.md

## Purpose
Job Fit Assistant reads a job posting and the candidate's CV and returns a structured
`JobDecision` (`APPLY` / `PREPARE` / `SKIP`). Hard rules (location, seniority, English) live
in code and override the LLM. It measures accuracy, cost and latency on a labeled test set.

## Commands
- Install: `uv sync` · Test: `uv run pytest` · Lint: `uv run ruff check .` · Format: `uv run ruff format .`
- Eval (dev set): `uv run python -m job_fit.eval --model <model> [--version agent|extractor] [--runs 3] [--judge]` · Page: `uv run python -m job_fit.ui`
- Hook: `.claude/settings.json` runs ruff and pytest after every edit of a `.py` file or `pyproject.toml`.

## Folders
- `src/job_fit/` – package (schema, rules, llm, workflow, dataset, metrics, pricing, eval, compare, agent, tools, extract, ui, judge, tracing)
- `data/cv.md` – candidate CV (git-ignored); `data/postings.jsonl` – labeled set, split into `postings_dev.jsonl` and `postings_heldout.jsonl`; `data/results/` – run outputs (git-ignored)
- `tests/` – pytest suite; `docs/results.md` – result tables; `docs/architecture.md` – why each LLM call exists; `.claude/hooks/` – post-edit check

## Rules
- Plan first: write the phase plan to `PLAN.md`, wait for approval, then code. One phase at a time; commit at the end as `phase-N: <short description>`.
- The held-out set is run once, at the end of Phase 11 (`--final-heldout`). Until then never evaluate it, read model output on it, or tune against it.
- Show ruff and pytest output as evidence before saying "done"; run `ruff format` before a commit.
- No Turkish anywhere in code: identifiers, comments, docstrings, strings, file names.
- Every function, including tests and private helpers, has a Google-style docstring (ruff `D`).
- API keys only from `.env`; never commit `.env` or service-account JSON keys.
- Do not log personal data beyond `data/cv.md` and posting texts; traces mask the CV. Labels come from the candidate, never from a model.
- LLM calls are mocked in tests. The tool only recommends; it never applies or sends.
- When in doubt, ask.

## Details
[PROCESS.md](PROCESS.md) (phases, done criteria) · [PLAN.md](PLAN.md) (plan, status, decisions) · [docs/python-style.md](docs/python-style.md) (read before coding)

## Lessons learned
Each line: what went wrong → the rule that prevents it.
- Heredocs with apostrophes fail in the Bash tool, there is no bare `python`, and `load_dotenv()` searches from the calling file → write scripts with the file tool, run them with `uv run python`, pass the `.env` path from outside the repo.
- A test expectation computed by hand was wrong → derive expected values from the constants (`ERROR_COSTS`) and run the test before trusting it.
- A model listed for the project still returned 404 → probe a model with one call, and read prices from the pricing page, before planning a run.
- Two eval runs were cut off by a quota → an aborted run is repeated or reported as "not run", never patched together.
- One agent run scored 85.7%, the next 61.9% → run a configuration 3 times (`--runs 3`); a difference below the noise figure in `docs/results.md` is not an improvement.
- The judge model gave the same score as the candidate on 4 of 10 rationales → check a judge against human scores first; an unchecked judge score decides nothing.
- A tool's code changed while its description, and so the prompt hash, stayed the same → the hash covers what the model is told, not what a tool returns; replay the recorded tool calls through old and new code before reusing an old row.
- A sub-agent was planned before the errors were read; the rationales then showed the mistake was in the judgment, not the facts → read the wrong decisions first, and write the keep/drop rule down before the run.
- Tuning a prompt against the test set fits it and breaks comparability → no change to prompts, schema or tool descriptions without re-running every compared row.
