# AGENTS.md

## Purpose
Job Fit Assistant reads a job posting and the candidate's CV and returns a structured
`JobDecision` (`APPLY` / `PREPARE` / `SKIP`). Hard rules (location, seniority, English) live
in code and override the LLM. It measures accuracy, cost and latency on a labeled test set.

## Commands
- Install: `uv sync` · Test: `uv run pytest` · Lint: `uv run ruff check .` · Format: `uv run ruff format .`
- Eval: `uv run python -m job_fit.eval --model <model> [--version agent] --allow-incomplete` · Page: `uv run python -m job_fit.ui`
- Hook: `.claude/settings.json` runs ruff and pytest after every edit of a `.py` file or `pyproject.toml`.

## Folders
- `src/job_fit/` – package (schema, rules, llm, workflow, dataset, metrics, pricing, eval, compare, agent, tools, ui)
- `data/cv.md` – candidate CV (git-ignored); `data/postings.jsonl` – labeled test set; `data/results/` – run outputs (git-ignored)
- `tests/` – pytest suite; `docs/results.md` – result tables; `.claude/hooks/` – post-edit check

## Rules
- Plan first: write the phase plan to `PLAN.md`, wait for approval, then code.
- One phase at a time; commit at the end as `phase-N: <short description>`.
- Show ruff and pytest output as evidence before saying "done"; run `ruff format` before a commit.
- No Turkish anywhere in code: identifiers, comments, docstrings, strings, file names.
- Every function, including tests and private helpers, has a Google-style docstring (ruff `D`).
- API keys only from `.env`; never commit `.env` or service-account JSON keys.
- Do not log personal data beyond `data/cv.md` and posting texts.
- LLM calls are mocked in tests. The tool only recommends; it never applies or sends.
- When in doubt, ask.

## Details
[PROCESS.md](PROCESS.md) (phases, done criteria) · [PLAN.md](PLAN.md) (plan, status, decisions) · [docs/python-style.md](docs/python-style.md) (read before coding)

## Lessons learned
Each line: what went wrong → the rule that prevents it.
- Heredocs with apostrophes fail in the Bash tool, and there is no bare `python` → write scripts with the file tool, run them with `uv run python`.
- `load_dotenv()` searches from the calling file, not the working directory → a script outside the repo must pass the `.env` path.
- A test expectation computed by hand was wrong → derive expected values from the constants (`ERROR_COSTS`) and run the test before trusting it.
- A model listed for the project still returned 404 → probe a model with one call, and read prices from the pricing page, before planning a run.
- Two eval runs were cut off by a quota → an aborted run is repeated or reported as "not run", never patched together.
- One agent run scored 85.7%, the next 61.9% → on 21 postings one run is not a result; repeat it, and treat one posting of difference as noise.
- Tuning a prompt against the test set fits it and breaks comparability → no change to prompts, schema or tool descriptions without re-running every compared row.
