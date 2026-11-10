# AGENTS.md

## Purpose
Job Fit Assistant reads a job posting and the candidate's CV and returns a structured
`JobDecision` (`APPLY` / `PREPARE` / `SKIP`). Hard rules (location, seniority, English)
live in code and override the LLM. The project measures accuracy, cost and latency of
different models, and a fixed workflow vs. an agent, on a labeled test set.

## Commands
- Install: `uv sync`
- Test: `uv run pytest`
- Lint: `uv run ruff check .` / format: `uv run ruff format .`
- Eval (Phase 2+): `uv run python -m job_fit.eval --model <model>`

## Folders
- `src/job_fit/` – package (schema, rules, llm, workflow, agent, tools, eval)
- `data/cv.md` – candidate CV; `data/postings.jsonl` – labeled test set
- `data/results/` – eval run outputs (git-ignored)
- `tests/` – pytest suite; `docs/results.md` – model comparison tables

## Rules
- Plan first: write the phase plan to `PLAN.md`, wait for approval, then code.
- One phase at a time; commit at the end as `phase-N: <short description>`.
- After every change run ruff and pytest; show the output as evidence.
- No Turkish anywhere in code: identifiers, comments, docstrings, strings, file names.
- Every function, including tests and private helpers, has a Google-style docstring
  (enforced by ruff `D` rules).
- API keys only from `.env`; never commit `.env` or service-account JSON keys.
- Do not log personal data beyond `data/cv.md` and posting texts.
- LLM calls are mocked in tests. The tool only recommends; it never applies or sends.
- When in doubt, ask.

## Details
- Full process, phases and done criteria: [PROCESS.md](PROCESS.md)
- Current plan and status: [PLAN.md](PLAN.md)
- Python best practices (read before writing code): [docs/python-style.md](docs/python-style.md)

## Lessons learned
_(Filled in during Phase 5.)_
