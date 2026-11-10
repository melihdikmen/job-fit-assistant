# Job Fit Assistant

Give it a job posting and a CV; it returns a structured `APPLY` / `PREPARE` / `SKIP`
decision with rationale, and measures how accurately, how fast and at what cost different
models make that decision on a labeled test set.

## Quick start

```
uv sync
cp .env.example .env   # then fill in your values
gcloud auth application-default login
uv run pytest
```

See [AGENTS.md](AGENTS.md) for commands and rules, [PROCESS.md](PROCESS.md) for the full
process, and [PLAN.md](PLAN.md) for current status.

_Full README (results, chosen model) is written in Phase 5._
