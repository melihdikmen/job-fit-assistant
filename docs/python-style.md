# Python best practices

Rules for all Python code in this repo. Ruff enforces part of it (`E, F, I, UP, B, D`);
the rest is checked in review.

## Language and naming
- English only: identifiers, comments, docstrings, strings, log and error messages.
- `snake_case` for functions and variables, `PascalCase` for classes,
  `UPPER_SNAKE_CASE` for module-level constants. Private helpers start with `_`.
- Names say what a thing is (`posting_text`, not `data` or `s`).
- No magic numbers or strings: lift them into named constants
  (`ALLOWED_CITIES`, `MAX_AGENT_STEPS`, `ERROR_COSTS`).

## Docstrings and comments
- Every module, class and function has a Google-style docstring, tests and private
  helpers included: one summary line, then `Args:`, `Returns:`, `Raises:` when they apply.
- Comments explain *why*, not *what*. Delete commented-out code.

## Types
- Type hints on every function parameter and return value.
- Modern syntax: `list[str]`, `dict[str, int]`, `str | None`.
- `Literal` or `Enum` for closed sets of values, never free strings.
- Pydantic models for data that crosses a boundary (LLM output, JSONL rows, saved
  results); `dataclass(frozen=True)` for internal value objects.

## Functions and modules
- Small functions with one job. If a docstring needs "and", split the function.
- Keep logic pure and I/O at the edges: `rules.py` and metric functions take values and
  return values; file, network and environment access live in `llm.py`, CLI code and
  loaders.
- No mutable default arguments, no module-level mutable state.
- Return new objects instead of mutating inputs (`model_copy(update=...)`).
- Each runnable module guards its entry point with `if __name__ == "__main__":`.
- One module must not know another's internals: the rest of the code calls `llm.py`
  through its public function and never imports a provider SDK directly.

## Errors
- Raise specific exceptions with a clear message; define project exceptions
  (e.g. `DecisionError`) instead of raising bare `Exception`.
- Never use bare `except:` or swallow errors silently. Catch the narrowest type.
- Chain with `raise ... from err` to keep the original cause.
- Fail early on bad input (missing file, invalid label, unknown provider).

## Files, config and secrets
- `pathlib.Path` instead of string paths; always pass `encoding="utf-8"` when opening
  text files (the default differs on Windows).
- Configuration comes from environment variables loaded from `.env`; read them in one
  place, not scattered through the code.
- Never hard-code or log API keys, project ids or credentials.

## LLM calls
- Every call has a timeout and a bounded number of retries.
- Measure latency with `time.perf_counter()`, not `time.time()`.
- Always return token counts and latency together with the result.
- Prompts are constants or templates in one place, not built inline across modules.

## Logging and output
- Use the `logging` module in library code; `print` only for CLI output meant for the user.
- Do not log personal data beyond the CV and posting texts.

## Tests
- One behavior per test; the test name states it (`test_onsite_outside_allowed_city_is_skip`).
- Arrange, act, assert, in that order.
- `pytest.mark.parametrize` for the same check over several inputs.
- No network, no real LLM, no real clock in tests: mock or inject them.
- Test both the positive and the negative case of every rule.

## Dependencies and tooling
- Add packages with `uv add` (dev tools with `uv add --dev`); never edit the lockfile
  by hand or use `pip install`.
- Run `uv run ruff check .`, `uv run ruff format .` and `uv run pytest` after every change.
- Do not silence ruff with `# noqa` unless the reason is written next to it.
