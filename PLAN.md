# PLAN.md

## Status

| Phase | Status | Commit | Notes |
|---|---|---|---|
| 0 – Setup | ✅ | see `git log --grep phase-0` | uv project, ruff + pytest configured, 1 smoke test |
| 1 – Structured output | ⬜ | | |
| 2 – Test set and eval | ⬜ | | |
| 3 – Model selection | ⬜ | | |
| 4 – Agent | ⬜ | | |
| 5 – Harden | ⬜ | | |
| 6 – UI (optional) | ⬜ | | |

Legend: ⬜ not started · 🟡 in progress · ✅ done

How this file works: the current phase has a detailed, step-by-step plan. Later phases
have a roadmap only; each is expanded into detailed steps at the start of that phase
and approved before any code is written ([PROCESS.md §4](PROCESS.md#4-working-rules-for-claude-code)).

---

## Open decisions (need the candidate's answer)

| # | Question | Needed by | Proposal |
|---|---|---|---|
| D2 | Hybrid role in a city other than Ankara/Istanbul: `SKIP`? | Phase 1 | Yes, treat like on-site |
| D3 | Remote role not open to Türkiye (e.g. "Remote – US only"): `SKIP`? | Phase 1 | Yes; schema needs a way to express it (see Phase 1) |
| D4 | `english_risk`: trust the LLM flag only, or also a keyword check in code? | Phase 1 | LLM flag OR keyword match on posting text |
| D5 | A posting whose call fails even after retry: how is it scored in eval? | Phase 2 | Count as wrong, with the worst cost in its row; report the count separately |
| D6 | Weighted error: total over the set, or mean per posting? | Phase 2 | Report both; compare models on the mean |
| D7 | Local machine RAM / GPU VRAM for Ollama? | Phase 3 | Check before picking the local model |
| D8 | Is Claude enabled in Vertex Model Garden for your project? | Phase 3 | If not, use OpenRouter for it |

## Decided

- D1 (2026-11-10): `data/cv.md` is **not committed**. It is git-ignored; the repo carries
  `data/cv.example.md` as a template.

## What the candidate does (Claude cannot)

- Write `data/cv.md` (Phase 1), collect and label ≥ 20 postings (Phase 2).
- `gcloud auth application-default login`, set the Vertex project in `.env` (Phase 1).
- Set a budget alert in the Google Cloud console (Phase 3).
- Write or approve the "which model I chose" and "is an agent needed" paragraphs —
  Claude drafts from the numbers, the candidate owns the conclusion (Phases 3, 4).
- Create the GitHub repo and approve the push (Phase 5).

---

## Phase 0 – Setup (done)

Source: [PROCESS.md §5 Phase 0](PROCESS.md#phase-0--setup)

1. [x] `uv init --package --name job-fit-assistant --python 3.12` (also runs `git init`).
       Import package is `job_fit`: rename `src/job_fit_assistant/` to `src/job_fit/` and
       set `[tool.uv.build-backend] module-name = "job_fit"`.
2. [x] `uv add --dev pytest ruff`.
3. [x] Ruff config in `pyproject.toml`: `target-version = "py312"`, `line-length = 100`,
       rules `E, F, I, UP, B, D`, `pydocstyle.convention = "google"`. Markdown files are
       excluded so ruff does not reformat code snippets inside the docs.
4. [x] Pytest config: `testpaths = ["tests"]`.
5. [x] Folder skeleton:
       - `src/job_fit/__init__.py` (module docstring only)
       - `tests/test_smoke.py` – imports `job_fit`. Without a test pytest exits with
         code 5 ("no tests collected"), which would fail the done check.
       - `data/cv.example.md` (template; the real `data/cv.md` is git-ignored, D1)
       - `docs/results.md` with the empty tables from PROCESS.md Phases 3 and 4
6. [x] `.env.example` exactly as in PROCESS.md.
7. [x] `.gitignore`: `.env`, `data/results/`, `__pycache__/`, `.venv/`, `.pytest_cache/`,
       `.ruff_cache/`, service-account key patterns (`*-sa.json`, `service-account*.json`),
       and `data/cv.md` (D1).
8. [x] Verify and show output: `uv sync`, `uv run ruff check .`,
       `uv run ruff format --check .`, `uv run pytest`.
9. [x] Commit `phase-0: project setup`; update the status table.

**Done when:** pytest runs, ruff is clean, first commit exists.

---

## Phase 1 – Structured output (roadmap)

**Dependencies:** `pydantic`, `google-genai`, `openai` (OpenAI-compatible client for
OpenRouter and Ollama), `python-dotenv`.

**Design**
- `schema.py` – `JobDecision` exactly as in PROCESS.md, `rationale` first. Add `Field`
  descriptions; they are sent to the model as part of the schema. Per D3, `location`
  description asks for country too (e.g. "Remote – Türkiye").
- `llm.py` – one entry point, provider and model as parameters:
  `generate_structured(provider, model, system, user, schema) -> LLMResult`
  (`parsed`, `raw_text`, `input_tokens`, `output_tokens`, `latency_s`).
  - `vertex`: `google-genai` with `response_mime_type="application/json"` and
    `response_schema`; tokens from `usage_metadata`.
  - `openrouter`, `ollama`: `openai` SDK with a JSON-schema `response_format`;
    tokens from `usage`.
  - A `reasoning` option is part of the signature from the start (used in Phase 3).
- `rules.py` – pure functions, no LLM: `apply_rules(decision, posting_text) -> JobDecision`.
  - Seniority `intern` / `junior` → `SKIP`.
  - Location: `onsite` (and `hybrid`, per D2) outside Ankara/Istanbul → `SKIP`;
    remote not open to Türkiye → `SKIP` (D3). City matching ignores case and
    Turkish characters (`İstanbul` = `Istanbul`).
  - English: sets `english_risk = true` (D4), never changes the decision by itself.
  - `unknown` values never trigger an override.
  - Every override appends `rule: <name>` to `rationale`; several can apply.
- `workflow.py` – `decide(posting_text, cv) -> DecisionResult` (final decision, the LLM's
  original decision, which rules fired, tokens, latency). One retry on validation error,
  then a clear `DecisionError`. Small CLI to run one posting file and print the result.

**Tests:** ≥ 5 for `rules.py` (PROCESS requires it); retry logic with a mocked LLM.

**Done when:** 3 sample postings run end to end, output validates, tokens and latency
are printed, rules have ≥ 5 passing tests.

**Risk:** rules act on fields the LLM extracts (location, seniority). A wrong extraction
means a wrong override; keeping the LLM's original decision in the result makes this
visible in Phase 2.

---

## Phase 2 – Test set and evaluation (roadmap)

- `data/postings.jsonl` in the PROCESS.md format, ≥ 20 postings, ≥ 5 per label,
  labeled by the candidate.
- Dataset validation (unique ids, valid labels, balance) runs at the start of every eval
  and fails loudly.
- `eval.py`, `uv run python -m job_fit.eval --model <model> [--provider ...] [--limit N]`:
  - accuracy, 3×3 confusion matrix, weighted error (cost matrix from PROCESS.md, D6)
  - majority-label baseline printed next to the model
  - total tokens, estimated cost (USD) from a dated price table, mean latency
  - list of wrong postings with rationale and which rules fired
  - also scores the LLM's decision *before* rules, to show what the rules add
  - saves `data/results/<date>_<model>.json`
- Failed postings handled per D5.
- Tests for metric functions on hand-made inputs (no LLM).

**Done when:** one command prints all metrics including the baseline.

---

## Phase 3 – Model selection and cost (roadmap)

- Shortlist 3 cloud models (strong / mid / cheap-fast), Vertex first. Exact model ids
  and list prices are checked at the start of the phase, not assumed now.
- Claude via Model Garden uses a different SDK (`anthropic[vertex]`) than Gemini; add it
  as its own provider only if D8 says it is enabled, otherwise use OpenRouter.
- OpenRouter calls pin the provider order and disable fallbacks.
- Local: one 7–8B model, 4-bit, via Ollama. Memory estimate written first
  (8B × 4 bits ÷ 8 ≈ 4 GB + context), checked against D7.
- Reasoning off vs. on for one model that supports it.
- Budget alert set before the full run (candidate).
- Fill `docs/results.md`; Claude drafts the Pareto / privacy paragraph from the numbers.

**Done when:** the table is filled and the choice is justified in writing.

---

## Phase 4 – Agent version (roadmap)

**Dependencies:** `langchain` plus the chat-model integration(s) for the chosen model.

- `tools.py` with `@tool`: `check_rules` (calls `rules.py`), `get_cv_section` (splits
  `cv.md` by headings – CV needs `## Experience`, `## Projects`, `## Skills`),
  `convert_currency` (fixed, dated USD/EUR → TRY rate, stated in the output).
- `agent.py`: `create_agent(model, tools, system_prompt, response_format=JobDecision)`.
  Step limit of 6 (mechanism checked against the installed LangChain version).
  Every tool call logged with its arguments. `rules.py` still runs on the final output.
- Eval gains `--version workflow|agent`; agent tokens are summed over all model calls.
- Same model for both versions; comparison table and conclusion in `docs/results.md`.

**Done when:** both versions pass eval, table and conclusion are written.

---

## Phase 5 – Harness, harden, humans (roadmap)

- Tests: positive + negative per rule, invalid JSON rejected by `schema.py`, ≥ 1 per
  tool, all LLM calls mocked, no network.
- Claude Code hook in `.claude/settings.json`: after `Edit`/`Write`, run
  `uv run ruff check .` and `uv run pytest -q`.
- `AGENTS.md` "Lessons learned" filled from real mistakes in Phases 1–4.
- README: purpose, how to run, result tables, chosen model and why, and the statement
  that the tool only recommends.
- Push to GitHub after checking that no `.env`, key file or CV is tracked.

**Done when:** tests and lint pass, the hook works, README is complete, code is on GitHub.

---

## Phase 6 – UI, optional (roadmap)

- Gradio app: paste a posting, choose model and version, see `JobDecision` and cost.
