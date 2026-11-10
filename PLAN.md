# PLAN.md

## Status

| Phase | Status | Commit | Notes |
|---|---|---|---|
| 0 – Setup | ✅ | see `git log --grep phase-0` | uv project, ruff + pytest configured, 1 smoke test |
| 1 – Structured output | ✅ | see `git log --grep phase-1` | `gemini-2.5-flash` on Vertex; 3 samples run live; 75 tests |
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
| D11 | Newer Gemini models (3.x) answer only in the `global` location; `.env` uses `us-central1`, where only `gemini-2.5-flash` answered. Switch `GOOGLE_CLOUD_LOCATION` to `global`? | Phase 3 | Yes, before picking the 3 cloud models |
| D5 | A posting whose call fails even after retry: how is it scored in eval? | Phase 2 | Count as wrong, with the worst cost in its row; report the count separately |
| D6 | Weighted error: total over the set, or mean per posting? | Phase 2 | Report both; compare models on the mean |
| D7 | Local machine RAM / GPU VRAM for Ollama? | Phase 3 | Check before picking the local model |
| D8 | Is Claude enabled in Vertex Model Garden for your project? | Phase 3 | If not, use OpenRouter for it |

## Decided

- D1 (2026-11-10): `data/cv.md` is **not committed**. It is git-ignored; the repo carries
  `data/cv.example.md` as a template.
- D2 (2026-11-10): a hybrid role outside Ankara/Istanbul is `SKIP`, same as on-site.
- D3 (2026-11-10): a remote role not open to Türkiye is `SKIP`.
- D4 (2026-11-10): `english_risk` is true if the LLM flags it **or** a keyword check on
  the posting text matches.
- D9 (2026-11-10): Turkish words are allowed in code only as match data inside named
  keyword constants in `rules.py` (city spellings, English-requirement phrases).
  Everything else stays English.
- D10 (2026-11-10): the candidate supplies 3 real postings as text files in
  `data/samples/` (format in `data/samples/README.md`).

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

## Phase 1 – Structured output (done)

Source: [PROCESS.md §5 Phase 1](PROCESS.md#phase-1--structured-output-fixed-workflow)

### Before coding (candidate)
- [ ] Copy `data/cv.example.md` to `data/cv.md` and fill it in.
- [ ] Create `.env` from `.env.example`, set `GOOGLE_CLOUD_PROJECT`.
- [ ] `gcloud auth application-default login` (gcloud is installed), Vertex AI API enabled.
- [ ] 3 sample postings (D10), ideally one clear fit, one where a rule should fire
      (e.g. on-site outside Ankara/Istanbul, or junior), one English-heavy.

### Steps
1. [x] `uv add pydantic google-genai python-dotenv`.
       The `openai` package is added in Phase 3, together with the OpenRouter and Ollama
       providers, when they can be run and verified.
2. [x] `schema.py` – `JobDecision` with the exact fields and order from PROCESS.md
       (`rationale` first). Each field gets a `Field(description=...)`; descriptions are
       sent to the model inside the schema. `location` format is fixed by its description:
       `"City, Country"` for on-site/hybrid, `"Remote (<region>)"` for remote
       (e.g. `Remote (Türkiye)`, `Remote (Worldwide)`, `Remote (US only)`), so that the
       D3 rule can read it without adding a field to the schema.
3. [x] `rules.py` – pure functions, no I/O, no LLM:
       - `violated_rules(location, work_mode, seniority) -> list[str]` (takes plain values,
         so the Phase 4 `check_rules` tool can call it directly)
       - `has_english_risk(posting_text) -> bool` (keyword match, D4/D9)
       - `apply_rules(decision, posting_text) -> RuleOutcome` (new `JobDecision` + names
         of rules that fired; the input object is not mutated)
       Behavior:
       - seniority `intern` / `junior` → `SKIP`, `rule: seniority`
       - `onsite` or `hybrid` outside Ankara/Istanbul → `SKIP`, `rule: location` (D2)
       - `remote` whose region is not Türkiye / worldwide → `SKIP`, `rule: location` (D3)
       - English: `english_risk = llm_flag or keyword_match`; decision unchanged (D4)
       - `unknown` work mode, location or seniority never triggers an override: a wrongly
         skipped good job costs 3, a wasted application costs 1
       - text matching ignores case and Turkish characters (`İstanbul` = `istanbul`)
       - every override appends `rule: <name>` to `rationale`; several can apply
4. [x] `llm.py` – the only module that imports a provider SDK or reads `.env`:
       `generate_json(provider, model, system, user, schema, *, reasoning=None,
       timeout_s=60) -> LLMResponse` (`text`, `input_tokens`, `output_tokens`, `latency_s`).
       - `vertex`: `google-genai` client, `response_mime_type="application/json"`,
         `response_schema=schema`, `temperature=0`; tokens from `usage_metadata`
         (thinking tokens counted as output); latency with `time.perf_counter()`.
       - `openrouter` / `ollama`: accepted by the signature, raise a clear
         "not implemented until Phase 3" error.
       - Returns raw JSON text; it does not validate. Validation and retry belong to the
         workflow, as PROCESS.md describes.
5. [x] `workflow.py`:
       - `SYSTEM_PROMPT` constant and `build_user_prompt(posting_text, cv)` with the
         posting and CV in clearly delimited blocks
       - `decide(posting_text, cv, *, model, provider="vertex", llm=generate_json)
         -> DecisionResult` — `llm` is injectable so tests pass a fake
       - validate with `JobDecision.model_validate_json`; on `ValidationError` retry once,
         then raise `DecisionError` (model, attempts, validation message; no CV text in
         the error)
       - apply `rules.py` to the validated output
       - `DecisionResult`: final `JobDecision`, the LLM's original decision, rules fired,
         input tokens, output tokens, latency (summed over attempts), attempts
       - CLI: `uv run python -m job_fit.workflow --posting <file> --model <id>
         [--cv data/cv.md]` prints the decision as JSON plus tokens and latency
6. [x] Tests (no network, no real LLM):
       - `test_schema.py`: valid JSON parses; invalid `decision` value rejected; missing
         field rejected; first field is `rationale`
       - `test_rules.py` (≥ 5 required, ~12 planned): on-site Bursa → SKIP; hybrid Izmir →
         SKIP; on-site `İstanbul` kept; remote Türkiye kept; remote US-only → SKIP;
         junior → SKIP; intern → SKIP; senior kept; unknowns never override; English
         keyword sets the flag without changing the decision; LLM flag kept when no
         keyword; two rules both appended; input not mutated
       - `test_workflow.py`: valid on first try; invalid then valid (2 attempts, tokens
         summed); invalid twice → `DecisionError`; rule override keeps the LLM's
         original decision in the result
       - `test_llm.py`: unknown provider rejected; Vertex response mapped to
         `LLMResponse` using a fake client
7. [x] Pick the Phase 1 model: a Flash-tier Gemini on Vertex. The exact model id is
       looked up at implementation time, not assumed.
8. [x] Live run on the 3 sample postings (`data/samples/`); show the full output,
       tokens and latency for each.
9. [x] `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`; show output.
10. [x] Commit `phase-1: structured output workflow`; update the status table.

### Live run results (2026-11-10, `gemini-2.5-flash`, `us-central1`)

| Posting | LLM decision | Final | Rules fired | In / out tokens | Latency |
|---|---|---|---|---|---|
| accenture-software-engineer | APPLY | APPLY | none | 2445 / 2038 | 16.1 s |
| izmir-senior-software-engineer | SKIP | SKIP | location | 2619 / 2649 | 15.5 s |
| miny-engineer-ai-inference-product | SKIP | SKIP | none | 2415 / 2700 | 18.5 s |

All three answers validated on the first attempt.

### Findings to carry into Phase 2
- Miny came out `SKIP` because Go is missing, while the example row in PROCESS.md labels
  it `APPLY`. That is the most expensive error in the cost matrix (missed good job, 3).
  Do not tune the prompt on one example; measure on the labeled set first.
- Output tokens (2000–2700) are far above the size of the JSON: `gemini-2.5-flash`
  thinks by default and thinking tokens are counted as output. This drives cost and
  latency; compare reasoning on/off in Phase 3.
- Extraction quirks seen: company `Brandsight` instead of `Miny`; seniority `unknown`
  for Accenture ("3+ years"); `english_risk` true for postings merely written in English.
  Location and work mode were reported as `unknown` when not stated (no guessing).
- The live run did not show a rule overriding an `APPLY` (the model already said `SKIP`
  for Izmir). That path is covered by unit tests only.
- Listing models needs a quota project on the local credentials
  (`GOOGLE_CLOUD_QUOTA_PROJECT` or `gcloud auth application-default set-quota-project`);
  `generate_content` works without it.

**Done when:** 3 sample postings run end to end, output validates against the schema,
tokens and latency are printed, `rules.py` has ≥ 5 passing tests.

**Not in this phase:** OpenRouter/Ollama providers, eval metrics, cost in USD, agent.

**Risks**
- Rules act on fields the LLM extracts (location, work mode, seniority). A wrong
  extraction means a wrong override. Keeping the LLM's original decision in
  `DecisionResult` makes this measurable in Phase 2.
- The `Remote (<region>)` convention depends on the model following the field
  description; an unrecognized region is treated as unknown (no override) only when
  the text is literally `unknown`, otherwise it is `SKIP`. Reviewed after the live run.
- Live calls spend Vertex credits: 3 postings on a Flash-tier model, a few cents at most.

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
