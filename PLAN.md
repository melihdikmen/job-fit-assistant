# PLAN.md

## Status

| Phase | Status | Commit | Notes |
|---|---|---|---|
| 0 – Setup | ✅ | see `git log --grep phase-0` | uv project, ruff + pytest configured, 1 smoke test |
| 1 – Structured output | ✅ | see `git log --grep phase-1` | `gemini-2.5-flash` on Vertex; 3 samples run live; 75 tests |
| 2 – Test set and eval | ✅ | see `git log --grep phase-2` | `gemini-2.5-flash`: 66.7% vs. 57.1% baseline, mean w.err 0.48 vs. 1.14; 21 postings, PREPARE has only 3 (D15); 149 tests |
| 3 – Model selection | ✅ | see `git log --grep phase-3` | chosen: `gemini-2.5-flash` with reasoning on, 71.4% / mean w.err 0.43 / $0.64 per 100 / 11.9 s; no local and no open-weight run (D7, D20); 195 tests |
| 4 – Agent | ✅ | see `git log --grep phase-4` | no agent needed (D27): workflow 69.0% / mean w.err 0.45 / $0.64 / 12.0 s vs. agent 73.8% / 0.26 / $0.75 / 14.9 s (means of 2 runs; agent runs differ by 5 postings, same tool calls every time); 265 tests |
| 5 – Harden | ✅ | see `git log --grep phase-5` | 322 tests; post-edit hook shown working; code review by a separate agent, 4 findings fixed (D33) and 8 reported; README; public repo https://github.com/melihdikmen/job-fit-assistant |
| 6 – UI (optional) | 🟡 | see `git log --grep phase-6` | local Gradio page (`uv run python -m job_fit.ui`): posting, 3 measured model choices, workflow or agent, decision and cost; localhost only, analytics off; 350 tests; committed, **push waits for the candidate's go-ahead** (D37) |

Legend: ⬜ not started · 🟡 in progress · ✅ done

How this file works: the current phase has a detailed, step-by-step plan. Later phases
have a roadmap only; each is expanded into detailed steps at the start of that phase
and approved before any code is written ([PROCESS.md §4](PROCESS.md#4-working-rules-for-claude-code)).

---

## Open decisions (need the candidate's answer)

| # | Question | Needed by | Proposal |
|---|---|---|---|

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

- D5 (2026-11-10): a posting whose call fails even after the retry counts as wrong and is
  charged the worst cost of its actual row; the number of failures is reported separately.
- D6 (2026-11-10): weighted error is reported as total and as mean per posting; models
  are compared on the mean.
- D12 (2026-11-10): `data/postings.jsonl` is committed. The `note` field carries no
  private remarks.
- D13 (2026-11-10): rows are added with `python -m job_fit.dataset add`; `postings.jsonl`
  stays the single source of truth.
- D14 (2026-11-10): the report prints all three constant predictors, majority marked.

- D15 (2026-11-10): Phase 2 is closed with 3 PREPARE postings instead of 5. Eval runs on
  this set need `--allow-incomplete` until two more PREPARE postings are added; adding
  them changes the dataset hash, so earlier runs must be repeated to stay comparable.
- D16 (2026-11-10): `data/postings/_labels.txt` (the candidate's working notes) is
  git-ignored; the posting texts and `postings.jsonl` are committed.

- D11 (2026-11-10, reversed the same day): `GOOGLE_CLOUD_LOCATION` is now `global` in the
  candidate's `.env`, so the newer Gemini models are reachable. `.env.example` keeps
  `us-central1` as PROCESS.md specifies. The Phase 2 reference run was made in
  `us-central1`; rerun `gemini-2.5-flash` in `global` at the start of Phase 3 so every
  row of the comparison table comes from the same location (latency can differ).

- D7 (2026-11-10): **no local model run.** The machine has 7.8 GB RAM and a GTX 1050
  with 3 GB VRAM; a 7–8B 4-bit model (about 4 GB + context) does not fit in VRAM. This
  departs from PROCESS.md Phase 3. The memory estimate is still written to
  `docs/results.md` and the "Local (Ollama)" row is marked "not run" with the reason.
  The `ollama` provider is still implemented and unit-tested with a fake client (it
  shares its code path with OpenRouter), but it is not exercised live.
- D17 (2026-11-10): the three cloud models are three Gemini tiers on Vertex (Pro /
  Flash / Flash-Lite). The OpenRouter account has no credit, so OpenRouter is used only
  with **free** (`:free`) models; no Claude row.

- D18 (2026-11-10): no PREPARE postings are added now. Phase 3 runs on the same 21
  postings as Phase 2 (dataset SHA-256 `296805a932ba…`), with `--allow-incomplete`.
- D19 (2026-11-10): the candidate accepts that the real `data/cv.md` is sent to the
  providers of free OpenRouter models, which may log prompts or train on them.

- D20 (2026-11-10): the free OpenRouter rows are recorded as "not run". Both full runs
  were stopped by OpenRouter's daily free-model quota (HTTP 429 `free-models-per-day`)
  after 8 postings; aborted runs are not saved and not patched into the table. The
  candidate decided not to repeat them: a free endpoint would not be used in practice.
  The `openrouter` provider itself is verified by two one-posting live runs.

- D21 (2026-11-10): the candidate confirmed the "Which model I chose and why" paragraph
  in `docs/results.md`. The chosen configuration is `gemini-2.5-flash` with reasoning
  on; Phase 4 compares workflow and agent on it.

- D22 (2026-11-11): the candidate could not find the prices; Claude read them from the
  raw HTML of the Vertex AI pricing page. Global, standard tier, per 1M tokens:
  `gemini-4-carbon` 2.00 / 10.00 through 2026-11-30 (promotional, 50% credits back), then
  4.00 / 20.00; `gemini-3.1-flash-lite` 0.25 / 1.50; `gemini-2.5-flash` 0.30 / 2.50. All
  match `pricing.py`, so no number in `docs/results.md` changed.

- D23 (2026-11-11): the agent gets the CV **only through `get_cv_section`**; its user
  message holds the posting alone. With the full CV in the prompt the tool would have
  no purpose.
- D24 (2026-11-11): live runs are 2 agent runs + 1 new workflow run; the Phase 3 run of
  2026-11-10 is the other workflow run. One run per version cannot separate them
  (noise is about 1 posting).
- D25 (2026-11-11): an agent posting that hits the step limit or returns no valid
  `JobDecision` counts as failed (D5). No fallback to the workflow, which would hide
  the agent's failures inside the agent's numbers.
- D26 (2026-11-11): `convert_currency` uses a fixed, dated rate in named constants,
  stated in the tool description and in `docs/results.md`.

- D27 (2026-11-11): the candidate confirmed the "Is an agent needed for this problem?"
  paragraph in `docs/results.md`: no agent. The fixed workflow on `gemini-2.5-flash`
  with reasoning on is the version in use; the agent code stays in the repo for the
  comparison only.

- D28 (2026-11-11): the GitHub repository is **public**. The pre-push check of Phase 5
  step 9 therefore covers the posting texts and this plan as well as secrets.
- D29 (2026-11-11): the corporate e-mail address must not be published. The author and
  committer of every commit are rewritten before the first push (address: D32).
- D30 (2026-11-11): the hook runs only when the edited file is `.py` or
  `pyproject.toml`.
- D31 (2026-11-11): the branch is renamed to `main` before the first push.
- D33 (2026-11-11): fix review findings R3 and R4 in the location rule. Claude's choices
  inside that decision: `Remote (Europe)` does not force a SKIP (ambiguous for a
  candidate in Türkiye, and a skipped good job is the costliest error); a district
  without its city (`Levent`) still counts as a place outside the allowed cities.
- D32 (2026-11-11): at the candidate's request Claude created the repository itself
  (`job-fit-assistant`, public, empty) through the GitHub API, with the token of the
  candidate's browser login in Git Credential Manager. The commit author is the
  account's GitHub `noreply` address.
- The candidate answered D28–D31 on 2026-11-11 without objecting to the Phase 5 plan;
  Claude took that as approval of the plan.

- D34 (2026-11-11): `gradio` is a normal runtime dependency, not an optional extra.
- D35 (2026-11-11): the page offers only configurations measured in Phase 3:
  `gemini-2.5-flash` reasoning on (default), `gemini-2.5-flash` reasoning off and
  `gemini-3.1-flash-lite`. `gemini-4-carbon` is left out. The agent can be combined with
  any of them; the page notes that only the default was measured.
- D36 (2026-11-11): the CV is read from `data/cv.md` on every request; the page shows
  only the path and has no CV input.
- D37 (2026-11-11): Phase 6 is pushed to GitHub after the pre-push check and the
  candidate's go-ahead.
- The candidate answered D34-D37 as proposed on 2026-11-11 without objecting to the
  Phase 6 plan; Claude took that as approval of the plan.

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

## Phase 2 – Test set and evaluation (done)

Source: [PROCESS.md §5 Phase 2](PROCESS.md#phase-2--test-set-and-evaluation)

### Before the live run (candidate)
- [x] Answer D5, D6, D12, D13, D14 (all decided as proposed, see Decided).
- [ ] Collect ≥ 20 real postings and label each one yourself, ≥ 5 per label. The label
      is the decision *you* would make with the hard rules included (an on-site Izmir
      role is `SKIP` even if the stack fits), because eval scores the final decision.
- [ ] Label before looking at any model output, so the model cannot anchor the labels.
      The 3 Phase 1 samples may be reused; their model outputs are already known, so
      label them from the posting alone.
- [ ] Include hard cases on purpose: borderline APPLY/PREPARE, a good stack in a blocked
      city, a junior role with a matching stack, a posting with no location.

Steps 1–7 need no dataset and no network; they can be built and tested while the
postings are being collected. Steps 8–9 need the labeled set.

### Module layout
PROCESS.md names only `eval.py`. Pure logic is split out so it can be tested without
files or an LLM ([docs/python-style.md](docs/python-style.md): logic pure, I/O at the edges):

| Module | Job | I/O |
|---|---|---|
| `dataset.py` | `Posting` model, load and validate `postings.jsonl`, `add` helper CLI | files |
| `metrics.py` | accuracy, confusion matrix, weighted error, baselines | none (pure) |
| `pricing.py` | dated list-price table, cost of a call | none (pure) |
| `eval.py` | run `decide` over the set, build the report, save the run, CLI | files, LLM |

### Steps
1. [x] `dataset.py`
       - `Posting` (Pydantic, `extra="forbid"`): `id`, `company`, `role`, `text`,
         `label: DecisionLabel`, `note: str = ""`.
       - `load_postings(path) -> list[Posting]`. Structural problems are fatal and name
         the line number: invalid JSON, missing field, unknown label, empty `text`,
         duplicate `id`. Raises `DatasetError`.
       - `check_coverage(postings) -> list[str]`: fewer than 20 postings or fewer than 5
         of a label. Fatal in eval unless `--allow-incomplete` is passed (needed for
         smoke runs before the set is complete); the shortfall is stored in the run file.
       - Helper (D13): `uv run python -m job_fit.dataset add --id <id> --company <c>
         --role <r> --label <L> --file <posting.txt> [--note <n>]` appends one escaped
         row and refuses a duplicate id. `... dataset check` prints counts per label.
       - Only `text` is ever sent to the model. `label`, `note`, `company` and `role`
         stay on the eval side, otherwise the answer leaks into the prompt.
2. [x] `metrics.py` – pure functions over `(actual, predicted)` label pairs:
       - `LABELS = ("APPLY", "PREPARE", "SKIP")`, `ERROR_COSTS` exactly as the PROCESS.md
         table (APPLY→SKIP 3, PREPARE→SKIP 2, everything else off-diagonal 1).
       - `accuracy`, `confusion_matrix` (3×3, rows actual, columns predicted),
         `weighted_error` returning total and mean per posting (D6).
       - A failed posting (D5) is passed as prediction `None`: counted wrong for
         accuracy, charged the worst cost of its actual row (APPLY 3, PREPARE 2,
         SKIP 1), left out of the 3×3 matrix and reported as a separate count.
       - `majority_label` (tie broken by `LABELS` order, so it is deterministic) and
         `constant_baseline(actual, label)`; the report shows the majority baseline and,
         per D14, the other two constant predictors.
       - The baseline is computed on the same set it is scored on. With no training
         step there is nothing to hold out; this is stated in the report.
3. [x] `pricing.py`
       - `PRICES_USD_PER_MILLION_TOKENS: dict[str, ModelPrice]` (input, output) with a
         `PRICES_CHECKED_ON` date. Starts with `gemini-2.5-flash` only; the price is
         read from the official Vertex pricing page at implementation time, not from
         memory. Thinking tokens are billed as output, which matches how `llm.py`
         already counts them.
       - `call_cost_usd(model, input_tokens, output_tokens) -> float | None`. An
         unknown model returns `None`: eval still runs and prints "price unknown"
         instead of a made-up number or a crash after the calls were paid for.
4. [x] `eval.py` – runner
       - `evaluate(postings, cv, *, model, provider, decide_fn=decide) -> EvalRun`;
         `decide_fn` is injectable so tests use a fake (no network).
       - Sequential, one progress line per posting (`[7/20] <id> label=APPLY
         predicted=SKIP 15.2s`). About 20 × 16 s ≈ 5 min on `gemini-2.5-flash`.
       - `LLMError` / `DecisionError` on one posting is caught and recorded (D5); the
         run continues. Three consecutive `LLMError`s abort the run with a clear
         message (wrong project, expired login), so a broken setup does not burn
         20 timeouts.
       - Per posting record: `id`, `label`, `predicted`, `llm_decision` (before rules),
         `fired_rules`, tokens, latency, attempts, `rationale`, `error`.
5. [x] `eval.py` – report (printed to the terminal, built by a pure `format_report`)
       - dataset: size and count per label
       - accuracy; 3×3 confusion matrix; weighted error total and mean; failed count
       - the same three numbers for the LLM's decision *before* rules, to show what
         the rules add or break
       - baselines next to the model, and one verdict line: beats / does not beat the
         majority baseline on accuracy and on mean weighted error
       - total input and output tokens, estimated cost (USD) for the run and per 100
         postings (the unit of the Phase 3 table), mean latency
       - wrong postings one by one: id, company, role, label → predicted, LLM decision,
         rules fired, candidate's `note`, rationale
6. [x] `eval.py` – save and CLI
       - `uv run python -m job_fit.eval --model <model> [--provider vertex]
         [--data data/postings.jsonl] [--cv data/cv.md] [--limit N] [--allow-incomplete]`
       - Saves `data/results/<YYYY-MM-DD>_<HHMMSS>_<model>.json` (git-ignored). The time
         is added to the PROCESS.md name so a second run on the same day does not
         overwrite the first; `/` and `:` in model ids are replaced. The clock is
         injected for tests.
       - File content: model, provider, timestamp, SHA-256 of the dataset file and of
         the system prompt (two runs are comparable only if both match), prices used,
         all metrics, baselines, per-posting records. No CV text, no posting text.
       - Exit code 0 when the run completes, 1 on a dataset or configuration error.
         Losing to the baseline is a result, not an error.
7. [x] Tests (no network, no real LLM, no real clock)
       - `test_dataset.py`: valid file loads; bad JSON, missing field, unknown label,
         empty text and duplicate id each rejected with the line number; coverage
         shortfalls reported; `add` writes a row that loads back and refuses a
         duplicate id.
       - `test_metrics.py`: hand-made inputs with known answers: perfect predictions;
         each cell of the cost matrix (parametrized); confusion-matrix orientation;
         failed posting charged the worst cost and excluded from the matrix; majority
         label and its tie-break; constant baselines; empty input rejected.
       - `test_pricing.py`: known model cost computed; unknown model returns `None`.
       - `test_eval.py`: fake `decide_fn` → expected metrics; one failing posting does
         not stop the run; three consecutive `LLMError`s abort; `--limit` respected;
         incomplete set rejected without `--allow-incomplete`; run file written with
         the expected name and keys and without the CV text; `main` exit codes.
8. [~] Dataset in place: 21 postings (APPLY 6, PREPARE 3, SKIP 12). `dataset check`
       still reports "3 postings labeled PREPARE, need at least 5"; the run below used
       `--allow-incomplete`. Accepted as is by the candidate (D15).
9. [x] Live run: `--limit 3` smoke run, then the full set (results below).
10. [x] `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`; show output.
11. [x] Update `AGENTS.md` (folder line), commit `phase-2: test set and evaluation`,
        update the status table.

### Live run results (2026-11-10, `gemini-2.5-flash`, `us-central1`)

Run file: `data/results/2026-11-10_220918_gemini-2.5-flash.json` (dataset SHA-256
`296805a932ba…`, prompt SHA-256 `a1bff7049339…`). 21 postings, no failed calls, no
schema retries.

| | Accuracy | Weighted error (total) | Mean per posting |
|---|---|---|---|
| Model, final (after rules) | 66.7% (14/21) | 10 | 0.48 |
| Model, LLM before rules | 61.9% (13/21) | 11 | 0.52 |
| Always SKIP (majority baseline) | 57.1% | 24 | 1.14 |
| Always APPLY | 28.6% | 15 | 0.71 |
| Always PREPARE | 14.3% | 18 | 0.86 |

Confusion matrix (rows actual, columns predicted):

| | APPLY | PREPARE | SKIP |
|---|---|---|---|
| **APPLY** | 3 | 2 | 1 |
| **PREPARE** | 0 | 2 | 1 |
| **SKIP** | 1 | 2 | 9 |

Tokens 47,570 in / 47,326 out; cost $0.1326 for the run, $0.63 per 100 postings (list
price); mean latency 14.8 s per posting.

### Findings to carry into Phase 3
- The model beats the majority baseline on both measures, but the margin on accuracy is
  2 postings (14 vs. 12 of 21). On weighted error the harder rival is "always APPLY"
  (0.71 vs. 0.48), about 5 cost points over 21 postings. Treat both as weak evidence.
- Errors lean cautious: 4 of 7 are a good or preparable job rated one step too low
  (APPLY→PREPARE twice, APPLY→SKIP for Miny, PREPARE→SKIP for Getir). Miny is the one
  cost-3 error, the same as in Phase 1: a single missing skill (Go) outweighs a
  matching main stack.
- The opposite error also occurs: 3 SKIP postings were rated APPLY or PREPARE where the
  candidate sees a hard requirement (Go backend 7+ years, accounting background, Flutter
  architect depth). The model treats these as closable gaps.
- Rules fired on 3 postings and changed 1 outcome: Kaslıfilozof (Bursa, on-site) went
  from the LLM's PREPARE to SKIP, which is where the 61.9% → 66.7% comes from. The
  junior and intern postings were already SKIP before the rule.
- Output tokens equal input tokens (about 2,250 per posting) and are 89% of the cost;
  thinking is the lever to test in Phase 3 (reasoning off vs. on).
- No prompt changes were made after seeing these results; this run is the reference.

### Caveats on the test set
- The labels came from a suggested list (`data/postings/_labels.txt`) that the candidate
  accepted as a whole on 2026-11-10 rather than writing each one independently.
- Posting texts under `data/postings/` are condensed (600–3,500 characters), shorter than
  the postings as published.
- PREPARE has 3 examples, so the PREPARE row of the matrix says very little.
- A failed posting's tokens are not counted (`decide` raises without returning usage);
  it did not matter in this run because nothing failed.

**Done when:** `uv run python -m job_fit.eval --model <model>` prints all metrics in one
command, including the baseline, on the candidate's ≥ 20 labeled postings.

**Not in this phase:** prompt tuning (the first full run is the reference point; changing
the prompt against these 20 postings afterwards would be fitting to the test set),
other models and providers, reasoning on/off, parallel calls, agent, `docs/results.md`
tables (Phase 3 fills them from the run files).

**Risks**
- 20 postings is a small sample: one posting is 5 points of accuracy. Differences of
  one or two postings between runs are noise, not a finding.
- `temperature=0` does not make Gemini fully deterministic, least of all with thinking
  on; two runs can differ. The run file keeps every prediction so differences can be
  inspected.
- Labels are one person's judgment. A wrong prediction may be a wrong label; the
  per-posting error list with the `note` is there to review both sides.
- Cost of the full run: about 20 × (2.5k input + 2.5k output) tokens on a Flash-tier
  model, a few cents. The exact figure is printed by the run itself.
- No new dependency is expected; the standard library covers JSONL, hashing and tables.

---

## Phase 3 – Model selection and cost (done)

Source: [PROCESS.md §5 Phase 3](PROCESS.md#phase-3--model-selection-and-cost)

State at the start (checked 2026-11-10): `.env` has `GOOGLE_CLOUD_LOCATION=global` and a
non-empty `OPENROUTER_API_KEY`; Ollama is not installed; `decide` and the eval CLI do not
pass `reasoning` through yet; `llm.py` rejects `openrouter` and `ollama`.

### Before the live runs (candidate)
- [x] D7, D17, D18, D19 decided (see Decided).
- [ ] Set a budget alert in the Google Cloud console (Billing → Budgets & alerts), e.g.
      5 USD with alerts at 50% / 90% / 100%. Expected spend for the whole phase is under
      2 USD at list price (estimate in Risks). An alert only notifies; it does not stop
      spending.

Steps 1–6 need no network and can be built before any of the above is finished.

### Rows of the comparison table

Model ids below are candidates. Step 7 lists what the project can actually call and
reads the list prices from the official pages; nothing is taken from memory.

| Row | Candidate | Provider | Reasoning | Why this row |
|---|---|---|---|---|
| Baseline | always SKIP | – | – | already computed by eval |
| A – strong | newest Gemini Pro available in `global` | vertex | model default (Pro cannot switch thinking off) | upper bound on quality |
| B – mid | `gemini-2.5-flash` | vertex | **off** | the Phase 2 model without thinking |
| C – cheap/fast | Gemini Flash-Lite tier | vertex | off | lower bound on cost and latency |
| X + reasoning | `gemini-2.5-flash` | vertex | **on** | same model as B, only the switch differs |
| O – open-weight, free (D17) | one `:free` open-weight model, chosen in step 7 | openrouter, pinned provider | default | stands in for the open-weight part of the phase; exercises provider pinning |
| O2 (if the daily quota allows) | a second `:free` model of a different size | openrouter, pinned provider | default | shows how much size matters among open models |
| Local | not run (D7) | – | – | row kept in the table with the memory estimate and the reason |

Rows B and X are the reasoning experiment: `gemini-2.5-flash` accepts a thinking budget
of 0 (off) and -1 (dynamic), so one model gives both rows and thinking is the only
variable. Row X is the Phase 2 reference configuration, now run in `global` (D11).

### Models fixed in steps 7–8 (2026-11-10)

Listed with `client.models.list()` in `global` and probed with a one-word request.
`gemini-2.5-pro` and `gemini-2.5-flash-lite` are listed but return 404 for this project.

| Row | Model id | Provider | Reasoning | List price in / out per 1M tokens |
|---|---|---|---|---|
| A – strong | `gemini-4-carbon` | vertex | default (cannot be switched off) | 2.00 / 10.00 through 2026-11-30, then 4.00 / 20.00 |
| B – mid | `gemini-2.5-flash` | vertex | off | 0.30 / 2.50 |
| C – cheap/fast | `gemini-3.1-flash-lite` | vertex | default (does not think by default) | 0.25 / 1.50 |
| X + reasoning | `gemini-2.5-flash` | vertex | on | 0.30 / 2.50 |
| O – open-weight | `nvidia/nemotron-3.5-super-vl:free` (120B total, 12B active) | openrouter, pinned to `nvidia` | default | 0 |
| O2 – small open-weight | `liquid/lfm-3-1.7b:free` (1.7B) | openrouter, pinned to `liquid` | default | 0 |

- Price sources: all three Gemini prices are confirmed on the Vertex AI pricing page
  (Global region, standard tier), see D22.
- D8: not tested. No Claude model is used in this phase (D17), so the question no
  longer affects anything; it is dropped.
- O2 was chosen as the size that would actually fit this machine (3 GB VRAM), which
  links the free row to the local-model question.
- OpenRouter account: free tier (`is_free_tier = true`), so free models are limited per
  day; the two full runs plus two one-posting smoke runs are 44 requests.
- Free models were filtered on `structured_outputs` in OpenRouter's public model list;
  `google/gemma-4-*:free` supports only plain JSON mode and was left out.

### Steps
1. [x] `uv add openai`. One SDK serves both new providers: OpenRouter and Ollama expose
       OpenAI-compatible endpoints, so they differ only in base URL, key and extra body.
2. [x] `llm.py` – OpenAI-compatible providers
       - `_generate_openai_compatible(client, model, system, user, schema, reasoning,
         timeout_s, extra_body)`: `chat.completions.create` with `temperature=0` and
         `response_format={"type": "json_schema", ...}` built from
         `schema.model_json_schema()`. Tokens from `usage` (`prompt_tokens`,
         `completion_tokens`; reasoning tokens are already inside `completion_tokens`).
         SDK errors are wrapped in `LLMError`, same as Vertex.
       - `openrouter`: base URL `https://openrouter.ai/api/v1`, key from
         `OPENROUTER_API_KEY`. The extra body pins the route:
         `provider = {"order": [<provider>], "allow_fallbacks": false,
         "require_parameters": true}`. The pinned provider per model lives in one named
         constant table. `reasoning` maps to OpenRouter's `reasoning` field.
       - `ollama`: base URL from `OLLAMA_BASE_URL`, dummy key. `reasoning=True` is
         rejected with a clear error unless the chosen model supports it.
       - `LLMResponse`, the `generate_json` signature and the "returns raw text, does
         not validate" contract stay as they are. Remove `PROVIDERS_PLANNED_FOR_PHASE_3`.
3. [x] Pass reasoning and timeout through
       - `workflow.decide(..., reasoning: bool | None = None, timeout_s=...)` forwards
         both to the `llm` callable.
       - Eval and workflow CLIs: `--reasoning on|off|default` (default `default`) and
         `--timeout-s` (a local model on CPU can need minutes per posting).
       - The run file records `reasoning` and the Vertex location; the file name gains a
         suffix (`..._gemini-2.5-flash_reasoning-off.json`) so on and off do not look
         like the same run.
4. [x] `pricing.py`
       - Add the chosen models with prices read in step 7; update `PRICES_CHECKED_ON`.
       - Key the table by `(provider, model)`: the same model can cost differently on
         Vertex and on OpenRouter, and `ollama` models get an explicit price of 0 so the
         report prints `$0.00` instead of "price unknown".
       - Docstring note: tiered prices above 200k prompt tokens do not apply (prompts
         here are about 2.3k tokens).
5. [x] `compare.py` (new: pure logic plus a thin CLI)
       - `load_runs(paths)` reads run files and refuses to put runs with different
         dataset or prompt hashes into one table (they are not comparable, D15).
       - `pareto_front(rows)`: a row is on the front if no other row is at least as good
         on mean weighted error, cost per 100 postings and mean latency, and strictly
         better on one of them. Pure function, unit-tested with hand-made rows.
       - `format_markdown_table(rows)` prints the PROCESS.md table with the baseline row
         and a Pareto mark in Notes.
       - `uv run python -m job_fit.compare data/results/<file> ...`, so the table in
         `docs/results.md` is generated from the run files instead of copied by hand.
6. [x] Tests (no network, fake clients)
       - `test_llm.py`: OpenAI-compatible response mapped to `LLMResponse`; the
         OpenRouter request carries the pinned provider and `allow_fallbacks=false`;
         Ollama uses `OLLAMA_BASE_URL`; missing `OPENROUTER_API_KEY` → `LLMError`; SDK
         error → `LLMError`; reasoning flag translated per provider.
       - `test_workflow.py` / `test_eval.py`: `reasoning` and `timeout_s` reach the fake
         `llm`; file name suffix; `--reasoning` parsing.
       - `test_pricing.py`: `(provider, model)` lookup; Ollama costs 0; unknown → `None`.
       - `test_compare.py`: Pareto front on known rows (dominated row dropped, ties
         kept); mismatched hashes rejected; table text.
7. [x] Discover models and prices (live, read-only): list the Gemini models reachable in
       `global`, fix rows A and C, check D8 with one Claude call on Vertex, read list
       prices from the official pricing pages. Write the final ids into this plan.
8. [x] Free OpenRouter model(s) and the local estimate
       - Pick row O from OpenRouter's public model list: price 0, open weights,
         `structured_outputs` among the supported parameters, context of at least 8k.
         Pin the provider that serves the free endpoint. Write the id, parameter count
         and provider into this plan.
       - Check the account's free-tier limits first (`GET /api/v1/key`). Free models
         are rate-limited per minute and per day; a full run is 21 calls plus retries,
         so the smoke run and the full run of one model may be all one day allows.
         Eval gains `--delay-s` (pause between postings) to stay under the per-minute
         limit; the pause is not counted in latency.
       - Local, on paper only (D7): `parameters (B) × bits ÷ 8 ≈ GB`, e.g.
         8B × 4 ÷ 8 = 4 GB; "4-bit" GGUF files (Q4_K_M) average closer to 4.5–5 bits,
         and context (KV cache) and runtime overhead come on top. Compared with the
         measured 3 GB VRAM / 7.8 GB RAM, with the conclusion that the model was not run.
9. [x] Smoke run per row with `--limit 2`, to catch a wrong id, missing schema support
       or an auth problem before spending a full run.
10. [~] Full runs, sequential, one model at a time (parallel runs would distort
        latency): rows A, B, C, X done; O and O2 stopped by the quota (D20). One repeat of
        row B to see run-to-run noise on this set.
11. [x] `docs/results.md`: generated table, the local memory estimate and why it was
        not run, the noise figure from the repeat, and a drafted "Which model I chose
        and why" paragraph (Pareto front; the privacy pick is argued from the
        open-weight row and the memory estimate). The candidate edits or approves it.
12. [x] `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`; show output.
13. [x] Update `AGENTS.md` (folder line: `compare`), record results and findings here,
        commit `phase-3: model selection and cost`, update the status table.

### Live run results (2026-11-10, Vertex AI `global`)

All runs: 21 postings, dataset SHA-256 `296805a932ba…`, prompt SHA-256 `a1bff7049339…`,
no failed calls, no schema retries. Run files are in `data/results/` (git-ignored).

| Run file | Accuracy | Weighted error total / mean | Before rules | Tokens in / out | Cost per 100 | Mean latency |
|---|---|---|---|---|---|---|
| `2026-11-10_224958_gemini-4-carbon.json` | 66.7% (14) | 13 / 0.62 | 57.1%, 0.71 | 56,615 / 25,388 | $1.75 | 57.3 s |
| `2026-11-10_231005_gemini-3.1-flash-lite.json` | 52.4% (11) | 10 / 0.48 | 42.9%, 0.57 | 53,954 / 5,700 | $0.10 | 1.7 s |
| `2026-11-10_231043_gemini-2.5-flash_reasoning-off.json` | 52.4% (11) | 10 / 0.48 | 42.9%, 0.57 | 47,570 / 7,841 | $0.16 | 2.6 s |
| `2026-11-10_231140_gemini-2.5-flash_reasoning-on.json` | 71.4% (15) | 9 / 0.43 | 71.4%, 0.43 | 47,570 / 48,192 | $0.64 | 11.9 s |
| `2026-11-10_231552_gemini-2.5-flash_reasoning-off.json` (repeat) | 47.6% (10) | 11 / 0.52 | 33.3%, 0.67 | 47,570 / 7,768 | $0.16 | 2.6 s |

Baseline (always SKIP): 57.1%, mean 1.14. Total list-price spend of the five runs: about
$0.59, plus a few cents for smoke runs and model probes.

### Findings to carry into Phase 4
- Chosen model for the workflow-vs-agent comparison: `gemini-2.5-flash` with reasoning
  on (D21). An agent makes several model calls per posting, so its cost and
  latency start from $0.64 per 100 postings and 11.9 s and go up.
- Noise: the repeated reasoning-off run changed 3 of 21 predictions, 1 posting of
  accuracy and 1 point of weighted error. Phase 4 needs a difference larger than that
  before calling the agent better or worse; consider running each version twice.
- Thinking is what moved accuracy (52.4% → 71.4% on the same model). Without it both
  cheap configurations fall below the baseline on accuracy while staying far ahead on
  weighted error, because they over-predict PREPARE.
- `gemini-4-carbon` is dominated: over-strict (4 good or preparable jobs rated SKIP),
  2.7 times the cost and 4.8 times the latency of the chosen configuration. One call
  took 197 s, so the default 60 s timeout is too short for it.
- Miny, A1 Mobile and Rubikpara (label APPLY) are rated lower by every model. Review the
  labels or the prompt's definition of APPLY; do not tune the prompt on these alone.
- The hard rules matter most when the model is weak: about +2 postings with reasoning
  off, no change with reasoning on.
- OpenRouter free tier: the daily free-model quota ran out after 10 requests from this
  project on that day. Free endpoints are not usable for a 21-posting eval without
  buying credits.
- `gemini-2.5-pro` and `gemini-2.5-flash-lite` are listed for the project but return
  404. A pricing page read during this phase suggested that older 2.5 models are due to
  be rerouted or retired; this is unverified. If `gemini-2.5-flash` stops answering,
  the chosen configuration must be re-measured on its successor.
- Shell note for the agent: heredocs containing apostrophes fail in this environment's
  Bash tool; write scripts with the file tool instead.

**Done when:** the table in `docs/results.md` is filled in from run files of one dataset
version and the choice is justified in writing.

**Not in this phase:** prompt tuning (the prompt hash must stay the same across all
rows), per-model prompts, parallel or batch calls, agent, UI, a fourth provider SDK
(`anthropic[vertex]`).

**Risks**
- Small sample: one posting is about 5 points of accuracy. Models that differ by one or
  two postings are tied; the repeat run in step 10 shows how large the noise is. The
  choice may have to rest on cost and latency rather than accuracy.
- No local run (D7): the local row of PROCESS.md stays empty and interview question 7
  is answered from the estimate, not from a measurement.
- **Personal data on free endpoints (D19):** every call sends `data/cv.md`. Providers
  of free OpenRouter models may log prompts and use them for training, and OpenRouter
  only routes to them if the account's privacy settings allow it. Vertex does not
  train on the prompts.
- Free endpoints are slow and queue behind paying traffic, so their latency is not
  comparable with the Vertex rows; this is stated in the table notes. Their list price
  is 0; the price of the paid route for the same model is given in Notes.
- A free model can be withdrawn or rate-limited mid-run. Three consecutive `LLMError`s
  already abort a run; an aborted run is repeated, not patched together.
- A small open model may fail the schema or mishandle Turkish postings. Failed
  postings are already scored as wrong (D5); a high failure count is a result to
  report, not something to hide by retrying more.
- Structured output differs by provider (Gemini response schema, OpenAI `json_schema`,
  Ollama grammar). `require_parameters` makes OpenRouter fail loudly instead of
  silently ignoring the schema.
- Latency is measured from one home connection, once per posting; treat it as a rough
  figure.
- Spend (list price, rough): Flash with thinking ≈ 0.13 USD per run (measured in
  Phase 2), Flash without thinking and Flash-Lite a few cents each, Pro ≈ 0.5–1 USD;
  under 2 USD in total on Vertex credits. The OpenRouter rows use free models only
  and cost nothing (D17).
- Preview model ids can be renamed or removed; the run file keeps the exact id used.

---

## Phase 4 – Agent version (done)

Source: [PROCESS.md §5 Phase 4](PROCESS.md#phase-4--agent-version)

State at the start (checked 2026-11-11): no `agent.py` or `tools.py`; `langchain` is not
a dependency; `eval.py` and `compare.py` know only the workflow; `compare.load_runs`
refuses runs with different prompt hashes, which a workflow run and an agent run always
have; `data/cv.md` has the headings `Summary`, `Experience`, `Projects`, `Skills`,
`Education`, `Languages`.

Fixed for both versions (D21): `gemini-2.5-flash` on Vertex AI `global`, reasoning on,
temperature 0, the same 21 postings (dataset SHA-256 `296805a932ba…`), the hard rules
applied in code to the final output. Only the harness differs.

### Before the live runs (candidate)
- [x] Answer D23–D26 (all decided as proposed, see Decided).
- [ ] Budget alert in the Google Cloud console, still open from Phase 3. An agent makes
      several model calls per posting, so this phase spends more per run than Phase 3.

Steps 1–8 need no decision on D24 and, except for the spike in step 1, no network.

### Design

| Question | Answer | Why |
|---|---|---|
| What is a "step"? | One model call. `MAX_AGENT_STEPS = 6`. | The model call is what costs money and time; tool calls here are local and free. |
| How is the limit enforced? | LangChain's model-call-limit middleware if the installed version has it, else LangGraph's `recursion_limit`. Chosen in step 1. | Not assumed from memory; the API changed between LangChain versions. |
| Who has the last word? | `rules.apply_rules` on the agent's final `JobDecision`, exactly as in the workflow. | `check_rules` as a tool is advice the model may skip or misread; the hard rules must not depend on the model choosing to call a tool. |
| How do tools get the CV? | `build_tools(cv)` returns the tools as closures over the CV text. | No module-level mutable state; tests build tools from a small fake CV. |
| Where is the chat model created? | `llm.chat_model(provider, model, reasoning, timeout_s)`. | `llm.py` stays the only module that imports a provider SDK or reads `.env`. Only `vertex` is supported for the agent; other providers raise a clear error. |
| How does the agent plug into eval? | `agent.decide_with_agent` has the same keyword signature as `workflow.decide` and returns the same `DecisionResult`. | Eval already takes an injectable `decide_fn`; nothing else in the runner changes. |
| What is logged? | Tool name and arguments, through `logging` and into the run file. Tool results are not stored. | Arguments are posting facts and a topic name; results of `get_cv_section` are CV text, which run files never contain. |

### Steps
1. [x] Dependencies and a spike. `uv add langchain` plus the Gemini chat-model package
       that can reach Vertex AI with the existing application-default login (candidate:
       `langchain-google-genai`; fall back to `langchain-google-vertexai`). Then one
       throw-away script in the scratchpad, one posting, to settle four things before
       any design is frozen:
       - `create_agent` accepts tools and `response_format=JobDecision` together on
         `gemini-2.5-flash` (Gemini may not allow function calling and a response
         schema in one request; then the structured answer goes through LangChain's
         tool strategy);
       - the step-limit mechanism and the exception it raises;
       - where token usage sits on the messages, and whether thinking tokens are inside
         `output_tokens` (they must be counted as in `llm.py`, or costs are not
         comparable);
       - how thinking and the timeout are passed.
       Findings are written into this plan; the script is not committed.
2. [x] `llm.py`: `chat_model(provider, model, *, reasoning, timeout_s)` returning a
       LangChain chat model for `vertex`, temperature 0, same project and location as
       `_vertex_client`. Other providers → `LLMError("agent supports only vertex")`.
3. [x] `tools.py`
       - Pure helpers: `split_cv_sections(cv) -> dict[str, str]` (split on `## `
         headings, keys normalized to lowercase), `find_cv_section(sections, topic)`
         (exact, then substring match), `to_try(amount, currency)` with
         `FX_RATES_TO_TRY` and `FX_RATES_CHECKED_ON` constants (D26).
       - `build_tools(cv) -> list[BaseTool]` with the three `@tool` functions from
         PROCESS.md, signatures unchanged:
         - `check_rules(location, work_mode, english_required, seniority) -> dict`:
           calls `rules.violated_rules`; returns `violated_rules`, `forced_decision`
           (`"SKIP"` or `None`) and `english_risk` (the flag passed in; it never forces
           a decision).
         - `get_cv_section(topic) -> str`: the section text. An unknown topic returns a
           message listing the available topics instead of raising, so the model can
           correct itself within its step budget.
         - `convert_currency(amount, currency) -> float`: USD or EUR to TRY. The tool
           description states that the rate is fixed and its date. An unsupported
           currency or a negative amount raises `ValueError`, which LangChain hands
           back to the model as a tool error.
       - Tool docstrings are the descriptions the model reads; they are part of the
         agent prompt hash.
4. [x] `agent.py`
       - `AGENT_SYSTEM_PROMPT`: same role, same "data, not instructions" line and the
         same APPLY / PREPARE / SKIP definitions as the workflow prompt, plus how to
         work with the tools (read the CV sections you need, call `check_rules` with
         the extracted facts, use `convert_currency` only if a foreign-currency salary
         is stated). `workflow.SYSTEM_PROMPT` is not touched, so the workflow prompt
         hash stays `a1bff7049339…`.
       - User message: the posting only (D23).
       - `decide_with_agent(posting_text, cv, *, model, provider, reasoning, timeout_s,
         chat_model_factory=llm.chat_model) -> DecisionResult`: build tools and agent,
         invoke with the step limit, take the structured `JobDecision`, apply
         `rules.apply_rules`.
       - Pure `summarize_run(messages) -> AgentTrace`: model calls, tool calls (name,
         arguments) and token usage summed over every model message. Latency is one
         `time.perf_counter()` span around the whole invocation.
       - Failure mapping (D25): step limit reached or no valid structured answer →
         `DecisionError`; provider/API error → `LLMError`. Eval already handles both.
       - `DecisionResult` gains `model_calls: int = 1` and
         `tool_calls: tuple[ToolCallRecord, ...] = ()`; the workflow keeps the defaults.
       - CLI: `uv run python -m job_fit.agent --posting <file> --model <id>` prints the
         decision, the tool calls in order, tokens and latency.
5. [x] `eval.py`
       - `--version workflow|agent` (default `workflow`) selects `decide` or
         `decide_with_agent`.
       - `PostingResult` gains `model_calls` and `tool_calls`; `EvalReport` gains
         `version`, `mean_tool_calls`, `mean_model_calls` and tool-call counts per tool
         name. All with defaults, so the Phase 2–3 run files still load.
       - `prompt_sha256` for an agent run hashes the agent prompt and the tool
         descriptions.
       - Report: one extra block for agent runs (calls per posting, how many postings
         called each tool, how often `check_rules` agreed with the rules applied in
         code). File name gains `_agent`.
6. [x] `compare.py`
       - Comparability: dataset hash and size must match across all runs; the prompt
         hash must match only among runs of the same version.
       - `--table versions` prints the Phase 4 table (Version, Accuracy, Weighted
         error, Cost per 100, Latency, Avg. tool calls), one row per run, repeated runs
         numbered. The existing model table stays the default.
7. [x] Tests (no network, no real LLM)
       - `test_tools.py`: `check_rules` on-site Bursa → `location`, junior →
         `seniority`, Istanbul senior → nothing, English flag passed through without a
         forced decision; `split_cv_sections` and `get_cv_section` (known topic, case
         and partial match, unknown topic lists the available ones, CV without
         headings); `convert_currency` USD, EUR, unsupported currency, negative amount.
       - `test_agent.py`: `summarize_run` on hand-made messages (tokens summed, tool
         calls in order); a scripted fake chat model drives the real `create_agent`
         through tool call → final answer; rules override an agent `APPLY`; step limit
         → `DecisionError`; missing structured answer → `DecisionError`; posting-only
         user message contains no CV text.
       - `test_llm.py`: `chat_model` rejects non-Vertex providers.
       - `test_eval.py` / `test_compare.py`: `--version` routing, new fields and
         defaults, old run file still loads, `_agent` file name, mixed-version
         comparison accepted, mismatched dataset rejected, version table text.
8. [x] `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`; show output.
9. [x] Live runs, sequential: agent smoke run `--limit 2`, then the full runs per D24.
       `--timeout-s` raised for the agent if the smoke run shows it is needed.
10. [x] `docs/results.md`: generated workflow-vs-agent table, tool usage, what the agent
        got right or wrong that the workflow did not, and a drafted "Is an agent needed
        for this problem?" paragraph. The candidate edits or approves it.
11. [x] Record results and findings here, commit `phase-4: agent version`, update the
        status table.

### Spike findings (step 1, 2026-11-11)

Installed: `langchain` 1.4.10, `langgraph` 1.2.20, `langchain-google-genai` 4.4.3 (it
reaches Vertex AI with `vertexai=True` and the existing application-default login).

- Tools and `response_format` work together through `ToolStrategy(JobDecision)`: the
  structured answer arrives as a call to a tool named `JobDecision`. It is excluded from
  the tool-call count. An invalid answer is sent back to the model and retried, which
  uses up a step.
- Step limit: `ModelCallLimitMiddleware(run_limit=6, exit_behavior="error")` raises
  `ModelCallLimitExceededError`; it is mapped to `DecisionError` (D25).
- Token usage is on each `AIMessage.usage_metadata`; `output_tokens` already includes
  thinking tokens, the same accounting as `llm.py`, so costs are comparable.
- Thinking and timeout are constructor arguments (`thinking_budget`, `timeout`).

### Deviations from the plan above

- No new `model_calls` field: `DecisionResult.attempts` already means "number of LLM
  calls" and holds the agent's step count. Only `tool_calls` was added.
- The version table has one column more than PROCESS.md asks for (Avg. model calls).
- FX rate (D26): the search for a November 2026 rate gave no clean source; the constants
  are approximate mid-market rates from xe.com dated 2026-11-04 (1 USD = 50.0 TRY,
  1 EUR = 56.1 TRY) and are labeled approximate in the code and in `docs/results.md`.
- `llm.py` now imports `httpx` (a transitive dependency of `google-genai` and `openai`)
  to name the timeout and connection errors a chat model can raise.

### Live run results (Vertex AI `global`, `gemini-2.5-flash`, reasoning on)

All runs: 21 postings, dataset SHA-256 `296805a932ba…`, no failed calls, no posting at
the step limit, no schema retries. Workflow prompt SHA-256 `a1bff7049339…` (unchanged),
agent prompt SHA-256 `95ff50b9f5ce…`. Run files are in `data/results/` (git-ignored).

| Run file | Accuracy | Weighted error total / mean | Before rules | Tokens in / out | Cost per 100 | Mean latency | Model / tool calls |
|---|---|---|---|---|---|---|---|
| `2026-11-10_231140_gemini-2.5-flash_reasoning-on.json` (workflow, Phase 3) | 71.4% (15) | 9 / 0.43 | 71.4%, 0.43 | 47,570 / 48,192 | $0.64 | 11.9 s | 1.00 / 0 |
| `2026-11-11_104916_gemini-2.5-flash_reasoning-on.json` (workflow) | 66.7% (14) | 10 / 0.48 | 61.9%, 0.52 | 47,570 / 48,104 | $0.64 | 12.1 s | 1.00 / 0 |
| `2026-11-11_103839_gemini-2.5-flash_agent_reasoning-on.json` | 85.7% (18) | 3 / 0.14 | 81.0%, 0.19 | 157,754 / 43,409 | $0.74 | 14.8 s | 3.67 / 3.71 |
| `2026-11-11_104355_gemini-2.5-flash_agent_reasoning-on.json` | 61.9% (13) | 8 / 0.38 | 61.9%, 0.38 | 160,816 / 45,052 | $0.77 | 15.0 s | 3.71 / 3.71 |

List-price spend of the three new runs: about $0.45, plus about $0.03 for the spike and
the smoke run.

### Findings to carry into Phase 5

- Conclusion (confirmed, D27): an agent is not needed. The
  agent made the same tool calls in the same order on every posting, cost 18% more, took
  24% longer and was far less stable (its two runs disagree on 8 of 21 postings, the
  workflow's on 2). Accuracy does not separate the versions.
- The agent never rated a good or preparable job as SKIP; the workflow did so twice per
  run. Its prompt differs from the workflow's and it reads only part of the CV, so this
  cannot be credited to tool calling. Worth testing as a prompt change in the workflow,
  on a larger test set, not by tuning against these 21 postings.
- The agent never read the `projects` section of the CV, although the prompt names it as
  available. `convert_currency` was called for all four postings with a USD salary, but
  `JobDecision` has no field that could use the result.
- `check_rules` agreed with the rules applied in code in 42 of 42 decisions and was
  called before the CV was read in 41 of them.
- Lessons for `AGENTS.md` (Phase 5): heredocs with apostrophes fail in the Bash tool, so
  scripts go through the file tool; there is no bare `python` on this machine, only
  `uv run python`; `load_dotenv()` looks for `.env` next to the calling script, not in
  the working directory; one test expectation was written from a wrong hand calculation
  of the weighted error and was caught only by running the test.
- `ruff` clean, 265 tests pass (195 before this phase).

**Done when:** both versions pass eval on the same dataset version, and the comparison
table and the conclusion are in `docs/results.md`.

**Not in this phase:** prompt tuning of either version after seeing results, agents on
other models or providers, a live FX API, tools beyond the three in PROCESS.md, memory
or multi-turn conversation, parallel runs, the Phase 5 hook and README.

**Risks**
- The expected result is that the agent is not better: the workflow already sees the
  whole posting and CV in one call, and the rules run in code either way. That is a
  valid outcome ("a fixed workflow can beat an agent"), not a failure of the phase.
- Small sample: one posting is about 5 points of accuracy and run-to-run noise is about
  one posting. Cost, latency and tool-call counts will be the solid part of the
  comparison; accuracy differences of one or two postings are a tie.
- `convert_currency` will probably never be called: no field of `JobDecision` holds a
  salary. Its call count is reported as is; an unused tool is itself a finding.
- With D23 the agent can decide after reading only part of the CV, or spend steps
  reading every section one by one. Both show up in the tool-call log.
- LangChain's token accounting for Gemini thinking tokens may differ from `llm.py`'s.
  Step 1 checks it; if it cannot be made equal, the difference is stated next to the
  table.
- Spend (list price, rough): workflow run about 0.13 USD (measured); agent run
  estimated at 2–4 times that, since the posting and earlier tool results are re-sent
  on every model call. Three runs plus smoke runs: under 1.50 USD.
- `gemini-2.5-flash` may be retired or rerouted (Phase 3 finding). If it stops
  answering, both versions are re-measured on its successor.
- New dependency weight: `langchain` pulls in `langgraph` and `langchain-core`. They
  are imported only by `agent.py`, `tools.py` and `llm.chat_model`.

---

## Phase 5 – Harness, harden, humans (done)

Source: [PROCESS.md §5 Phase 5](PROCESS.md#phase-5--harness-harden-humans)

State at the start (checked 2026-11-11): `ruff check` and `ruff format --check` clean;
265 tests pass in 4.6 s (7.6 s wall time with `uv run`); no `.claude/` directory, so no
hook; `README.md` is a 19-line stub; no git remote; the `gh` CLI is not installed;
`.env`, `data/cv.md` and `data/results/` were never committed (checked with `git log
--all`); the source has no code that sends, posts or opens anything besides the LLM
calls; all 8 commits carry a corporate e-mail address as author (D29).

Most of the tests PROCESS.md asks for were written with the code in Phases 1–4. This
phase therefore audits them instead of rewriting them, and spends its effort on the
parts that do not exist yet: the hook, a code review, the lessons, the README, GitHub.

### Before the push (candidate)
- [x] Answer D28–D31.
- [x] Repository created by Claude at the candidate's request (D32).
- [x] Approve the push after reading the pre-push check of step 9.

Steps 1–8 need no decision except D30 (step 3) and no network.

### Steps
1. [x] Test audit. Write a table "requirement → existing test(s)" for the three
       PROCESS.md bullets and add a test only where a cell is empty:
       - `rules.py`: positive and negative per rule: location on-site/hybrid (D2),
         location remote (D3), seniority, English flag; and at the `apply_rules`
         level, an override by each rule alone. Known today: the override is tested
         for location and for two rules together, not for seniority alone.
       - `schema.py`: malformed JSON rejected (exists); also JSON of the wrong shape
         (a list, `null`) if not covered.
       - `tools.py`: every tool has at least one test (exists for all three).
       - Mocking: confirm no test can reach the network. Add an autouse fixture in
         `tests/conftest.py` that makes `llm._vertex_client` and the OpenAI client
         constructor raise, so a future test that forgets to mock fails loudly
         instead of spending credits.
2. [x] Human-control guard. One test that the agent's tool set is exactly the three
       PROCESS.md tools (exists) and one that the package's runtime dependencies in
       `pyproject.toml` match an allow-list, so adding e.g. a browser-automation or
       mail library requires a deliberate edit of the test. The guarantee itself is
       stated in the README (step 7) and in `AGENTS.md` (already there).
3. [x] Hook: `.claude/settings.json` (committed, so it works for anyone who clones)
       with a `PostToolUse` hook on `Edit|Write`.
       - It runs `.claude/hooks/post_edit_check.py` through `uv run python`: read the
         hook JSON from stdin, skip unless the edited file is `.py` or
         `pyproject.toml` (D30), run `uv run ruff check .` and `uv run pytest -q`.
       - On failure it exits with code 2 and prints the tool output to stderr; that
         is what Claude Code feeds back to the agent. On success it is silent.
       - `ruff format` is **not** run by the hook: a formatter that rewrites a file
         right after the agent edited it makes the agent's next edit fail on stale
         text. Format stays a manual step before commit.
       - The exact hook JSON shape and which shell runs it on Windows are read from
         the Claude Code documentation at implementation time, not from memory.
       - The script is pure standard library, has docstrings, and its file filter is
         a small pure function with its own unit test.
4. [x] Prove the hook works, and show the output: (a) edit a `.py` file so that ruff
       fails → the hook reports it; (b) break a test expectation → the hook reports
       it; (c) edit a Markdown file → nothing runs; then revert both. A hook added in
       a running session may need the session restarted before it fires; if so, this
       step is done right after a restart.
5. [x] Code review, "even when tests pass". Read `src/job_fit/` module by module with
       fresh eyes for: wrong behavior at the edges (substring matching of cities and
       regions in `rules.py`, token accounting, the failure paths of D5 and D25),
       personal data in logs or run files, secrets, dead code. Findings are listed
       here with a verdict each. A confirmed bug gets a failing test first, then the
       fix. Anything that would change model-facing text (prompts, schema or tool
       descriptions) is **reported, not changed**: it would change the prompt hash
       and make the Phase 3–4 numbers stale.
6. [x] `AGENTS.md` – fill "Lessons learned" from real events in Phases 1–4, each as
       "what went wrong → the rule that prevents it". Sources: the findings sections
       of this file. Candidates: heredocs with apostrophes fail in the Bash tool;
       only `uv run python` exists; `load_dotenv()` searches from the calling file;
       an expected value computed by hand was wrong, so compute it from the cost
       table; model ids and prices are looked up, never recalled; an aborted eval
       run is repeated, never patched; no prompt change after seeing results; one
       run is not a result on 21 postings. The file must stay short (PROCESS.md: 40
       lines; it is 37 now), so the list is at most 8 one-line items and other
       sections are tightened if needed. Add the hook to the Commands section.
7. [x] `README.md`, written for someone who has never seen the project:
       what it does and the three decisions; the **human-control statement** (it only
       recommends; it never applies, sends messages or fills in forms); how it works
       (workflow, rules in code, why rationale comes first); install and
       configuration; how to run one posting, the eval and the comparison; the Phase
       3 and Phase 4 tables copied from `docs/results.md` with a link to the full
       text; the chosen model and why, and "no agent" and why, in a few sentences
       each; limits (21 postings, 3 PREPARE, no local run); project layout; how it
       was built with Claude Code (AGENTS.md, plan per phase, hook, review). No new
       numbers: every figure comes from `docs/results.md`.
8. [x] `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`; show
       output. Commit `phase-5: harness, tests, hook, README`.
9. [x] GitHub.
       - Pre-push check, output shown to the candidate: `git ls-files` reviewed for
         `.env`, key files, `data/cv.md`, `data/results/`; a search of the whole
         history for key-like strings (`AIza`, `sk-or-`, `private_key`, the project
         id from `.env` compared by presence only, never printed); author addresses.
       - Apply D29 (author rewrite if chosen) and D31 (branch name).
       - `git remote add origin <url>`, then `git push -u origin <branch>` **only
         after the candidate's explicit go-ahead**. Pushing publishes; it is the one
         step of this project that cannot be taken back.
       - Verify with `git ls-remote origin`.
10. [x] Update the status table and this section; commit and push the final status.

### Results of steps 1–7 (2026-11-11)

`ruff check` and `ruff format --check` clean; 322 tests pass (265 at the start).

**Test audit (step 1)**

| Requirement (PROCESS.md) | Covered by | Added in this phase |
|---|---|---|
| `rules.py`, location on-site/hybrid: positive, negative | `test_location_outside_allowed_area_is_violation`, `test_location_inside_allowed_area_passes` | – |
| `rules.py`, location remote (D3): positive, negative | same two tests (remote rows) | override at `apply_rules` level |
| `rules.py`, seniority: positive, negative | `test_intern_and_junior_roles_are_violation`, `test_other_seniorities_pass` | override by seniority alone at `apply_rules` level |
| `rules.py`, English flag: positive, negative | `test_strong_english_requirement_is_detected`, `test_weak_or_absent_...` | – |
| `schema.py`: invalid JSON rejected | `test_malformed_json_is_rejected` | JSON of the wrong shape (`[]`, `null`, a string, a number, empty) |
| `tools.py`: at least 1 test per tool | 6 for `check_rules`, 3 for `get_cv_section`, 3 for `convert_currency` | – |
| Anything that calls an LLM is mocked | fakes in every test module | `tests/conftest.py` refuses every socket connection |

**Hook (steps 3–4).** `.claude/settings.json` + `.claude/hooks/post_edit_check.py`.
Shown working in the session: an unused import was reported by ruff, a wrong test
expectation by pytest, both fed back to the agent; edits of `AGENTS.md` and `README.md`
ran nothing. The hook was picked up without restarting the session.

**Code review (step 5).** Done by a separate agent that had not written the code, so it
was not anchored on the author's assumptions. `metrics.py`, `dataset.py`, `schema.py`
and the dominance logic of `compare.py` came out clean; no CV or posting text reaches
logs or run files; no outward action besides the LLM call.

| # | Finding | Verdict | Action |
|---|---|---|---|
| R1 | `llm._generate_vertex` caught only `APIError`; a timeout or connection error (plain `httpx` error after the SDK's retries) would crash a whole eval run and save nothing | confirmed | **fixed**, failing test first |
| R2 | `rules._normalize` did not handle decomposed Unicode (`I` + combining dot, `u` + diaeresis): `İstanbul` or `Türkiye` in that form was treated as a blocked place, and the English keyword check missed `İngilizce` | confirmed | **fixed**, failing tests first; the English flag of all 21 postings is unchanged |
| R3 | Remote rule fails closed: `Fully remote`, `Remote-first`, `Remote (Europe)`, `Remote (TR)`, `Remote (not specified)` are SKIP because any leftover word counts as a region not on the allow-list. Same shape for office work: `Not specified`, or `Türkiye` without a city, is SKIP | confirmed | **fixed** (D33), failing tests first: a location that names no place, or only the country, is unknown |
| R4 | Substring allow-list lets blocked places through: `Remote (Anywhere in the US)`, `Remote (Global, excluding Turkey)`, `Istanbul Street 5, Berlin` | confirmed | **fixed** (D33), failing tests first: whole-word matching, and wording that excludes Türkiye violates. `Istanbul Street 5, Berlin` still passes: a street named after an allowed city cannot be told from the city without a gazetteer |
| R5 | `prompt_sha256` covers the system prompt and the user template, not the `JobDecision` field descriptions, the retry note or the CV; two runs with a changed schema description or CV get the same hash | confirmed | reported; a fix changes every hash and makes the saved runs incomparable |
| R6 | Tokens and latency of a failed posting are recorded as 0, so a model that fails often looks cheaper and can land on the Pareto front | confirmed | reported; no run in `docs/results.md` had a failed posting (known caveat since Phase 2) |
| R7 | Three consecutive `LLMError`s abort the run and discard the completed postings; empty answers and 429s count as setup failures | plausible | reported |
| R8 | `agent.py`: the `StructuredOutputError` branch cannot be reached with `ToolStrategy`'s default error handling; the step-limit branch handles the case | confirmed | reported; harmless |
| R9 | `check_rules` tool answers "no violation" for an unrecognized `work_mode` such as `on-site` | confirmed | reported; the rules in code still decide; a typed argument would change the tool text the model sees |
| R10 | A CV without `## ` headings gives the agent no sections and no error; a bare `## ` line matches every topic | confirmed | reported; agent is comparison-only (D27) |
| R11 | Provider error text is copied into the saved `error` field and can contain the Google Cloud project id | plausible | reported; run files are git-ignored |
| R12 | Prices are for `global`; with `GOOGLE_CLOUD_LOCATION` unset the default `us-central1` costs about 10% more than computed. English keyword check has false positives and negatives (`basic English` after `Strong Python`) | confirmed | reported; all table rows were run in `global` |

**Location rule after D33.** `docs/results.md` carries a note that its runs used the
earlier rule. In all saved runs the location rule fired on one posting only (Bursa,
on-site), which still fires; run files do not store the extracted location, so the
other direction cannot be checked. No re-run was made.

**GitHub (step 9).** Pre-push check on `main`: no `.env`, CV, run file, label notes
or key file tracked; no key-like string in the history (the one match is the line of
this plan that names the search patterns); the Google Cloud project id is not in the
history. Author and committer of all commits were rewritten to the `noreply`
address; the first commit also carried a corporate user name as author name, which
the first check had missed and the second caught. Branch renamed to `main`, pushed
after the candidate's go-ahead, verified with `git ls-remote`. The commit hashes
changed with the rewrite.

**Deviations from the plan above**
- The network guard blocks `socket.connect` instead of the SDK client constructors:
  it covers every SDK at once, and two existing tests legitimately construct a real
  (unconnected) OpenAI client.
- `AGENTS.md` has 7 lessons, not 8, to stay at 40 lines.
- Seen while working: the hook reports a failure between two edits of one change (a
  name used before its import was added). Expected (see Risks); adding the import
  first avoids it.

**Done when:** all tests and lint pass, the hook is shown working, the README is
complete, and the code is on GitHub.

**Not in this phase:** prompt or schema changes, new eval runs, new features, CI on
GitHub Actions (not asked for; the hook is the local check), the Phase 6 UI,
relabeling the test set.

**Risks**
- The hook runs after every code edit (about 8 s). During a multi-file change the
  tests are red between edits by design; the feedback is then noise the agent must
  read correctly. Accepted: PROCESS.md asks for exactly this hook.
- A hook is code that runs automatically on the candidate's machine. It is committed
  and reviewed like any other code, and it only runs ruff and pytest.
- The code review may find a bug whose fix changes results (e.g. in `rules.py`).
  Then the affected numbers in `docs/results.md` are marked, and whether to re-run
  is the candidate's decision, not made silently.
- Posting texts are other companies' text. In a private repo that is not an issue;
  for a public one it is part of D28.
- Rewriting authors (D29) changes every commit hash; the hashes are not referenced
  anywhere in the repo (the status table uses `git log --grep`), so nothing breaks.

---

## Phase 6 – UI, optional (in progress)

Source: [PROCESS.md §5 Phase 6](PROCESS.md#phase-6--optional-ui): "Gradio: paste a
posting, choose model and version (workflow/agent), see the `JobDecision` and the cost."

State at the start (checked 2026-11-11): working tree clean on `main`, 322 tests;
`gradio` is not a dependency; `workflow.decide` and `agent.decide_with_agent` share one
keyword signature and return the same `DecisionResult` (decision, LLM decision before
rules, fired rules, tokens, latency, attempts, tool calls);
`pricing.call_cost_usd(provider, model, input_tokens, output_tokens)` already turns
tokens into USD; `tests/conftest.py` refuses every socket connection, so no test can
start a web server.

The UI is a thin shell. It adds no decision logic, no prompt and no schema change, so
the prompt hashes and every number in `docs/results.md` stay valid.

### Before coding (candidate)
- [x] Answer D34–D37 (all decided as proposed, see Decided).
- [x] Approve this plan.

### Design

| Question | Answer | Why |
|---|---|---|
| Where does the logic live? | `ui.py` has a pure handler `run_decision(posting_text, choice, version, *, cv_loader, decide_fns) -> DecisionView` and a separate `build_app()` that only wires Gradio components to it. | The handler is testable with fakes, without Gradio and without a server (python-style: logic pure, I/O at the edges). |
| How are model choices defined? | One named table `MODEL_CHOICES`: label → (provider, model id, reasoning). Only Vertex rows (D35). | The agent supports only `vertex`; a free-text model box would invite ids that have no price or return 404. |
| What does the page show? | Decision (large), rationale, the other `JobDecision` fields as a table, fired rules and the LLM's decision before the rules, input/output tokens, cost in USD for this posting (and "price unknown" if `call_cost_usd` returns `None`), latency, model calls, and for the agent the tool calls in order (name and arguments). | The cost and the rule overrides are the point of the project; hiding them would make the UI a demo of the model only. |
| What happens on an error? | Empty posting → message, no model call. `LLMError` / `DecisionError` → one readable line in the page, no traceback, no CV or posting text in it. | Same failure types eval already handles; a stack trace in a browser can leak the project id (R11). |
| Who can reach the page? | `launch(server_name="127.0.0.1", share=False)`. No `--share` flag at all. | A share link would publish a page that sends the candidate's CV to a model for anyone who has the URL. |
| What leaves the machine? | Only the LLM call, as before. Gradio analytics are switched off (`analytics_enabled=False` and the `GRADIO_ANALYTICS_ENABLED` variable), flagging is off, nothing is written to disk, posting text and CV are not logged. | AGENTS.md: no personal data beyond the CV and posting texts in logs; Gradio's flagging saves inputs to a folder and its analytics call home by default. |
| Human control? | The page has one button, "Decide". No apply, send, open-link or save action. The text under the result states that it is a recommendation. | PROCESS.md Phase 5: the tool only recommends. |

### Steps
1. [x] Dependency and a spike. `uv add gradio` (or as an extra, D34); read the installed
       version's documentation instead of recalling the API. A throw-away script in
       the scratchpad settles: the Blocks/launch API of that version; how analytics
       and flagging are switched off and whether merely importing `gradio` or building
       a `Blocks` opens a socket (the test suite forbids it); how an error is shown
       without a traceback. Findings are written here; the script is not committed.
2. [x] `tests/test_harness.py`: add `gradio` to `ALLOWED_RUNTIME_DEPENDENCIES` in the
       same change as `pyproject.toml`, since the post-edit hook runs that test.
3. [x] `ui.py` – pure part
       - `MODEL_CHOICES`, `DEFAULT_MODEL_CHOICE`, `VERSIONS = ("workflow", "agent")`.
       - `DecisionView` (frozen dataclass): the strings and rows the page displays.
       - `run_decision(...)`: validate input, load the CV (D36), call the `decide_fn`
         of the chosen version, compute the cost, map everything to `DecisionView`;
         map `LLMError`, `DecisionError` and a missing CV file to an error view.
       - `format_cost(cost_usd)`: fractions of a cent need more digits than `$0.01`.
4. [x] `ui.py` – Gradio part
       - `build_app(decide_fns=..., cv_path=...) -> gr.Blocks`: posting textbox, model
         dropdown, version radio, "Decide" button, the outputs above.
       - `main(argv)`: `uv run python -m job_fit.ui [--port 7860] [--cv data/cv.md]`,
         bound to localhost.
5. [x] Tests (no network, no real LLM, no server started)
       - `test_ui.py`: workflow and agent routed to the right fake; the chosen row's
         provider, model and reasoning reach it; cost equals `call_cost_usd` for the
         fake's tokens (derived from the price table, not typed by hand); unknown
         price → "price unknown"; rule override shows both decisions and the rule
         name; agent tool calls listed in order; empty posting makes no call;
         `LLMError` and `DecisionError` become an error view that contains neither CV
         nor posting text; missing CV file → clear message; every `MODEL_CHOICES`
         row has a price and provider `vertex`; `build_app()` builds without
         launching (if the spike shows it opens no socket).
6. [x] `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`; show output.
7. [x] Live check, output shown: start the app, run one sample posting from
       `data/samples/` through the workflow and once through the agent, compare the
       decision, tokens and cost with the CLI output for the same posting, and check
       an empty posting. The page is driven through Gradio's local API (or a browser
       if the candidate prefers to click); a screenshot is not committed, because it
       would show a real posting next to a real decision about the candidate.
8. [x] `README.md`: a short "Web page" section (how to start it, what it shows, that it
       is local-only and only recommends). `AGENTS.md`: `ui` in the folder line and
       the run command, staying within 40 lines.
9. [~] `ruff format`, commit `phase-6: gradio ui`, update the status table; push only
       after the pre-push check and the candidate's go-ahead (D37).

### Results (2026-11-11)

`ruff check` and `ruff format --check` clean; 350 tests pass (322 at the start, 28 new
in `tests/test_ui.py`).

**Spike findings (step 1).** Installed: `gradio` 6.31.0.
- Importing `gradio` opens no socket. Building a `Blocks` object does: it creates an
  asyncio event loop, which on Windows connects a loopback socket pair, and the test
  suite refuses every connection. So there is no `build_app()` test; the wiring is
  covered by the live check.
- Analytics are switched off per app (`Blocks(analytics_enabled=False)`) and by the
  `GRADIO_ANALYTICS_ENABLED` variable; both are set.
- Flagging exists only on `gr.Interface`, not on `Blocks`, so there is nothing to
  switch off. `launch` is called with `share=False`, `ssr_mode=False`,
  `mcp_server=False`, `pwa=False`.
- Handled errors are returned as a view with a message, so Gradio's error path (and
  its traceback option) is not used.

**Live check (step 7).** The page was started on `127.0.0.1` and driven through its
local API with `data/samples/izmir-senior-software-engineer.txt`, default model.

| Request | Decision | Before rules | Rules fired | Tokens in / out | Cost | Latency | Model / tool calls |
|---|---|---|---|---|---|---|---|
| empty posting | none, message shown | – | – | 0 / 0 | – | – | 0 / 0 |
| workflow | SKIP | SKIP | location | 2,619 / 2,649 | $0.0074 | 17.7 s | 1 / 0 |
| agent | SKIP | SKIP | location | 7,428 / 1,896 | $0.0070 | 12.7 s | 3 / 3 |

- The workflow row has the same tokens, decision and rule as the Phase 1 CLI run of
  this posting (2,619 / 2,649, SKIP, location), and the cost is the price table applied
  to those tokens (2,619 × 0.30 + 2,649 × 2.50 per 1M = $0.0074).
- Share URL: none. Host names looked up while the app was built, launched and idle:
  none besides `127.0.0.1`. During a decision: `aiplatform.googleapis.com` and
  `oauth2.googleapis.com` (the model call). The test client used to drive the page
  (`gradio_client`) looks up `huggingface.co` on its own; the app does not.
- The page was not looked at in a browser by Claude; layout and readability are for
  the candidate to judge.

**Deviations from the plan above**
- `run_decision` takes `load_cv` (a function returning the CV text) and `decide_fns`;
  the name in the plan was `cv_loader`.
- Model text (rationale, extracted facts, tool arguments, error messages) is shown in
  plain-text components and tables, never as Markdown or HTML. Not in the plan: a
  posting could talk the model into writing an image link, which a Markdown component
  would load, sending whatever is in the URL to another server.
- `README.md`: "never ... opens a web page" became "never ... visits a job site",
  because the tool now serves a local page.
- `main` checks at start that the CV file exists (exit code 1), in addition to reading
  it on every request (D36).

**Done when:** `uv run python -m job_fit.ui` opens a local page where a pasted posting,
a model and a version give the `JobDecision` and the cost; tests and lint pass.

**Not in this phase:** new eval runs or result tables, prompt or schema changes,
OpenRouter/Ollama choices in the page, fetching a posting from a URL, file upload, CV
editing, history of past decisions, login, deployment or a public link, streaming.

**Risks**
- Gradio is a large dependency (web server, many transitive packages). It is imported
  only by `ui.py`; nothing else in the package depends on it.
- The API differs between Gradio major versions; step 1 reads the installed version
  instead of trusting memory.
- If building a `Blocks` object needs a socket, the build test is dropped and the
  wiring is covered by the live check only; the handler tests are unaffected.
- A decision takes about 12 s (workflow) to 15 s (agent) with the default choice; the
  page shows a progress state, not a streamed answer.
- Spend: two or three live decisions, about 2 cents at list price.
- Running the agent with a model other than the default is unmeasured (D35); the page
  says so next to the choice.
