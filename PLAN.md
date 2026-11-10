# PLAN.md

## Status

| Phase | Status | Commit | Notes |
|---|---|---|---|
| 0 – Setup | ✅ | see `git log --grep phase-0` | uv project, ruff + pytest configured, 1 smoke test |
| 1 – Structured output | ✅ | see `git log --grep phase-1` | `gemini-2.5-flash` on Vertex; 3 samples run live; 75 tests |
| 2 – Test set and eval | ✅ | see `git log --grep phase-2` | `gemini-2.5-flash`: 66.7% vs. 57.1% baseline, mean w.err 0.48 vs. 1.14; 21 postings, PREPARE has only 3 (D15); 149 tests |
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
