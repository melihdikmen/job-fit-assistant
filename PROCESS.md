# Job Fit Assistant – Process Document

> AI FDE course, Week 1 project.
> This file explains what the project is, the order of work, and when each step counts as "done".
> Claude Code works from this file: at the start of each phase it reads the relevant section and drafts a plan.

---

## 1. Goal

A tool that takes a job posting and the candidate's CV and returns one of three decisions:

| Decision | Meaning |
|---|---|
| `APPLY` | Good fit, apply now |
| `PREPARE` | Mostly a fit with 1–2 gaps; apply after short preparation |
| `SKIP` | Not a fit (location, language, skills or seniority don't match) |

The real goal is not just to decide, but to **measure** the decision: which model, at what cost, decides how accurately on a labeled test set? Is a fixed workflow better, or an agent?

### Candidate rules (enforced in code, not left to the LLM)
- Location: must be **Ankara**, **Istanbul** (on-site or hybrid) or **remote (Türkiye)**. On-site in any other city → `SKIP`.
- English: if the posting requires **fluent English** or the team works in English → raise a flag (`english_risk = true`). Not a reason to skip on its own.
- Seniority: internship and junior roles → `SKIP`.

---

## 2. Tech stack

- Python 3.12+, package manager: `uv`
- LLM access:
  - **Google Vertex AI** (primary, free credits available): Gemini models, plus partner models such as Claude through Model Garden where enabled. Use the `google-genai` SDK, or Vertex's OpenAI-compatible endpoint for Gemini.
  - OpenRouter (optional, OpenAI-compatible) for models not available on Vertex
  - Ollama (local)
- Schema: Pydantic v2
- Agent: LangChain `create_agent` + `@tool`
- UI (optional, Phase 6): Gradio
- Testing and quality: `pytest`, `ruff`
- Secrets: `.env` file (never committed). `.env.example` kept in the repo as a template.

---

## 3. Folder structure

```
job-fit-assistant/
├── AGENTS.md            # Short instructions for Claude Code, links here
├── PROCESS.md           # This file
├── PLAN.md              # Claude's current plan and status per phase
├── .env.example
├── data/
│   ├── cv.md            # Candidate's CV (plain text)
│   ├── postings.jsonl   # Labeled test set
│   └── results/         # Output of each eval run (not committed)
├── src/job_fit/
│   ├── schema.py        # Pydantic models
│   ├── rules.py         # Rules enforced in code
│   ├── llm.py           # Vertex AI / OpenRouter / Ollama client
│   ├── workflow.py      # Fixed workflow (single LLM call)
│   ├── agent.py         # Agent version
│   ├── tools.py         # Agent tools
│   └── eval.py          # Evaluation against the test set
├── tests/
└── docs/
    └── results.md       # Model comparison tables and notes
```

---

## 4. Working rules (for Claude Code)

1. **Plan first, then code.** At the start of each phase, write the steps to `PLAN.md` and do not implement before approval.
2. **One phase at a time.** Don't start the next phase until the current one is done. Commit at the end of each phase. Commit message: `phase-N: <short description>`.
3. **`/clear` between phases.** The repo is the memory: write status to `PLAN.md`, don't rely on the chat.
4. **After every change** run `uv run ruff check .` and `uv run pytest`. Don't say "done" until both pass.
5. **Show evidence.** Don't say "it works"; show the command output.
6. **Never put API keys in code.** Read them only from `.env`.
7. **Personal data:** Don't log any personal data other than `data/cv.md` and posting texts.
8. **When in doubt, ask.** Don't assume.

---

## 5. Phases

### Phase 0 – Setup
**Tasks**
- `uv init`, add `pytest` and `ruff`, create the folder structure.
- `AGENTS.md` (max 40 lines): purpose, commands, folders, rules. Link to this file for details.
- `.env.example`:
  ```
  GOOGLE_CLOUD_PROJECT=
  GOOGLE_CLOUD_LOCATION=us-central1
  GOOGLE_GENAI_USE_VERTEXAI=true
  OPENROUTER_API_KEY=
  OLLAMA_BASE_URL=http://localhost:11434/v1
  ```
- Vertex authentication: `gcloud auth application-default login` on the developer machine. Never commit service-account JSON keys.
- `.gitignore`: `.env`, `data/results/`, `__pycache__/`, `.venv/`

**Done when:** `uv run pytest` runs (even with no tests), `ruff` is clean, first commit is made.

---

### Phase 1 – Structured output (fixed workflow)
**Topics:** LLMs are stateless, tokens, context window, structured outputs with Pydantic, rationale first.

**Tasks**
- `schema.py`: a `JobDecision` model. **Field order matters: rationale first.**
  ```python
  class JobDecision(BaseModel):
      rationale: str               # Think first: 3–5 sentences
      company: str
      role: str
      location: str
      work_mode: Literal["remote", "hybrid", "onsite", "unknown"]
      english_risk: bool
      seniority: Literal["intern", "junior", "mid", "senior", "lead", "unknown"]
      matching_skills: list[str]
      missing_skills: list[str]
      decision: Literal["APPLY", "PREPARE", "SKIP"]
  ```
- `llm.py`: a single client where the model name and provider (`vertex`, `openrouter` or `ollama`) are parameters. Start with Vertex (Gemini supports structured output via a response schema). The rest of the code must not care which provider is used.
- `workflow.py`: `decide(posting_text, cv) -> JobDecision`
  - System prompt + posting + CV → one call → validate with Pydantic.
  - On a schema error, retry **once**; if it still fails, raise a clear error.
  - Return input tokens, output tokens and latency for every call.
- `rules.py`: the rules from Section 1. **Rules override the LLM's decision.** For example, if the LLM says `APPLY` but the job is on-site in Bursa, the result becomes `SKIP` and "rule: location" is appended to the rationale.

**Done when:** It works on 3 sample postings, output matches the schema, tokens and latency are printed, and `rules.py` has at least 5 tests.

---

### Phase 2 – Test set and evaluation
**Topics:** Writing your own test, labels, a baseline, the business cost of false positives and false negatives.

**Test set format** – `data/postings.jsonl`, one posting per line:
```json
{"id": "miny-2026-11", "company": "Miny", "role": "Engineer — AI Inference & Product",
 "text": "...full posting text...", "label": "APPLY", "note": "RN+TS main stack, Go missing"}
```
- At least **20 postings**. Labels are set by **the candidate**, not by a model.
- Aim for balance: at least 5 examples of each decision.

**Metrics** – `eval.py`
- Accuracy
- Confusion matrix (3×3)
- **Weighted error score** (lower is better):

  | Actual \ Predicted | APPLY | PREPARE | SKIP |
  |---|---|---|---|
  | **APPLY** | 0 | 1 | **3** (missed a good job) |
  | **PREPARE** | 1 | 0 | 2 |
  | **SKIP** | **1** (wasted application) | 1 | 0 |

- **Baseline:** a "dumb model" that always predicts the most common label. If the real model can't beat it, it isn't useful.
- Total tokens, estimated cost (USD) and average latency.
- Each run is saved as `data/results/<date>_<model>.json`.
- Wrongly decided postings are listed with their rationale, so errors can be reviewed one by one.

**Done when:** `uv run python -m job_fit.eval --model <model>` prints all metrics in one command, including the baseline.

---

### Phase 3 – Model selection and cost
**Topics:** Trading off intelligence, speed and cost; the Pareto front; OpenRouter; open-weight and local models; quantization; reasoning effort.

**Tasks**
- Pick **3 cloud models**: one expensive and strong, one mid-range, one cheap and fast. Use Vertex AI first to spend the free credits (e.g. a Pro-tier and a Flash/Flash-Lite-tier Gemini model, and a Claude model via Model Garden if enabled). Use OpenRouter only for a model Vertex doesn't offer. Check a benchmark such as Artificial Analysis to build the shortlist, but let your own test set decide.
- Track spend against the free credits in the Google Cloud console, and set a **budget alert** before running the full test set.
- Run **1 local model** with Ollama: 7–8B parameters, 4-bit quantized.
  - First estimate the memory needed and write it to `docs/results.md`: `parameters (billions) × bits ÷ 8 ≈ GB` + room for context.
- On a model that supports reasoning, run once with **reasoning effort off** and once **on**.
- If you use OpenRouter, pin the provider (provider order). Different providers can differ in price and precision.
- Record the cost using the list price per token even when credits pay for it, so the numbers are comparable.

**Results table** – `docs/results.md`:

| Model | Accuracy | Weighted error | Cost per 100 postings | Avg. latency | Notes |
|---|---|---|---|---|---|
| Baseline | | | 0 | 0 | |
| Model A | | | | | |
| Model B | | | | | |
| Model C | | | | | |
| Local (Ollama) | | | 0 (+ electricity) | | |
| Model X + reasoning | | | | | |

- Below the table, a **"Which model I chose and why"** paragraph: which one sits best on the Pareto front? Which one would you pick if privacy were required?

**Done when:** The table is filled in and the choice is justified in writing.

---

### Phase 4 – Agent version
**Topics:** Agent = model + harness, tool calling, frameworks, "a fixed workflow can beat an agent".

**Tools** – `tools.py`, LangChain `@tool`:
- `check_rules(location: str, work_mode: str, english_required: bool, seniority: str) -> dict`: calls `rules.py`.
- `get_cv_section(topic: str) -> str`: returns the relevant part of the CV (experience, projects, skills).
- `convert_currency(amount: float, currency: str) -> float`: converts a USD or EUR salary to TRY. If no FX API is available, use a fixed rate and say so.

**Agent** – `agent.py`:
- `create_agent` with model + tools + system prompt. Output is still a `JobDecision`.
- Set a **maximum step limit** (e.g. 6) to avoid infinite loops.
- Log which tool was called with which arguments.

**Comparison:** same test set, same model, agent vs. workflow:

| Version | Accuracy | Weighted error | Cost | Latency | Avg. tool calls |
|---|---|---|---|---|---|
| Workflow | | | | | 0 |
| Agent | | | | | |

- Conclusion paragraph: **Is an agent needed for this problem?** Give a reasoned answer.

**Done when:** Both versions pass eval, the comparison table and conclusion are written.

---

### Phase 5 – Harness, Harden, Humans
**Topics:** Checks (linters, tests, hooks), human accountability, "review the code even when tests pass".

**Tasks**
- Tests:
  - `rules.py`: a positive and a negative test for every rule.
  - `schema.py`: a test that invalid JSON is rejected.
  - `tools.py`: at least 1 test per tool. Anything that calls an LLM is mocked.
- Claude Code **hook**: run `ruff` and `pytest` automatically after a file is edited.
- Update `AGENTS.md` with lessons learned while working with the agent, e.g. "it made this mistake, here's how it was prevented". This is the harness improving itself.
- README: what the project does, how to run it, result tables, the chosen model and why.
- **Human control:** the tool only recommends. It never applies, sends messages or fills in forms.

**Done when:** All tests and lint pass, the hook works, the README is complete, and the code is on GitHub.

---

### Phase 6 – (Optional) UI
- Gradio: paste a posting, choose model and version (workflow/agent), see the `JobDecision` and the cost.

---

## 6. The interview story

When the project is done, these questions should be answerable with numbers:
1. Which model did you choose for this task, and why? What were the accuracy, cost and latency?
2. What was the baseline, and by how much did your model beat it?
3. Why is a false negative more expensive here, and how did you build that into the evaluation?
4. Why did you put the rationale field before the decision field?
5. Which decisions did you leave to the LLM, and which did you write in code?
6. Was an agent necessary? When is an agent unnecessary?
7. When does a local model make sense, and how did you estimate its memory needs?
8. How did you work with Claude Code? (AGENTS.md, plan, phases, hooks, code review)

---

## 7. Week 2 – Become an AI Architect

Phases 0–6 are done. Week 2 extends the same project; it does not start a new one. The repo stays the memory: same `AGENTS.md`, same `PLAN.md`, same working rules (Section 4).

**Goal of Week 2:** turn Job Fit from "one decision per posting" into a small, reliable system that processes a batch of postings, asks the human at the right moment, measures itself properly, and can suggest its own improvements without being trusted blindly.

**Still true:** the tool only recommends. It never applies, sends messages or fills in forms.

### Phase 7 – Better evals and observability
**Topics:** three kinds of evals, confusion matrix, precision/recall, cost of mistakes, Cohen's kappa, noise, LLM as a judge, traces.

**Tasks**
- **Re-label the test set yourself.** The current labels started from Claude's suggestions. Go through all postings and fix them (Miny, a1mobile, Rubikpara are known doubtful ones). Grow the set to at least 40 postings, still at least 5 per class.
- **Split the data:** `data/postings_dev.jsonl` (used while improving prompts) and `data/postings_heldout.jsonl` (about 30%, never looked at while tuning).
- `eval.py` additions:
  - Precision and recall per class (especially APPLY recall: missed good jobs).
  - **Cohen's kappa** against the labels. Show that the "always most common label" baseline scores 0.
  - **Noise:** run each configuration 3 times and report mean ± spread. A difference smaller than the spread is not a real improvement.
  - **Sensitivity check on the cost matrix:** rerun the weighted error with 2 alternative weightings (e.g. missed APPLY = 2 and = 5). Does the best model change?
- **LLM as a judge** for the `rationale` field: a second, different model scores each rationale 1–5 against a written rubric (mentions location rule, language risk, matching and missing skills, consistent with the decision). Check the judge on 10 rationales you scored yourself; report agreement.
- **Observability:** send every call to Langfuse (free tier or self-hosted): prompt, output, tokens, latency, cost, model. Tag runs by experiment name.

**Done when:** one command prints per-class precision/recall, kappa, mean ± spread over 3 runs, judge score, and the results table in `docs/results.md` is updated with these columns.

---

### Phase 8 – Agent architecture: logic in code, sub-agents
**Topics:** the three traps, start with a business problem and a baseline, start with one agent, sub-agents, business logic in code.

**Tasks**
- Write in `docs/architecture.md` (max 1 page): the business problem, the baseline, and why each LLM call exists. Name which of the three traps ("the agent for X", human-like agents, plausible-looking output) the old agent version fell into, if any.
- Move any remaining business logic out of prompts into `rules.py` and expose it as tools (location, language, seniority, salary conversion).
- Try **one** sub-agent only if it has a clear reason: e.g. a cheaper model that extracts structured facts from the posting (location, work mode, language, stack) before the main decision. Measure: does it improve accuracy or cost on the dev set? Keep it only if yes.

**Done when:** architecture doc exists, and the sub-agent experiment has a keep/drop decision backed by numbers.

---

### Phase 9 – Agentic RAG, MCP and memory
**Topics:** RAG, agentic RAG, retrieval evals, MCP (stdio vs HTTP), short- and long-term memory.

**Tasks**
- **RAG over your own material:** chunk and embed `cv.md`, project READMEs (Vouch, AcenteDefteri, StreetShelf, Job Fit) and short notes about past jobs. Store in a local vector store (Chroma or pgvector).
- **Agentic RAG:** the decision step may call `search_experience(query)` itself, as many times as it decides (cap at 3). The output adds an `evidence` list with the chunks used.
- **Retrieval eval:** for 15 postings, write by hand which chunks should be found. Report recall@3 and MRR. Did the facts get found?
- **MCP server (stdio):** expose `decide_posting(text)` and `search_experience(query)` as an MCP server so Claude Code / Claude Desktop can call it. Keep it local; no secrets in tool descriptions.
- **Long-term memory (one kind only):** a `preferences.md` file the tool reads at start (e.g. "no on-site outside Ankara/İstanbul", "salary floor 150k TL net"). When you override a decision, the tool may propose a new preference line; you approve it before it is written.

**Done when:** retrieval recall@3 and MRR are reported, the MCP server works from Claude Code, and at least one preference was learned through an approved override.

---

### Phase 10 – Durable state, human in the loop, batch
**Topics:** durable execution, LangGraph state/nodes/edges, checkpoints, `interrupt()`, dispatcher, retries, max concurrency.

**Tasks**
- Rebuild the pipeline as a **LangGraph** graph: `load_posting → extract → decide → (human review) → record`.
- **Checkpoints** in SQLite, one thread per posting. Kill the process in the middle of a batch and restart: it must continue where it stopped, without deciding a posting twice (use the posting id as an idempotency key).
- **Human in the loop with `interrupt()`:** pause for your approval when the decision is APPLY, when confidence is low, or when the rule layer overrode the LLM. Record what you saw, what you chose and how long it took. Watch for decision fatigue: if you approve 20 in a row without changes, note it.
- **Dispatcher:** process a batch of 30 postings with `max_concurrency` set (e.g. 4) and a `RetryPolicy` for 429/timeout errors.
- Keep side effects (writing to `decisions.jsonl`) in their own node, so a resumed run does not write twice.

**Done when:** a 30-posting batch survives a forced crash, human approvals are recorded, and no posting appears twice in the output.

---

### Phase 11 – Self-improvement (guarded)
**Topics:** the improver, keep or roll back, limits, held-out data, recursion.

**Tasks**
- An **improver** script reads the latest eval results and the wrong decisions, and proposes **one** change: a prompt edit, a model swap or a reasoning-effort change. It writes the proposal to `proposals/NNN.md`.
- **You approve** before it runs. Then evals run on the dev set.
- **Keep or roll back rule:** keep the change only if weighted error improves by more than the noise measured in Phase 7, and no other key metric (APPLY recall, kappa, cost per 100 postings) gets much worse.
- **Limits:** floors on APPLY recall, a cost budget per run, an allowed-models list, and human approval for every change.
- Final check on the **held-out set** only at the end, once. If held-out results are much worse than dev, you overfit; write that down.
- (Optional) Let the improver also suggest one change to its own instructions. Same rules apply.

**Done when:** at least 3 improver cycles are logged with keep/roll-back decisions, and the held-out result is reported.

---

### Week 2 interview story
Add these to Section 6:
9. How did you know an improvement was real and not noise?
10. What did Cohen's kappa tell you that accuracy did not?
11. When did you choose a sub-agent, and when did you drop it?
12. How did retrieval quality affect the final decision quality?
13. Where does the human step in, and how did you avoid decision fatigue?
14. What happens if the process crashes in the middle of a batch?
15. How did you stop the self-improver from overfitting or breaking things?

---

## 8. Status tracking

At the end of each phase, Claude Code updates this table at the top of `PLAN.md`:

| Phase | Status | Commit | Notes |
|---|---|---|---|
| 0 – Setup | ⬜ | | |
| 1 – Structured output | ⬜ | | |
| 2 – Test set and eval | ⬜ | | |
| 3 – Model selection | ⬜ | | |
| 4 – Agent | ⬜ | | |
| 5 – Harden | ⬜ | | |
| 6 – UI (optional) | ⬜ | | |
| 7 – Evals & observability | ⬜ | | |
| 8 – Agent architecture | ⬜ | | |
| 9 – RAG, MCP, memory | ⬜ | | |
| 10 – Durable state & HITL | ⬜ | | |
| 11 – Self-improvement | ⬜ | | |