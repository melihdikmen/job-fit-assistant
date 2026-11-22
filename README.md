# Job Fit Assistant

Give it a job posting and a CV. It returns a structured decision, `APPLY`, `PREPARE` or
`SKIP`, with the reasoning behind it. The project also **measures** that decision: on a
labeled test set it compares models on accuracy, cost and latency, and compares a fixed
workflow against an agent.

| Decision | Meaning |
|---|---|
| `APPLY` | Good fit, apply now |
| `PREPARE` | Mostly a fit with one or two gaps; apply after short preparation |
| `SKIP` | Not a fit (location, language, skills or seniority) |

Built as the Week 1 project of an AI FDE course.

## The tool only recommends

It reads text and shows a decision. It never applies to a job, sends a message, fills
in a form or visits a job site; the person reads the recommendation and acts on it.
Two tests keep it that way: the runtime dependencies must match an allow-list, and the
agent's tool set must be exactly its three read-only tools
([tests/test_harness.py](tests/test_harness.py)).

## How it works

1. **One model call** gets the posting and the CV and must answer with JSON that matches
   the `JobDecision` schema ([src/job_fit/schema.py](src/job_fit/schema.py)). The
   `rationale` field comes first and `decision` last, so the model writes its reasoning
   before it commits to an answer.
2. The answer is **validated** with Pydantic. An invalid answer is retried once; a
   second failure is an error, not a guess.
3. **Hard rules run in code** and override the model
   ([src/job_fit/rules.py](src/job_fit/rules.py)):
   - on-site or hybrid outside Ankara and Istanbul → `SKIP`
   - remote but not open to Türkiye → `SKIP`
   - internship or junior role → `SKIP`
   - a strong English requirement raises the `english_risk` flag; it never changes the
     decision by itself

   A rule that fires appends `[rule: <name>]` to the rationale. Unknown facts never
   trigger a rule: wrongly skipping a good job is the most expensive mistake.

What is left to the model: reading the posting, extracting the facts, and judging the
skill match. What is in code: everything the candidate would never negotiate on.

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and a Google Cloud project with
Vertex AI enabled.

```
uv sync
cp .env.example .env                      # set GOOGLE_CLOUD_PROJECT
gcloud auth application-default login
cp data/cv.example.md data/cv.md          # write your CV; the file is git-ignored
uv run pytest
```

API keys are read only from `.env`, which is never committed.

## Usage

Decide one posting:

```
uv run python -m job_fit.workflow --posting data/samples/accenture-software-engineer.txt --model gemini-2.5-flash
```

Or use the local web page: paste a posting, choose a model and a version (workflow or
agent), and see the decision, the rationale, the rules that fired, tokens, latency and
the cost of that one decision at list price:

```
uv run python -m job_fit.ui               # then open http://127.0.0.1:7860
```

The page listens on this machine only and has no share-link option, because every
request sends the CV in `data/cv.md` to the chosen model. It has one button, "Decide",
and stores nothing. Gradio's usage analytics are switched off.

Evaluate a configuration on the dev set (`data/postings_dev.jsonl`). One command prints
accuracy, the confusion matrix, precision and recall per class, Cohen's kappa, the
weighted error under three cost matrices, the baselines, tokens, cost and latency,
lists every wrong decision with its rationale, and saves each run to `data/results/`.
`--runs 3` repeats the configuration and reports mean and spread; `--judge` lets a
second model rate the rationales:

```
uv run python -m job_fit.eval --model gemini-2.5-flash --reasoning on --runs 3 --judge
uv run python -m job_fit.eval --model gemini-2.5-flash --reasoning on --version agent --runs 3
uv run python -m job_fit.eval --model gemini-2.5-flash --reasoning on --version extractor --runs 3
```

Build the comparison tables from saved runs:

```
uv run python -m job_fit.compare --table reference data/results/<run>.json ...
uv run python -m job_fit.compare --table costs data/results/<run>.json ...
```

Maintain the labeled set and check the judge against your own scores:

```
uv run python -m job_fit.dataset label --source <rows.jsonl>   # label postings yourself
uv run python -m job_fit.dataset split                         # dev and held-out files
uv run python -m job_fit.judge sample --run data/results/<run>.json
uv run python -m job_fit.judge score
uv run python -m job_fit.judge agreement
```

The held-out set (`data/postings_heldout.jsonl`) is refused by `eval` unless
`--final-heldout` is given. It is looked at once, at the very end.

Live calls cost money (list prices are in the tables below). The tests make no network
calls: every LLM call is mocked, and a fixture refuses any socket connection.

### What leaves the machine

- The posting and the CV go to the model provider (Vertex AI) with every decision.
- If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in `.env`, every model call
  is also traced to Langfuse: prompt, output, tokens, latency, cost, model. The CV text
  is replaced by a hash marker in every trace; posting texts and the model's rationale
  are traced as they are. Without the keys nothing is traced.

## How decisions are scored

The labeled set has 40 real postings (APPLY 9, PREPARE 6, SKIP 25), split by a seeded
shuffle into a dev set of 27 and a held-out set of 13. Every label was suggested by
Claude and then confirmed or changed by the candidate. Mistakes are not equally expensive, so besides accuracy each run gets a
**weighted error** (lower is better):

| Actual \ Predicted | APPLY | PREPARE | SKIP |
|---|---|---|---|
| **APPLY** | 0 | 1 | **3** (missed a good job) |
| **PREPARE** | 1 | 0 | 2 |
| **SKIP** | 1 (wasted application) | 1 | 0 |

The baseline is a "model" that always answers `SKIP`, the most common label. Because
that baseline is right on most postings, accuracy flatters it; **Cohen's kappa**
measures agreement beyond what the label mix gives for free, and the baseline's kappa
is exactly 0.

## Results

Full tables, caveats and reasoning: [docs/results.md](docs/results.md).

### Week 2 reference (current)

Dev set, 27 postings, three runs per configuration, mean ± standard deviation.

| Configuration | Accuracy | Weighted error (mean) | APPLY recall | Kappa | Cost per 100 postings | Avg. latency |
|---|---|---|---|---|---|---|
| Baseline (always SKIP) | 63.0% | 0.96 | 0.0% | 0.00 | $0.00 | 0.0 s |
| Workflow, gemini-2.5-flash, reasoning on | 65.4% ± 2.1% | 0.51 ± 0.06 | 44.4% ± 9.6% | 0.37 ± 0.04 | $0.62 | 11.4 s |
| Workflow, gemini-2.5-flash, reasoning off | 60.5% ± 2.1% | 0.40 ± 0.02 | 44.4% ± 9.6% | 0.38 ± 0.02 | $0.16 | 2.7 s |
| Workflow, gemini-3.1-flash-lite | 55.6% ± 0.0% | 0.44 ± 0.00 | 50.0% ± 0.0% | 0.32 ± 0.00 | $0.10 | 1.9 s |
| Agent, gemini-2.5-flash, reasoning on | 67.9% ± 5.7% | 0.35 ± 0.08 | 72.2% ± 9.6% | 0.48 ± 0.09 | $0.74 | 14.3 s |

- **Noise figure: 0.077** of mean weighted error between runs of one configuration. A
  smaller difference between two configurations is not an improvement.
- Accuracy barely beats the baseline; kappa shows that every configuration does real
  work (0.32 to 0.48 against 0).
- The Week 1 choice (workflow, reasoning on) has the worst weighted error here: it is
  the only configuration that rates good jobs as `SKIP` (5 times in three runs). The
  agent has the best, by twice the noise figure against that workflow, and is still
  the least stable (8 of 27 predictions changed between its runs).
- The ranking is the same whether a missed good job costs 2, 3 or 5.
- A judge model rated the rationales 4.1 to 4.6 out of 5, but it agreed with the
  candidate's own scores on only 4 of 10 rationales, so that score is reported and not
  relied on.
- The held-out set has not been run. The Week 1 choice is not changed yet:
  the next phases look for why that workflow rates good jobs as `SKIP`; see the end of
  [docs/results.md](docs/results.md).

### Sub-agent experiment (Phase 8)

A cheap model (`gemini-3.1-flash-lite`) extracts the facts from the posting, then
`gemini-2.5-flash` judges the fit; three runs per row against the workflow rows above.
The keep-or-drop rule was written down before the runs.

| Pair | Weighted error (mean) | APPLY recall | Cost per 100 postings | Avg. latency |
|---|---|---|---|---|
| Reasoning on: workflow → with extractor | 0.51 → 0.48 | 44.4% → 38.9% | $0.62 → $0.58 | 11.4 → 12.3 s |
| Reasoning off: workflow → with extractor | 0.40 → 0.37 | 44.4% → 50.0% | $0.156 → $0.154 | 2.7 → 4.0 s |

**Dropped.** The change is a third of the noise figure. The hard rules fired on the same
three postings in every run of both versions, so the facts were already read correctly;
the costly mistake is made when the model weighs one missing skill.
[docs/architecture.md](docs/architecture.md) explains why each model call exists and
what Phase 8 moved into code.

The two sections below are the Week 1 results, on the earlier set of 21 postings. They
are not comparable with the tables above.

### Model comparison (Week 1)

| Model | Accuracy | Weighted error (mean) | Cost per 100 postings | Avg. latency | Notes |
|---|---|---|---|---|---|
| Baseline | 57.1% | 1.14 | $0.00 | 0.0 s | always SKIP |
| gemini-4-carbon | 66.7% | 0.62 | $1.75 | 57.3 s | strong tier; promotional price |
| gemini-2.5-flash, reasoning off | 52.4% | 0.48 | $0.16 | 2.6 s | |
| gemini-3.1-flash-lite | 52.4% | 0.48 | $0.10 | 1.7 s | Pareto front |
| **gemini-2.5-flash, reasoning on** | **71.4%** | **0.43** | $0.64 | 11.9 s | Pareto front, chosen |
| Open-weight via OpenRouter (free) | not run | not run | 0 | – | stopped by the daily free-model quota |
| Local (Ollama) | not run | not run | 0 (+ electricity) | – | a 7–8B model does not fit 3 GB of VRAM |

**Chosen: `gemini-2.5-flash` with reasoning on.** It has the best accuracy and the
lowest weighted error, and it is on the Pareto front. Switching reasoning on was the
change that mattered: the same model goes from 52.4% to 71.4%, at 4 times the cost. The
most expensive model was not the best; it was too strict and rated good jobs as `SKIP`.
For a handful of postings a day, $0.0064 and 12 seconds per posting are irrelevant; for
thousands of postings, Flash-Lite would be the better choice.

### Workflow vs. agent (Week 1)

Same model and settings. The agent (LangChain `create_agent`, three tools, at most six
model calls) sees the posting and reads the CV through a tool.

| Version | Accuracy | Weighted error (mean) | Cost per 100 postings | Avg. latency | Avg. model calls | Avg. tool calls |
|---|---|---|---|---|---|---|
| Workflow, mean of 2 runs | 69.0% | 0.45 | $0.64 | 12.0 s | 1.00 | 0.00 |
| Agent, mean of 2 runs | 73.8% | 0.26 | $0.75 | 14.9 s | 3.69 | 3.71 |

**An agent is not needed here.** In 42 decisions the agent made the same tool calls in
the same order, which is a fixed workflow that the model re-derives on every posting. It
cost 18% more, took 24% longer, and was far less stable: its two runs disagree on 8 of
21 postings (85.7% and 61.9% accuracy), the workflow's on 2. An agent would earn its
cost if the steps depended on each other, for example looking up the company or asking
a question when a posting is unclear.

### Limits

- Week 1: 21 postings, so one posting is about 5 points of accuracy. Week 2: 27 dev
  postings and three runs; the standard deviations come from three values each.
- `PREPARE` has 4 dev postings (3 in Week 1); nothing can be said about that class.
- The labels are one person's judgment, and in Week 2 they were confirmed with a
  suggested label on screen, not given independently.
- No local and no open-weight model was measured.

## Project layout

```
src/job_fit/
  schema.py     JobDecision (Pydantic)
  rules.py      hard rules, pure functions
  llm.py        the only module that talks to a provider (Vertex AI, OpenRouter, Ollama)
  workflow.py   fixed workflow: one call, validate, retry once, apply rules
  agent.py      agent version (comparison only)
  tools.py      the agent's three read-only tools
  extract.py    extractor version: a cheap sub-agent reads the facts, the model judges
                the fit (measured in Phase 8, comparison only)
  dataset.py    load, validate, label and split the labeled set
  metrics.py    accuracy, confusion matrix, precision/recall, kappa, weighted error, baselines
  pricing.py    dated list prices
  eval.py       run a configuration over the dev set (once or repeatedly), report, save
  judge.py      a second model rates the rationales; checked against the candidate
  tracing.py    optional Langfuse tracing with the CV masked
  compare.py    tables from saved runs: Pareto front, spread over runs, cost matrices
  ui.py         local Gradio page around the workflow and the agent
data/           postings.jsonl (labeled set), postings_dev.jsonl, postings_heldout.jsonl,
                label_log.jsonl, judge_calibration.jsonl, postings/, samples/, cv.example.md
tests/          pytest suite, no network
docs/results.md full results
docs/architecture.md  the business problem, the baseline, why each LLM call exists
```

## How it was built

The code was written with Claude Code, one phase at a time, under a small harness:

- [AGENTS.md](AGENTS.md): short standing instructions, plus lessons learned from real
  mistakes.
- [PROCESS.md](PROCESS.md): the phases and when each counts as done.
- [PLAN.md](PLAN.md): a plan per phase, approved before any code, with every decision
  recorded. The chat is cleared between phases; the repository is the memory.
- A hook ([.claude/settings.json](.claude/settings.json)) runs `ruff` and `pytest` after
  every code edit and hands failures back to the agent.
- A code review at the end, because passing tests are not proof of correct code.
- The conclusions (which model, agent or not) were drafted from the numbers by the
  agent and confirmed by the candidate.

## Development

```
uv run ruff check .
uv run ruff format .
uv run pytest
```
