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

Evaluate a model on the labeled set (`data/postings.jsonl`); this prints accuracy, the
confusion matrix, the weighted error, the baselines, tokens, cost and latency, lists
every wrong decision with its rationale, and saves the run to `data/results/`:

```
uv run python -m job_fit.eval --model gemini-2.5-flash --reasoning on --allow-incomplete
uv run python -m job_fit.eval --model gemini-2.5-flash --reasoning on --version agent --allow-incomplete
```

Build the comparison tables from saved runs:

```
uv run python -m job_fit.compare data/results/<run>.json ...
uv run python -m job_fit.compare --table versions data/results/<run>.json ...
```

Live calls cost money (list prices are in the tables below). The tests make no network
calls: every LLM call is mocked, and a fixture refuses any socket connection.

## How decisions are scored

The test set has 21 real postings labeled by the candidate (APPLY 6, PREPARE 3,
SKIP 12). Mistakes are not equally expensive, so besides accuracy each run gets a
**weighted error** (lower is better):

| Actual \ Predicted | APPLY | PREPARE | SKIP |
|---|---|---|---|
| **APPLY** | 0 | 1 | **3** (missed a good job) |
| **PREPARE** | 1 | 0 | 2 |
| **SKIP** | 1 (wasted application) | 1 | 0 |

The baseline is a "model" that always answers `SKIP`, the most common label.

## Results

Full tables, caveats and reasoning: [docs/results.md](docs/results.md).

### Model comparison

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

### Workflow vs. agent

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

- 21 postings: one posting is about 5 points of accuracy, and repeating a run changes
  one posting or more. Differences of that size are noise.
- Only 3 `PREPARE` examples; nothing can be said about that class.
- The labels are one person's judgment.
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
  dataset.py    load and validate the labeled set
  metrics.py    accuracy, confusion matrix, weighted error, baselines
  pricing.py    dated list prices
  eval.py       run a model over the set, report, save
  compare.py    tables and Pareto front from saved runs
  ui.py         local Gradio page around the workflow and the agent
data/           postings.jsonl (labeled set), postings/, samples/, cv.example.md
tests/          pytest suite, no network
docs/results.md full results
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
