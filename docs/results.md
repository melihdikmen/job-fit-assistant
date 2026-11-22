# Results

Costs use list prices per token, even when credits pay.

> **Week 1 and Week 2.** The sections "Model comparison (Phase 3)" and "Workflow vs. agent
> (Phase 4)" were measured on the Week 1 set of 21 postings. In Week 2 the set grew to 40
> postings with labels confirmed by the candidate and was split into a dev and a held-out
> part. The current numbers are in "Week 2 reference (Phase 7)" at the end; the two parts
> are not comparable.

> **Note (2026-11-11, after all runs below).** A code review in Phase 5 changed the
> location rule in `rules.py`: names are matched as whole words, a location that names
> no place (`Fully remote`, `Not specified`, the country alone) is treated as unknown,
> and wording that excludes Türkiye (`Global, excluding Turkey`) is a violation. The
> runs below were made with the earlier rule and were not repeated. In those runs the
> location rule fired on one posting only (on-site in Bursa), and it still fires on it.
> Whether the new rule would fire anywhere else cannot be checked, because run files
> do not store the location the model extracted.

## Model comparison (Phase 3)

Runs of 2026-11-10 on the 21 labeled postings (APPLY 6, PREPARE 3, SKIP 12), same
dataset (SHA-256 `296805a932ba…`) and same prompt (SHA-256 `a1bff7049339…`) for every
row, Vertex AI location `global`, one call at a time. Weighted error is the mean error
cost per posting (lower is better; a missed good job costs 3, a wasted application 1).

The rows with numbers are generated from the run files with
`uv run python -m job_fit.compare <run files>`; the "not run" rows are added by hand.

| Model | Accuracy | Weighted error (mean) | Cost per 100 postings | Avg. latency | Notes |
|---|---|---|---|---|---|
| Baseline | 57.1% | 1.14 | $0.00 | 0.0 s | always SKIP |
| A, strong: gemini-4-carbon | 66.7% | 0.62 | $1.75 | 57.3 s | thinking cannot be switched off; promotional price, doubles on 2026-12-01 |
| B, mid: gemini-2.5-flash (reasoning off) | 52.4% | 0.48 | $0.16 | 2.6 s | |
| B again: gemini-2.5-flash (reasoning off), run 2 | 47.6% | 0.52 | $0.16 | 2.6 s | repeat, to measure noise |
| C, cheap: gemini-3.1-flash-lite | 52.4% | 0.48 | $0.10 | 1.7 s | Pareto front |
| X + reasoning: gemini-2.5-flash (reasoning on) | 71.4% | 0.43 | $0.64 | 11.9 s | Pareto front |
| Open-weight via OpenRouter (free) | not run | not run | 0 | – | stopped by the daily free-model quota after 8 of 21 postings; not repeated |
| Local (Ollama) | not run | not run | 0 (+ electricity) | – | does not fit this machine, see the estimate below |

No call failed and no answer needed a schema retry in any of the five runs.

### What the numbers say

- **Reasoning is the lever that mattered.** The same model, `gemini-2.5-flash`, goes from
  52.4% to 71.4% accuracy when thinking is switched on (11 → 15 of 21 correct). The price
  is 4 times the cost ($0.16 → $0.64 per 100 postings; output tokens 7.8k → 48.2k) and
  4.6 times the latency (2.6 s → 11.9 s).
- **Without thinking the models hedge.** Both non-thinking rows are below the baseline
  on accuracy (52.4% vs. 57.1%) but far ahead on weighted error (0.48 vs. 1.14): they
  answer PREPARE too often (6 and 4 SKIP postings rated PREPARE), which is wrong but
  cheap, and never rate an APPLY posting as SKIP.
- **The most expensive model was not the best.** `gemini-4-carbon` got every SKIP right
  (12 of 12) but is too strict on the other side: 2 APPLY and 2 PREPARE postings were
  rated SKIP, the costliest kind of mistake. It is dominated by `gemini-2.5-flash` with
  reasoning on, which is better on weighted error, 2.7 times cheaper and 4.8 times
  faster. One posting took 197 s.
- **Pareto front** (weighted error, cost, latency): `gemini-3.1-flash-lite` and
  `gemini-2.5-flash` with reasoning on. `gemini-2.5-flash` with reasoning off is
  dominated by Flash-Lite (same error, cheaper, faster).
- **Rules.** With reasoning off, the hard rules raised accuracy by about 2 postings
  (42.9% → 52.4% on both models). With reasoning on they changed no outcome.

### How much to trust the differences

- Repeating the reasoning-off run gave 10 instead of 11 correct and a total weighted
  error of 11 instead of 10; 3 of 21 predictions changed. The Phase 2 run of the
  reasoning-on configuration (in `us-central1`) had 14 correct and 0.48, against 15 and
  0.43 here. So **a difference of one posting or one error point is noise.**
- That makes the weighted-error gap between the two Pareto rows (0.43 vs. 0.48, one
  error point over 21 postings) a tie. The accuracy gap (15 vs. 11 correct) is larger
  than the noise seen, but rests on 21 postings.
- Three APPLY postings (Miny, A1 Mobile, Rubikpara) are rated lower by every model. That
  is either a shared blind spot of the prompt or a sign the labels deserve a second look.
- PREPARE has 3 examples; nothing can be said about that class.
- Prices are the Vertex AI list prices for the Global region, standard tier, read from
  the Vertex AI pricing page on 2026-11-11. `gemini-4-carbon` is at a promotional price
  through 2026-11-30 (2.00 / 10.00 per 1M tokens, given as 50% credits back); at its
  regular price (4.00 / 20.00) the row would cost $3.50 per 100 postings.
- Latency was measured once per posting from one home connection.

### Local model memory estimate

`parameters (billions) × bits ÷ 8 ≈ GB`, plus room for context.

| | 8B model, "4-bit" | 4B model, "4-bit" | 1.7B model, "4-bit" |
|---|---|---|---|
| Rule of thumb (4 bits) | 8 × 4 ÷ 8 = 4.0 GB | 2.0 GB | 0.85 GB |
| Typical Q4_K_M file (about 4.85 bits per weight) | about 4.9 GB | about 2.4 GB | about 1.0 GB |
| Plus context cache and runtime overhead (about 0.5–1 GB at 8k context) | about 5.5–6 GB | about 3–3.5 GB | about 1.5–2 GB |

The machine has a GTX 1050 with **3 GB VRAM** and **7.8 GB RAM** shared with Windows.
A 7–8B model, the size PROCESS.md asks for, does not fit in VRAM, and on the CPU it
would leave under 2 GB for everything else. A 4B model is borderline; only a model of
about 2B fits comfortably. These figures are estimates from the formula, not
measurements: **no local model was run** (decision D7 in PLAN.md).

A model of that size (`liquid/lfm-3-1.7b`, 1.7B) and a large open-weight model
(`nvidia/nemotron-3.5-super-vl`) were tried through free OpenRouter endpoints. Each
returned a schema-valid decision on a single posting, but the full runs were stopped by
OpenRouter's daily free-model quota after 8 postings, so there is no accuracy figure for
any open-weight model.

### Which model I chose and why

_Drafted by Claude from the numbers above and confirmed by the candidate on 2026-11-10._

I chose **`gemini-2.5-flash` with reasoning on**. It has the best accuracy (71.4%, 15 of
21, against 57.1% for the always-SKIP baseline) and the lowest weighted error (0.43
against 1.14), and it sits on the Pareto front. The other model on the front,
`gemini-3.1-flash-lite`, is 6 times cheaper and 7 times faster with a weighted error
that is equal within noise, but it gets 4 fewer postings right and rates half of the
SKIP postings as PREPARE, which would send me to prepare for jobs I should ignore. For
this tool the extra cost does not matter: I check a handful of postings a day, so
$0.0064 instead of $0.0010 per posting and 12 seconds instead of 2 are both irrelevant.
If the same decision had to be made for thousands of postings, or inside an interactive
page, Flash-Lite would be the better choice. The strongest model, `gemini-4-carbon`, is
ruled out: it costs more, takes about a minute per posting and makes more of the
expensive mistakes.

If privacy were required, meaning the CV may not leave my machine, none of the measured
models would qualify and I would need a local open-weight model. On this hardware that
means a model of about 2B parameters, which I could not measure; given that thinking was
what separated good from mediocre results here, I would expect such a model to do
clearly worse, and I would test it on this same set before trusting it. With a machine
with 8 GB of VRAM or more, a 7–8B model at 4-bit would be the first thing to try.

## Workflow vs. agent (Phase 4)

Same model and settings for both versions: `gemini-2.5-flash` on Vertex AI `global`,
reasoning on, temperature 0, the same 21 postings (dataset SHA-256 `296805a932ba…`), the
hard rules applied in code to the final answer. Only the harness differs:

- **Workflow:** one model call that sees the posting and the whole CV.
- **Agent:** LangChain `create_agent` with three tools (`check_rules`, `get_cv_section`,
  `convert_currency`) and at most 6 model calls per posting. It sees the posting and
  must read the CV through `get_cv_section`.

Each version was run twice, because Phase 3 showed that one run can be off by a posting.
Workflow run 1 is the Phase 3 run of 2026-11-10; the other three are from 2026-11-11.
The table is generated with `uv run python -m job_fit.compare --table versions <run files>`.

| Version | Accuracy | Weighted error (mean) | Cost per 100 postings | Avg. latency | Avg. model calls | Avg. tool calls |
|---|---|---|---|---|---|---|
| Workflow, run 1 | 71.4% | 0.43 | $0.64 | 11.9 s | 1.00 | 0.00 |
| Workflow, run 2 | 66.7% | 0.48 | $0.64 | 12.1 s | 1.00 | 0.00 |
| Agent, run 1 | 85.7% | 0.14 | $0.74 | 14.8 s | 3.67 | 3.71 |
| Agent, run 2 | 61.9% | 0.38 | $0.77 | 15.0 s | 3.71 | 3.71 |
| **Workflow, mean of 2** | 69.0% | 0.45 | $0.64 | 12.0 s | 1.00 | 0.00 |
| **Agent, mean of 2** | 73.8% | 0.26 | $0.75 | 14.9 s | 3.69 | 3.71 |

The two "mean" rows are computed by hand from the rows above. No call failed, no posting
hit the step limit and no answer needed a schema retry in any of the four runs.

### What the numbers say

- **Accuracy does not separate the versions.** The agent has both the best run (18 of
  21) and the worst (13 of 21); the workflow's runs (15 and 14) lie in between.
- **The agent is much less stable.** Its two runs disagree on 8 of 21 postings; the
  workflow's two runs disagree on 2. Same model, same temperature 0: a decision reached
  over several model calls has more places to drift.
- **The agent made no expensive mistake.** In neither run did it rate a good or
  preparable job as SKIP; the workflow did that twice in each run (5 of its 9 and 10
  error points). That is why the agent's weighted error is lower in both runs (0.14 and
  0.38 against 0.43 and 0.48). In run 2 the agent got there by hedging: it rated 5 SKIP
  postings as PREPARE, which is wrong but cheap in the cost matrix, the same pattern as
  the models without thinking in Phase 3.
- **Cost and latency rose less than the number of calls suggests:** about 18% more cost
  and 24% more time for 3.7 times the model calls. The agent sends 3.3 times the input
  tokens (the conversation is re-sent on every call) but writes slightly fewer output
  tokens (about 44k against 48k per run), and output tokens are what this model
  charges for.

### How the agent used its tools

Tool usage was identical in both runs, call for call:

| Tool | Calls per run | Postings | What it was used for |
|---|---|---|---|
| `check_rules` | 21 | 21 of 21 | Called first, before reading the CV, in 41 of 42 cases. |
| `get_cv_section` | 49 | 21 of 21 | `experience` 21, `skills` 21, `summary` 7. Never `projects`, `education` or `languages`. |
| `convert_currency` | 8 | 4 of 21 | Both ends of the USD salary range in the four Sezzle postings. |

- The agent did not explore. It ran the same three steps on every posting: check the
  rules, read experience and skills, answer. That is a fixed workflow which the model
  re-derives on every posting, at 3 to 6 model calls each (3 calls for 19 of the 42
  decisions, 4 for 18, 5 for 4, 6 for 1).
- It never read the Projects section, so it decided on less of the CV than the workflow,
  which always sees all of it.
- `check_rules` changed nothing. Its answer matched the rules applied in code for all 42
  decisions, and the code would have applied them anyway. The rules changed one outcome
  in agent run 1 and none in run 2.
- `convert_currency` was called for every posting with a USD salary, but its result has
  nowhere to go: `JobDecision` has no salary field and salary is not part of the
  decision. It uses a fixed rate (1 USD = 50.0 TRY, 1 EUR = 56.1 TRY, approximate
  mid-market rates from xe.com on 2026-11-04), not a live one.

### How much to trust the differences

- Two runs per version on 21 postings. The agent's own runs differ by 5 correct
  postings, more than the gap between the versions' means (73.8% vs. 69.0%, one
  posting). The accuracy difference is noise.
- The weighted-error difference points the same way in both runs, but the two versions
  also differ in prompt (the agent's instructions mention tools and it reads part of the
  CV), so the lower error cannot be credited to tool calling as such.
- The earlier caveats still hold: 3 PREPARE postings, labels accepted from a suggested
  list, latency from one home connection.

### Is an agent needed for this problem?

_Drafted by Claude from the numbers above and confirmed by the candidate on 2026-11-11._

No. An agent earns its cost when the steps cannot be known in advance: when what to look
up next depends on what the last lookup returned. This task is not like that. Every
posting needs the same inputs (the posting, the CV, the rules), and the agent showed it:
in 42 decisions it made the same tool calls in the same order, so the "decisions" it was
given the freedom to make were never really open. A fixed sequence written in code does
the same work in one model call.

What the agent added was cost and instability, not accuracy. It needed 3.7 model calls
per posting, cost 18% more and took 24% longer, and its two runs disagreed on 8 of 21
postings against 2 for the workflow. For a tool whose answer I act on, a decision that
changes when I ask again is worse than one that is consistently a little too strict. Of
its three tools, `check_rules` duplicated what the code enforces anyway,
`convert_currency` produced numbers the decision never uses, and `get_cv_section` only
made the agent see less of a CV that fits in the prompt (under 800 words).

The one result in the agent's favor is that it never skipped a good job, which gave it
the lower weighted error in both runs. That is worth following up, but in the workflow:
the likely cause is the different prompt and the more lenient reading it produced, not
tool calling, and a prompt change is cheaper to test and to keep than an agent loop.

An agent would become the right choice if the task grew steps that depend on each
other: looking up the company, fetching a posting from a URL, comparing against earlier
applications, or asking me a question when the posting is unclear. Until then the fixed
workflow stays the version I use.

## Week 2 reference (Phase 7)

Runs of 2026-11-22 on the **dev set**: 27 postings (APPLY 6, PREPARE 4, SKIP 17), dataset
SHA-256 `02f2239aa4ad…`, Vertex AI location `global`, one call at a time. Workflow prompt
SHA-256 `5401d5edd350…`, agent prompt SHA-256 `e71ed0536d15…` (both hashes now cover the
schema field descriptions), CV SHA-256 `aab1b80bd000…`. Each configuration was run three
times. No call failed and no answer needed a schema retry in any of the 12 runs.

These numbers are not comparable with the Week 1 sections above: the set, the labels,
the location rule and the hash definition all changed.

### The set and its labels

- 40 postings: the 21 of Week 1 plus 19 new ones. APPLY 9, PREPARE 6, SKIP 25.
- **How the labels were made.** Every label started as a suggestion written by Claude.
  The candidate went through all 40 postings with the suggestion on screen and
  confirmed or changed each one. 5 of 40 were changed, all among the new postings (four
  PREPARE became SKIP, one SKIP became PREPARE); Cohen's kappa between suggested and
  confirmed labels is 0.78. The labels are the candidate's decision, but they are not
  independent of the suggestions. PROCESS.md asked for labels made without them.
- Split by a seeded shuffle within each label, not by hand: dev 27, held-out 13
  (APPLY 3, PREPARE 2, SKIP 8). **The held-out set has not been run.** It is evaluated
  once, at the end of Phase 11.
- Posting texts are condensed (370 to 3,500 characters), as in Week 1.

### Reference table

Generated with `uv run python -m job_fit.compare --table reference <run files>`. Values
are the mean ± the sample standard deviation over the three runs.

| Configuration | Runs | Accuracy | Weighted error (mean) | APPLY recall | Kappa | Cost per 100 postings | Avg. latency | Judge score |
|---|---|---|---|---|---|---|---|---|
| Baseline (always SKIP) | – | 63.0% | 0.96 | 0.0% | 0.00 | $0.00 | 0.0 s | – |
| gemini-2.5-flash (reasoning on) | 3 | 65.4% ± 2.1% | 0.51 ± 0.06 | 44.4% ± 9.6% | 0.37 ± 0.04 | $0.62 | 11.4 s | 4.63 |
| gemini-2.5-flash (reasoning off) | 3 | 60.5% ± 2.1% | 0.40 ± 0.02 | 44.4% ± 9.6% | 0.38 ± 0.02 | $0.16 | 2.7 s | 4.48 |
| gemini-3.1-flash-lite | 3 | 55.6% ± 0.0% | 0.44 ± 0.00 | 50.0% ± 0.0% | 0.32 ± 0.00 | $0.10 | 1.9 s | 4.11 |
| gemini-2.5-flash (agent, reasoning on) | 3 | 67.9% ± 5.7% | 0.35 ± 0.08 | 72.2% ± 9.6% | 0.48 ± 0.09 | $0.74 | 14.3 s | 4.41 |

The judge score is from `gemini-3.1-pro-preview` on run 3 of each configuration; see
"Judge" below before reading anything into it.

Range of the mean weighted error over the three runs: workflow with reasoning on 0.44
to 0.56, reasoning off 0.37 to 0.41, Flash-Lite 0.44 in every run, agent 0.26 to 0.41.

**Noise figure: 0.077.** That is the largest standard deviation of the mean weighted
error among the configurations (the agent's). A difference between two configurations
that is smaller than this is not evidence of an improvement. Phase 11 uses it as its
keep-or-roll-back threshold.

### Precision and recall per class

Means over the three runs. Recall: of the postings that deserve a decision, the share
that got it. Precision: of the postings given a decision, the share that deserved it.

| Configuration | APPLY recall / precision | PREPARE recall / precision | SKIP recall / precision |
|---|---|---|---|
| Workflow, reasoning on | 44% / 47% | 42% / 31% | 78% / 83% |
| Workflow, reasoning off | 44% / 62% | 58% / 20% | 67% / 100% |
| Workflow, Flash-Lite | 50% / 43% | 50% / 20% | 59% / 100% |
| Agent, reasoning on | 72% / 65% | 58% / 30% | 69% / 95% |

PREPARE has 4 postings on dev, so its recall moves in steps of 25 points; its row says
little. APPLY has 6, one posting is 17 points.

### What the numbers say

- **Accuracy barely separates anything from the baseline; kappa does.** "Always SKIP"
  is right on 63.0% of this set. The workflow with reasoning on reaches 65.4%, about
  half a posting more, and the two cheap configurations are below the baseline. By
  accuracy the model looks useless. Kappa removes what the label mix gives for free:
  the baseline scores exactly 0, and every configuration is between 0.32 and 0.48. The
  workflow with reasoning off has lower accuracy than the baseline (60.5%) and the same
  kappa as the workflow with reasoning on (0.38 and 0.37).
- **Only one configuration makes the expensive mistake.** The workflow with reasoning
  on rated an APPLY posting as SKIP 5 times in three runs and a PREPARE posting as
  SKIP 3 times. The agent did the second twice and the first never; the two cheap
  configurations did neither. That is the whole reason the workflow with reasoning on
  has the worst weighted error.
- **The agent has the best weighted error and the best APPLY recall.** Its range of
  weighted error (0.26 to 0.41) does not overlap with that of the workflow with
  reasoning on (0.44 to 0.56), and the difference of the means, 0.16, is twice the
  noise figure. Against the workflow with reasoning off (0.40) the agent's lead is
  0.05, inside the noise: a tie.
- **The cheap configurations hedge.** With reasoning off, 17 of the 22 errors in three
  runs are SKIP postings rated PREPARE, and PREPARE precision is 20%: four of five
  "prepare" answers are for jobs to ignore. That is cheap in the cost matrix and a
  nuisance in use.
- **The agent is the least stable.** 8 of 27 predictions changed between its runs,
  against 5 and 6 for the workflow with reasoning on and off and 0 for Flash-Lite,
  whose three runs are identical. The agent's accuracy ranged from 63.0% to 74.1%.
- **Rules.** The location rule fired on one posting per run and the seniority rule on
  two. They changed 2 outcomes in three runs with reasoning on, 6 with reasoning off,
  5 on Flash-Lite and none for the agent.
- **The agent still does not explore.** `check_rules` once per posting (27 of 27 in
  every run, its answer equal to the rules applied in code every time);
  `get_cv_section` 59 to 62 times per run, for `experience` and `skills` on every
  posting, `summary` on 17 of 81 and `education` on 3, never `projects`;
  `convert_currency` 4 to 6 times; 3.6 model calls per posting.
- **Three postings are wrong in every run of every configuration:** Miny (label APPLY)
  and Fuzul mobile architecture and Turkcell senior mobile (label SKIP). Either the
  prompt has a blind spot all four share, or these labels are the debatable ones.

### Does the cost matrix decide the ranking?

Generated with `uv run python -m job_fit.compare --table costs <run files>`. Mean
weighted error with the cost of a missed good job (APPLY rated SKIP) set to 3, 2 and 5;
the predictions are the same, only their price changes.

| Configuration | missed APPLY = 3 (default) | missed APPLY = 2 | missed APPLY = 5 |
|---|---|---|---|
| gemini-2.5-flash (reasoning on) | 0.51 (rank 4) | 0.44 (rank 3) | 0.63 (rank 4) |
| gemini-2.5-flash (reasoning off) | 0.40 (rank 2) | 0.40 (rank 2) | 0.40 (rank 2) |
| gemini-3.1-flash-lite | 0.44 (rank 3) | 0.44 (rank 3) | 0.44 (rank 3) |
| gemini-2.5-flash (agent, reasoning on) | 0.35 (rank 1) | 0.35 (rank 1) | 0.35 (rank 1) |

No. The best configuration is the same under all three matrices. Three of the four
rows do not move at all, because those configurations never miss an APPLY; only the
workflow with reasoning on does, and it is last or tied for third whatever a miss costs.

### Judge

A second model, `gemini-3.1-pro-preview`, rated the rationale of every decision in run
3 of each configuration (108 rationales, none failed) against a written rubric: names
matching skills, names missing skills, reflects the hard rules that fired, mentions
the English risk, consistent with the decision, and an overall score from 1 to 5. It
saw the posting, the CV, both decisions and the fired rules, but not the label.

| Configuration | Mean score | Matching skills | Missing skills | Hard rules | English risk | Consistent |
|---|---|---|---|---|---|---|
| Workflow, reasoning on | 4.63 | 100% | 93% | 100% | 93% | 100% |
| Workflow, reasoning off | 4.48 | 96% | 85% | 100% | 96% | 96% |
| Workflow, Flash-Lite | 4.11 | 96% | 67% | 100% | 93% | 100% |
| Agent, reasoning on | 4.41 | 96% | 96% | 100% | 96% | 100% |

**The judge was checked against the candidate and did not pass.** The candidate scored
10 rationales of the workflow with reasoning on, by the same rubric, before seeing the
judge's scores:

| | Result |
|---|---|
| Same score | 4 of 10 |
| Within one point | 7 of 10 |
| Mean absolute difference | 1.2 points |
| Judge minus candidate, mean | +0.6 points |

Three pairs are three points apart (candidate 2, judge 5, twice; candidate 5, judge 2,
once). The judge is also lenient: it gave 5 to 21 of 27 rationales of that run. So the
judge score is **reported, not trusted**: it is not used to rank configurations here
and must not become a keep-or-roll-back criterion in Phase 11 unless the rubric or the
judge is changed and the check repeated. Ten rationales are too few to say why the two
disagree. The "hard rules" criterion is 100% everywhere by construction: the rule note
is appended by code, so that line of the rubric measures the program, not the model.

`gemini-4-carbon` was the planned judge. On the evening of the runs it answered
"resource exhausted" for nearly every call, so the judge was changed (decision D47 in
PLAN.md).

### Observability

Every model call of the 12 runs and of the judge pass was traced to Langfuse (cloud, EU
region) with prompt, output, tokens, latency, cost and model, tagged by experiment
(`week2-ref-…`), version and run number. The CV is replaced by a hash marker in every
trace; this was checked on a stored trace, which held none of the CV's 77 maskable
lines. Posting texts and the model's rationales are traced as they are.

### How much to trust this

- 27 postings and three runs per configuration. The standard deviations are computed
  from three values; they show the order of magnitude of the noise, not its exact size.
- The labels were confirmed with the suggested label on screen (see above).
- PREPARE: 4 postings on dev. Nothing here supports a statement about that class.
- The agent and the workflow differ in prompt and in how much of the CV they read, as
  in Week 1; the agent's better result cannot be credited to tool calling as such.
- Prices are Vertex AI list prices, Global, standard tier, read on 2026-11-22. Spend at
  list price: about $1.30 for the 12 runs and $1.95 for the judge, trials included.
- Latency was measured once per posting from one home connection.

### What this means for the Week 1 conclusions

_Drafted by Claude from the numbers above and confirmed by the candidate on 2026-11-22._

The Week 1 choice, the fixed workflow on `gemini-2.5-flash` with reasoning on, does not
hold up on this set. It has the worst weighted error of the four configurations, by
more than the noise, and it is the only one that throws away good jobs, which is the
mistake the cost matrix was built to punish. Week 1 said "no agent" because the agent
added cost and instability without accuracy. On the larger set the agent is still the
least stable and still the most expensive, but it finds 72% of the APPLY postings
against 44% and never rated one of them SKIP in three runs.

That is not yet a reason to switch. Week 1 already suspected that the agent's
advantage comes from its prompt and its narrower view of the CV, not from tool
calling, and nothing measured here separates those. The workflow with reasoning off
gets within noise of the agent's weighted error at a fifth of the price and a fifth of
the latency, by hedging into PREPARE. So the open question for the next phases is
narrow: why does the workflow with reasoning on talk itself into SKIP, and can a change
to it (Phase 8: fewer decisions left to the model; Phase 11: one guarded prompt change
at a time) remove that mistake without the agent loop? The held-out set stays closed
until that work is done.
