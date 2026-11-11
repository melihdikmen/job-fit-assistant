# Results

Costs use list prices per token, even when credits pay.

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

| Version | Accuracy | Weighted error | Cost | Latency | Avg. tool calls |
|---|---|---|---|---|---|
| Workflow | | | | | 0 |
| Agent | | | | | |

### Is an agent needed for this problem?

_To be written._
