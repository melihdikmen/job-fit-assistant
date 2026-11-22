# Architecture

_Drafted by Claude from the measurements in [results.md](results.md) and confirmed by the
candidate on 2026-11-22._

## The business problem

One candidate reads many job postings. Reading one properly takes minutes, and most are
not worth it. The tool reads the posting and the CV and recommends `APPLY`, `PREPARE` or
`SKIP` with a reason; it never applies. The mistakes do not cost the same: a good job
thrown away costs 3, a preparable one thrown away 2, every other mistake 1. So the
system is judged on weighted error and on APPLY recall, not on accuracy.

## The baseline

"Always SKIP", the most common label: 63.0% accuracy, weighted error 0.96, APPLY recall
0%, kappa 0.00 on the dev set (27 postings). Accuracy is nearly free here, so a model
call has to earn its place on weighted error and kappa.

## Why each LLM call exists

| Call | Why a model | What code does around it | Status |
|---|---|---|---|
| **Decide** (`workflow.py`, one call) | A posting is free text in two languages; weighing a skill gap against a CV cannot be written as rules. | Validates the JSON, retries once, applies the hard rules, raises the English flag. | In use. Weighted error 0.51 with reasoning on, 0.40 off. |
| **Agent loop** (`agent.py`, 3.6 calls) | No reason of its own: it does what the one call does, in more steps. | Same rules, applied after it. | Comparison only. |
| **Extract** (`extract.py`, 1 extra call) | Reading facts needs no judgment and no CV, so a cheaper model can do it. | Joins facts and judgment, same rules. | Tried in Phase 8, dropped (below). |
| **Judge** (`judge.py`) | Rates rationales against a rubric. | Compared with the candidate's own scores. | Not in the product. Failed its check: 4 of 10 equal. |

## What is in code, and what is left to the model

In `rules.py`, with the last word: location, seniority, the English flag (the model's
reading or a keyword match) and the currency conversion. The agent's `check_rules` tool
calls the same functions. Left to the model on purpose: reading the facts the rules
run on, and the line between APPLY, PREPARE and SKIP (decision D50). That line is where
the costly mistake is made: in all 8 cases where the workflow threw a good job away, no
rule fired, and the rationale filed one partly covered requirement under "the core
stack does not match".

## The three traps and the Week 1 agent

- **"The agent for X".** The agent was built because the phase was called "agent
  version", not because the problem has open steps. In 123 decisions (42 in Week 1, 81
  in Week 2) it called `check_rules` once and read `experience` and `skills` every
  time: a fixed workflow, written as prose in its prompt and re-derived on every posting.
- **Human-like agent.** `get_cv_section` makes the model leaf through the CV the way a
  recruiter would. The CV is under 800 words and fits in the prompt; the tool only made
  the agent decide on less of it (it never read Projects).
- **Plausible-looking output.** `check_rules` looks like diligence and changed nothing:
  its answer equalled the rules in code on every posting. `convert_currency` produced
  numbers no field or rule uses. Rationales read well (the judge gave 4.1 to 4.6 of 5)
  while a third or more of the decisions were wrong.

## The sub-agent experiment

Extractor on `gemini-3.1-flash-lite`, then the main model, against the workflow with
the same reasoning setting, three runs each, rule fixed in advance (keep only if
weighted error improves by more than the noise figure 0.077, or cost falls by 10%,
without losing APPLY recall).

| Pair | Weighted error | APPLY recall | Cost per 100 | Latency |
|---|---|---|---|---|
| Reasoning on | 0.51 → 0.48 | 44.4% → 38.9% | $0.62 → $0.58 | 11.4 → 12.3 s |
| Reasoning off | 0.40 → 0.37 | 44.4% → 50.0% | $0.156 → $0.154 | 2.7 → 4.0 s |

**Dropped.** The change is a third of the noise. The rules fired on the same three
postings in every run of both versions: the facts were already right, so a second
model to read them fixes nothing.
