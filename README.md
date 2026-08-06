# insight-consolidation-v1

[![ci](https://github.com/Pavel-Tk/insight-consolidation-v1/actions/workflows/ci.yml/badge.svg)](https://github.com/Pavel-Tk/insight-consolidation-v1/actions/workflows/ci.yml)

**Can an agent recover what is actually driving a customer, from sparse evidence buried in volume?**

An RL environment for [Prime Intellect's Environments Hub](https://app.primeintellect.ai/dashboard/environments), built on `verifiers` v1.

Each episode is one customer account: between 50 and 5,000 ordinary business documents - emails, meeting notes, support tickets, CRM entries, call summaries. Somewhere in there, four or five unremarkable traces jointly entail why the account is really behaving the way it is. No single document says it. Most of the corpus is irrelevant. Some of it argues convincingly for the obvious wrong answer.

The agent must name the driver **and** cite the specific documents that entail it.

---

## Why this task

The insight tier of agent memory is where everything currently fails, and it fails by a wide margin.

- **KnowMe-Bench** ([arXiv:2601.04745](https://arxiv.org/abs/2601.04745)) splits into memory, reasoning and insight. Memory tasks run 59-79%. Insight tasks cap at **22.6%**, and adding a memory system barely moves them - Mem0 lifts extraction from 65.4 to 73.2 while *lowering* temporal reasoning from 54.1 to 51.6.
- **PersonaMem-v2** ([arXiv:2512.06688](https://arxiv.org/html/2512.06688)) finds frontier models at 37-48% on implicit preference inference, dropping to **33.0%** when the evidence contradicts the demographic prior, and **17.5%** on knowing whose preference a stored fact is.

Retrieval keeps improving and insight does not follow. A benchmark sitting at ~22% with a clear mechanism for why is a good place to put an RL environment; a benchmark everyone scores 90% on teaches a model nothing.

The Environments Hub already has memory environments - `vedant/supersede` tests using the current fact rather than the stale one across long multi-session interactions, `dennwslee/taulong_bench` bounds cross-case memory over sequential support tasks, and there is a restart-aware filesystem memory environment. They test *holding* information: supersession, bounded context, retrieval under pressure.

What none of them ask for is a claim about intent that no document states, backed by the specific documents that entail it. That gap is the one this environment fills, and it is a narrow gap rather than an empty field.

## How it stays honest

The field's credibility problem is judges and self-reported numbers. Zep's LoCoMo score was corrected from 84 to 75.14 after an arithmetic error. EverMemOS's 92.32 reproduced at 38.38. MemPalace's "100%" traced back to three hardcoded patches. GPT-4o-mini judges accept roughly 63% of intentionally wrong-but-topical answers.

Five design decisions follow from that.

**1. Construct-then-hide.** The latent driver is sampled *first*, as a structured object. Evidence is then manufactured to entail it only in aggregate. Ground truth is a variable we chose, never a human annotation of prose, and never an artifact of phrasing.

**2. The citation gate.** The agent's free-text driver is scored by a judge - but that judge holds the answer key, and its score is multiplied by a gate that ramps with citation F1 against the minimal entailing set. A correct-sounding driver with no supporting evidence keeps 10% of its score.

Against a 4-document entailing set, with `gate_floor=0.1` and `gate_full=0.5`:

| cited | precision | recall | F1 | gate |
|---|---|---|---|---|
| all 4 entailing docs | 1.00 | 1.00 | 1.00 | **1.00** |
| 2 of 4 | 1.00 | 0.50 | 0.67 | **1.00** |
| 1 of 4 | 1.00 | 0.25 | 0.40 | **0.82** |
| 1 right, 1 wrong | 0.50 | 0.25 | 0.33 | **0.70** |
| 20 docs incl. all 4 | 0.20 | 1.00 | 0.33 | **0.70** |
| all 50 docs (volume 50) | 0.08 | 1.00 | 0.15 | **0.37** |
| none, or none correct | 0.00 | 0.00 | 0.00 | **0.10** |

Read the shotgun rows honestly, because they are the interesting ones. **The gate does not stop over-citing.** Padding to 20 documents to be sure of covering the answer still keeps 70% of the judged score - the same as one right and one wrong citation. Only indiscriminate citation at scale is bitten hard, and that is recall saturating against collapsing precision, not the gate doing something clever.

What the gate does stop is the failure it was built for: asserting a driver you cannot evidence at all. That path floors at 0.10. And guessing the stereotype does not pay either, but that is the judge's doing rather than the gate's - calibration below scores the decoy at 0.025.

Over-citing is priced by `evidence_grounding` instead, which is raw F1 at weight 0.5. On the total reward, a correct driver cited precisely earns 1.75; the same driver shotgunned across 20 documents earns 1.12. A real cost, and a smaller one than the gate is often assumed to impose. Sharpening it is [issue #7](https://github.com/Pavel-Tk/insight-consolidation-v1/issues/7), not a silent change.

**3. Model-free baselines ship in the repo.** Letta scored 74.0% on LoCoMo with nothing but a filesystem and `grep`, which means most published memory architectures beat a trivial baseline by single digits. Run `python -m insight_consolidation_v1.baselines` before believing any model number:

```
  grep (top hits)        P=0.051  R=0.138  F1=0.072   (n=49)
  grep (budget 4)        P=0.041  R=0.041  F1=0.041   (n=49)
  random 4 docs          P=0.031  R=0.031  F1=0.031   (n=49)
  cite everything        P=0.031  R=1.000  F1=0.057   (n=49)
```

Lexical retrieval recovers about 14% of the entailing evidence at poor precision - better than random, nowhere near solving it. If `grep (top hits)` recall ever approaches 1.0, the episodes need regenerating.

**4. The judge is calibrated before it is trusted, and the calibration ships.** KnowMe-Bench reports Cohen's kappa against human labels. This environment has no human labels yet, so it measures construct validity instead: the judge holds the answer key, so we know what it *should* say about four kinds of answer. `python judge_check.py` grades each schema's own description, its one-line label, the stereotype the anti-prior split is built from, and a generic claim that fits any struggling account.

```
  reference   mean=1.000   (the latent driver's own description, verbatim)
  label       mean=0.889   (the same driver, compressed to one line)
  stereotype  mean=0.025   (the decoy - the answer it was warned about)
  generic     mean=0.275   (plausible, fits any account)

  separation (label - stereotype) = +0.864
```

Run this before quoting any judged number. It is forty calls and it is the difference between a reward and a random variable - the first time it ran here it returned **-0.050**, scoring the decoy *above* the verbatim correct answer. The judge was fine; the harness was reading the grade off the first digit in the model's reasoning scratchpad. A benchmark can be wrong in a direction that flatters nobody, and this one was, until something measured it.

**5. One world per schema, shared by signal and noise alike.** Every schema fixes its own setting: the product being sold, the customer's industry, the named internal teams, the specific thing at stake. Claims Engineering built Project Tern and it cannot read handwritten loss-run supplements. Marketing Technology and Data Platform Engineering are fighting over customer identity resolution. The PCI DSS 4.0 assessment is in the second week of November.

Ambient noise and prior-decoys draw from that same world. An earlier draft let noise pick team names from a generic pool, which meant signal documents were the only ones naming the real teams - separable by vocabulary alone. Closing that leak is visible above: grep F1 fell from 0.119 to 0.072.

## The rubric

**Trained**

| reward | weight | verification |
|---|---|---|
| `grounded_insight` | 1.0 | judged semantic match against the latent driver, gated by citation F1 |
| `evidence_grounding` | 0.5 | F1 of cited ids against the minimal entailing set - set arithmetic |
| `confabulation` | 0.25 | abstained correctly on a null episode - deterministic |

**Diagnostics** (weight 0, recorded not trained): `insight_ungated` so the gate's effect is visible rather than assumed, `anti_prior`, `revision`, `volume`, `docs_read`, `citation_precision`, `citation_recall`, `answer_parsed`.

Only one dimension calls a model, and it calls it as a semantic-equivalence check against a fixed reference rather than an open-ended quality grade. A judge asked "does this mean the same thing as this?" is a far narrower instrument than a judge asked "how insightful is this?".

## First run

MiniMax-M2, n=30, `max_turns=60`, volumes 50 and 500, judged by MiniMax-M2. Per-episode rows
are in [`results.jsonl`](results.jsonl) - all thirty, nothing dropped, nothing filtered.

```
grounded_insight     0.418  (sd 0.387)     <- the headline
insight_ungated      0.508  (sd 0.402)     <- same judgement, gate removed
evidence_grounding   0.357  (sd 0.311)
confabulation        0.800  (sd 0.407)
citation_precision   0.310  (sd 0.290)
citation_recall      0.450  (sd 0.368)
answer_parsed        0.867  (sd 0.346)
docs_read             29.8
turns                 44.4
```

| variant | n | grounded | ungated | grounding | docs read |
|---|---|---|---|---|---|
| `revision` | 5 | 0.724 | 0.850 | 0.563 | 23.2 |
| `anti_prior` | 8 | 0.474 | 0.531 | 0.449 | 37.2 |
| `normal` | 11 | 0.465 | 0.614 | 0.392 | 31.3 |
| `null` | 6 | **0.000** | 0.000 | 0.000 | 22.8 |

| volume | n | grounded | grounding | docs read |
|---|---|---|---|---|
| 50 | 20 | 0.449 | 0.418 | 37.0 |
| 500 | 10 | 0.356 | 0.236 | 15.4 |

**The null row is the result worth reading.** `confabulation` measures whether the agent
abstained exactly when it should have. It scores 0.800, which sounds respectable and is
entirely an artifact of composition: **the model abstained zero times in thirty episodes.**
It scored 23/24 on "correctly did not abstain" and 0/6 on "correctly abstained", and 24/30 is
0.800. Shown six accounts where nothing coherent is entailed, it invented a driver six times
out of six.

Drop the null episodes and the headline reads 0.522 instead of 0.418. That 0.104 is the exact
size of the flattery a benchmark without a confabulation control hands out for free, measured
rather than asserted, and it is why the null variant exists.

Two more things the run says, both preliminary at this n:

- **Volume hurts grounding more than insight.** At 500 documents the agent reads 15 of them
  instead of 37 and citation grounding falls from 0.418 to 0.236, while `insight_ungated`
  does not fall at all (0.487 to 0.550). It keeps forming opinions at the same rate while its
  ability to evidence them degrades. That is the failure mode the citation gate was built to
  price, and it is visible in one run.
- **`anti_prior` costs less than expected** - 0.474 against `normal`'s 0.465, essentially no
  difference. Either tripling the decoys is not enough pressure, or the judge's warning about
  the common wrong answer is doing the work for the agent. Worth investigating before anyone
  cites the anti-prior split as evidence of anything.

`answer_parsed` at 0.867 means roughly one episode in eight never produced a usable answer
contract. `forced_commit` at 0.133 means four episodes ran out of turns and had to be asked
for a final answer.

Caveats, stated rather than buried. n=30 is small and the standard deviations are large
relative to the means - treat every number here as an order of magnitude, not a measurement.
Volume 5000 was not run. Sampling temperature is not pinned ([issue #8](https://github.com/Pavel-Tk/insight-consolidation-v1/issues/8)), so this is not
bit-reproducible. And the agent grades itself: judge calibration above says MiniMax-M2 is a
competent grader of this task, but a second family would be better and is one flag away.

## Episode variants

| variant | what it tests |
|---|---|
| `normal` | consolidation of sparse signals |
| `anti_prior` | prior-supporting decoys are tripled, so the stereotype answer is maximally attractive while the evidence still entails the truth |
| `revision` | the driver holds for two thirds of the timeline, then late evidence overturns it. Correct answer is the new driver |
| `null` | nothing is entailed. Correct behaviour is to abstain. Without this, a confident guesser looks competent |

Volumes sweep 50 / 500 / 5,000 documents. `volume` is recorded per episode, so the sparsity curve comes out of one run by bucketing rather than three separate evals. At 5,000 the agent cannot read everything inside the turn budget and must decide what to look at - which is the actual skill, and the reason this is an RL environment rather than a QA set.

## Install and run

```bash
uv pip install -e .
python -m pytest tests -q                      # no keys needed
python -m insight_consolidation_v1.baselines   # no keys needed
```

This is a `verifiers` **v1** taskset, so the entrypoints are `eval` and `validate`, not the
v0 `vf-eval`:

```bash
uv run validate                                        # conformance check
uv run eval insight-consolidation-v1 --num_tasks 20    # needs a judge model key
prime eval run <owner>/insight-consolidation-v1        # same thing, from the Hub
```

`run_eval.py` is a reference harness that needs one chat API and the standard library. It
drives the same evidence store through a text command loop (`SEARCH:` / `READ:` / `INDEX:`)
and scores with the rubric imported from the package rather than reimplemented, so its
numbers mean the same thing.

```bash
export MINIMAX_TOKEN_PLAN_API=sk-...
python run_eval.py --n 30                          # OpenAI dialect, MiniMax-M2
python run_eval.py --n 30 --base https://api.openai.com --key-var OPENAI_API_KEY --model gpt-5.4-mini
```

It writes `results.jsonl` - one row per episode, committed to this repo for the run reported
below - and prints a breakdown by variant and by volume with standard deviations.

Two things it does that are worth copying if you write your own harness. It grades with a
separate client when you ask it to, because one model grading itself is not a measurement:

```bash
python run_eval.py --n 30 \
    --judge-dialect anthropic --judge-base https://api.anthropic.com \
    --judge-model claude-sonnet-5 --judge-key-var ANTHROPIC_API_KEY
```

And it **drops** episodes whose API calls fail rather than scoring them zero. An empty reply
is indistinguishable from a model that declined to answer, so a harness that swallows a 429
reports an outage as a capability result. The first run of this environment did exactly that
until it was fixed; the run below reports how many episodes were dropped, and it is zero.

For training, use the infinite variant and bound it with `--num_tasks`:

```python
from insight_consolidation_v1 import InsightProceduralTaskset
```

**Platform note.** `verifiers.v1` imports `fcntl`, so `uv run eval` and `uv run validate` are
Linux and macOS only. Everything that does not touch the rollout stack - the generator, the
evidence store, the baselines, the tests and `run_eval.py` - is pure standard library and
runs on Windows too. That is why `insight_consolidation_v1.store` exists separately from
`servers/tool.py`.

## Backends

The evidence store rebuilds each corpus deterministically from its spec, so the environment is self-contained and anyone can run it - no vendor backend, no API key beyond the judge model.

**Not yet implemented:** a pluggable backend so the same episodes can be served by a real memory system rather than the built-in store. `EvidenceToolsetConfig.substrate_base_url` is a placeholder and nothing reads it. Planned for v0.2 against the [Hermes memory API](https://github.com/Substrate-memory/hermes-substrate-wiki), which would let the environment compare memory architectures on identical ground truth. Treat the current numbers as measuring agents, not memory systems.

I build [Substrate](https://trysubstrate.co), an agent memory system, which is exactly why this environment does not depend on it and why the backend hook is documented as absent rather than implied. A benchmark for a category, published by a vendor in that category, is worth nothing unless it runs and scores identically without the vendor's product.

## Reproducibility

An episode is fully determined by `(seed, schema_key, volume, variant)`, so the corpus never travels on the wire and a 5,000-document episode is a few hundred bytes of task data. `tests/test_generator.py` asserts reproducibility **across processes** under varying `PYTHONHASHSEED`, not just within one - a PRNG seeded from Python's builtin `hash()` reproduces perfectly inside a single run and silently differs between runs, which would quietly void every published score. That bug was in the first draft and the test is what caught it.

## Editing the schemas

`insight_consolidation_v1/schemas.py` is the domain layer and the highest-leverage file in the repo. Ten latent driver schemas drawn from enterprise buying psychology - blame risk mistaken for price sensitivity, internal ownership fights fought through vendor evaluations, pilots that are really free consulting, accounts whose adoption is one team deep.

The honest risk with this design is that mechanical schemas turn the benchmark into a logic puzzle wearing a psychology costume, measuring pattern matching rather than understanding. Four rules in that file guard against it, and `test_no_single_document_names_the_driver` enforces the most important one automatically: if any signal leaks the driver, the task is retrieval and the test fails.

## Status

v0.1.0. Early. The schemas are a first draft and will change as they meet real models.

Known limitations are [filed as issues](https://github.com/Pavel-Tk/insight-consolidation-v1/issues) rather than left to be discovered: ten schemas is thin, the answer space is effectively closed, there is no pluggable backend, the judge has construct validity but no human-agreement number, volume 5000 is unrun, and sampling temperature is not pinned.

Three defects were found in this environment's own scoring path by measuring it rather than reading it, and all three flattered the design:

- the judge's grade was being read off the first digit in the model's reasoning scratchpad, which made every judged score noise
- budget-exhausted episodes were graded on a half-finished investigation, because the forced-commit fallback could never fire
- the citation gate table claimed shotgunning scored 0.10 when it scores 0.70

They are described where they happened rather than quietly fixed, because a benchmark that has never caught itself being wrong has not been checked.

Built by Pavel Tkachyk. MIT.

## References

KnowMe-Bench [arXiv:2601.04745](https://arxiv.org/abs/2601.04745) · PersonaMem-v2 [arXiv:2512.06688](https://arxiv.org/html/2512.06688) · LongMemEval-V2 [arXiv:2605.12493](https://arxiv.org/html/2605.12493v1) · MemoryArena [arXiv:2602.16313](https://arxiv.org/html/2602.16313v1) · MemGym [arXiv:2605.20833](https://arxiv.org/abs/2605.20833) · Memory-R1 [arXiv:2508.19828](https://arxiv.org/abs/2508.19828) · [verifiers v1](https://www.primeintellect.ai/blog/verifiers-v1) · ["The Benchmark Theatre"](https://essays.bloo-mind.ai/posts/2026-05-20-mem-eval/)
