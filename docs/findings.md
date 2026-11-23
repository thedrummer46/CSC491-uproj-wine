# Project record

A running log of **facts and decisions**: what was done, why, and what was measured about the setup. These entries are final. Comparisons *between treatments* are not recorded here until all 5 repeats are scored and checked with `same`. Until then, they're observations, not findings.

Each entry says where it belongs in the report: **[Methods]**, **[Results]**, **[Discussion]** or **[AI disclosure]**.

---

## Setup and test suite

- **[Methods] One shared test suite.** `data/splits.json` holds 5 seeded 80/20 splits (seed 1; 1279 training / 320 test batches). Every treatment is scored on these exact rows. LLM repeat *r* is scored on split *r*. T7 is written once and scored on all 5. The lecturer confirmed 5 repeats earns full credit for this group (Nov 22). The splits were reduced from 20 to 5 by regenerating with the same seed, so they're identical to the first 5 of the original 20.
- **[Methods] Statistics.** `same` is used unchanged from the lecturer's `ezr.py` (Cliff's delta + Kolmogorov–Smirnov, no Cohen test: `eps=0`). `ranks` follows gloss6: sort by median, and a new rank starts only when a treatment is not `same` as the rank's leader.
- **[Discussion] `same` is unreliable at 5 values per side.** In 2000 simulated pairs drawn from the *same* distribution, it said "different" **69%** of the time at n=5 (51% at n=10, 29% at n=20, 9% at n=50). It caught a real 1-SD gap 90% of the time at n=5. So differences found with 5 repeats need cautious wording.
- **[Methods] ezr version.** The lecturer's Python `ezr.py`, copied unchanged into `lib/ezr.py` from `timm/ezr` at commit `e2d95ae`. The repo's Python `.gitignore` template ignored `lib/`, so that line was commented out.

## The interface (skills.json)

- **[Methods] Who wrote it.** Steps 1–3 were written by Tyler and used for T5/T6 repeat 1 *before* T7 existed. That departs from the lecturer's advice to do T7 first. Steps 4–10 were drafted by Claude and agreed by the group. Each step's interface is frozen once T7 has built and scored it, and never changed after any trial has run it.
- **[Methods] Step order changed for reuse** (Nov 22): choose moved from step 7 to step 5, right after flow, because choosing is flow's learn-one-at-a-time loop plus a selection rule. Final order: remember → guess → certify → flow → choose → blame → justify → fix → spot → race.
- **[Methods] "Good" means Quality+ of 6 or more** (53% of batches). Used by `worth`, `dragging` and `verdict`.
- **[Methods] In choose, both goal cells are hidden (`'?'`) until `label(row)` is called.** Each call counts as one tasting, and the scorer enforces the 20-tasting budget.
- **[Methods] "Off-the-shelf model" in ask 20** means always predicting the average Quality+ of the training batches. That's a baseline every treatment can build, including T1/T2 without scikit-learn.

## Scoring decisions and fixes

- **[Methods] Step 1** was first planned as the average `wins` of 5 returned batches. On Nov 22 it changed to *all* batches `remember` returns (at most 5%), because the interface says "at most 5%" without saying "best first."
  - **[Discussion] Known loophole:** returning fewer batches raises the average. T1 repeat 1 returns 5 batches where the others return 16, and scores 84.7 against 79.4 on split 1. Allowed by the interface, but step 1 scores aren't strictly comparable. *Open decision:* switch to scoring the top 5 of what's returned.
- **[Methods] Scorer fix (Nov 23):** the step 1 `summary` check accepted only the mean as "typical." T1 and T2 report the median, which is valid, so they were wrongly marked "WRONG." The check now accepts the mean or median, and step 1 was re-scored.
- **[Methods] Step 2:** Quality+ is hidden in every test batch passed to `guess` ("from its lab sheet alone"). Alcohol stays visible, since it's on the lab sheet.
- **[Methods] Step 3:** of 500 random test pairs, the share `certify` gets right among those it calls different. Calling a winner between two batches with equal true Quality counts as wrong. A coin flip scores about **28%** under this rule, since about 40% of pairs are ties.
- **[Methods] Step 4:** batches are fed one at a time; every 100 batches, the test error is measured; the score is the average over that curve.
- **[Methods] Step 6:** a plain ezr tree refit on only the 3 blamed readings, scored by test error. Benchmarks on split 1: 3 arbitrary readings (sugar, chlorides, pH) give 0.68; all 10 readings give 0.50.

## Line counts

- **[Methods] "New code per step"** = non-blank lines in each treatment's `stepN.py` as of tag `stepN`, averaged over repeats (`scripts/newcode.py`, "cloc per tag"). This approach was chosen because T5/T6 steps 1–3 were committed together, so `git diff step1..step2` couldn't separate them. Tags may be placed after the fact on any commit that has every treatment's `stepN.py`.
- **[Methods] Lines include comments and docstrings.** The LLM files carry long docstrings, so they look bigger than their actual code. Not changed, so that it stays one consistent rule for everyone.

## Trial setup

- **[Methods]** Each LLM step is one `claude -p` call: model `opus` (Claude Opus 5.5), no tools, one turn, a $2 cap per call, run in a fresh empty temp folder outside the repo (no CLAUDE.md, no history). The prompt contains:
  - the CSV header and column conventions
  - the background (plain Python / scikit-learn / the full text of `ezr.py`)
  - that trial's own earlier steps, each marked "importable as `import stepK`"
  - the step's two asks and its interface

  T2/T4/T6 also get "Reuse the library and the earlier steps wherever you can" from step 2 on.
- **[Methods] Limit:** the saved replies (`results/raw/`) don't include the prompt sent. Prompts can only be rebuilt from `scripts/run_trial.py` as committed.
- **[Methods] Workload and speed requirement (added Nov 23).** Each step in skills.json now states the scorer's workload (`calls`). The prompt says everything for one split must finish within **90 seconds** on an ordinary laptop. It states the workload and the limit, not *how* to meet them: no hint to cache. The scorer runs each (treatment, repeat, split) in its own process and stops it at 90 s, recording it blank as "too slow". Why: T1's and T6's repeat 1 rebuilt their models on every call (up to 28 min per split), which nothing in the original interface ruled out. T7's slowest step uses 38 s. Repeat 1 of T1, T2, T5 and T6 was run under the old prompt; it's kept in git history (commit "T1 and T2 repeat 1, steps 1-6 (before workload requirement)", and `a06232e` for T5/T6), then deleted and re-run under the new prompt so every repeat sees the same interface. The old runs' costs stay in `cost.csv`.
- **[Methods] Never re-run or edit an existing step.** The runner overwrites without asking, and later steps were built on the original.

## Facts about the code (repeat 1)

- **[Discussion] Treatments not told to reuse still do.** The prompt shows earlier steps as importable. T5 step 3 imports step 2 (`guess`, `_fit`, `_at`, `_known`). T1 imports earlier steps from step 2 on. The reuse instruction is therefore a weaker treatment than its name suggests.
- **[Discussion] T6 reuses real functions:** step 2 imports step 1's header helpers; step 3 imports step 2's `scored`, `expected` and `guess`.
- **[Results] Some LLM code rebuilds its model on every call:**

  | Treatment | Slow call | Scoring time per split |
  |---|---|---|
  | T6 | `guess` | about 322 s for step 2 |
  | T1 | `certify` | about 3.4 s per call, so about 28 min for step 3 |
  | T7, T5 | — | about 1 s |

  The time is spent on local computing, not plan usage. *Open decision:* add a stated calling pattern and a time limit to skills.json, and re-run repeat 1.
- **[Discussion] Density stands in for alcohol.** `blame` may not name a goal column, so Alcohol is excluded. But Density correlates with alcohol (r = −0.50), against r = −0.17 with Quality, so a treatment that blames Density brings alcohol's predictive power in through the side door. T1 repeat 1 blamed Density.

## About the data

- **[Discussion] The two goals conflict.** Alcohol and Quality correlate at +0.48, but `Alcohol-` is to be minimized. Quality 7–8 batches have a median alcohol of 11.6%, against 10.2% overall.
- **[Discussion] Quality is coarse:** 3: 10, 4: 53, **5: 681, 6: 638**, 7: 199, 8: 18. So 83% of batches score 5 or 6, and 14% score 7 or more.
- **[Discussion] Random tasting is a strong baseline.** The best of 20 random tastings has a median `wins` of 68, better than 98% of batches. Only 4 of 1599 batches score `wins` of 90 or more.

## T7 design decisions

- **[Methods] Alcohol is treated as a lab reading** in the steps that predict Quality alone (guess, certify, flow, blame), by renaming `Alcohol-` to `Alcohol` in step 2's `table`. Split 1 prediction error: 0.50, against 0.54 with alcohol as a goal (baseline 0.68). Alcohol is never *named* by `blame`, because the interface forbids goal columns.
- **[Methods] Both goals are kept in choose** (and race, which builds on it), because ask 14 names both. Tested and rejected on 25 runs (median `wins`):

  | Design | Median `wins` |
  |---|---|
  | Plain `acquire` | 67 |
  | ezr `holdout` as is | 62 |
  | Alcohol visible, search on Quality alone | 45–65 |
  | Scoring on Quality alone | best Quality 7.24, against 7.12 for random |
  | **Chosen: `acquire` 15 tastings (`Start=8`), then the tree's top 5, then the best of all 20** | **73 (mean 78)** |

  Random tasting gives 69.
- **[Methods] Flow uses nearest neighbours** (ezr `xdist` + `heapq`, k=10): curve average 0.54, final 0.49, about 38 s per split. Rejected: a tree grown once at 100 batches (0.59), and one average batch per Quality level (0.61).
- **[Methods] Leaf size 32 for certify.** Leaf sizes 4–128 all gave 55–57% accuracy; 32 made the fewest confident calls. At ezr's default leaf of 4, `same` called 72% of neighbouring leaves "different."
- **[Methods] Remember ranks every batch** (`ydist` on all rows), because past labels are already paid for. Only choose spends a tasting budget.

## Cost

- **[Results] Spend so far:** $7.09, 19 successful calls plus 1 failed call (T5 step 1, $0.38, no code returned), as of Nov 23.
- **[Discussion] Cost per step rises,** because each prompt carries all earlier steps:

  | Step | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | Average cost | $0.22 | $0.29 | $0.30 | $0.43 | $0.60 | $0.71 |

  Projected: about $6.50 per 10-step trial, about $200–220 for all 30 trials (about $65–75 per person, under the $100 cap). The trial runner's `--cap` adds up the shared `cost.csv`, so it's a team total, not per person.

## AI disclosure

- **[AI disclosure] Claude built the harness:** the splits, the stats wrapper, every scorer, the tests, `demo.py`, `newcode.py`, the skills.json interface for steps 4–10, and the docs.
- **[AI disclosure] Claude designed T7 with Brett.** For every step it gave outlines and pseudocode, tested design options against the splits (all the comparisons above), and reviewed each version. Its experiments drove T7's key choices: alcohol as an input, leaf sizes, nearest neighbours, and choose's 15 + 5 plan.
- **[AI disclosure] Who wrote T7's code:**

  | Step | Written by |
  |---|---|
  | 1–4 | Brett, with fixes Claude suggested and explained. Claude also tidied step 4's demo output. |
  | 5 | Brett and Claude: Claude fixed `taste`'s last lines, wrote `main`, and added the budget cap. |
  | 6 | Claude entirely, at Brett's request, from a plan Brett approved. |

- **[AI disclosure] So T7 is human-directed, AI-assisted development** by an assistant that knew all ten steps in advance and could test designs. It isn't unaided human coding, and that's part of why its code is small.

---

## Observations (provisional, n = 1, untested)

*Comparisons between treatments go here once all 5 repeats are scored and checked with `same`. Until then they're kept out of this record on purpose.*
