# How each skill is assessed

**Status:** Agreed by the group (Nov 22). Per the lecturer, we worked through each skill and asked "how would we assess this one?" before running any treatments.

## Ground rules (all skills)

- **One test suite.** `data/splits.json` holds 5 fixed, seeded 80/20 splits (1279 train / 320 test). Every treatment, T1–T7, is scored on exactly these splits. Nothing re-splits.
- **One number per split.** Each skill below yields one score per split, so each treatment gets 5 scores per skill. LLM repeat *r* (T1–T6) is scored on split *r*; T7 is written once and scored on all 5.
- **Speed limit.** Each step's interface states the scorer's workload (`calls` in `scripts/skills.json`). Everything for one split must finish within **90 CPU seconds** (the job's own processor time, single-threaded, so jobs running side by side don't affect each other), or it scores blank, the same as a crash.
- **Compare only with `same`.** Treatments are ranked with `harness/stats.py` (`same`, `ranks`), the lecturer's Cliff's delta + KS test. Ties share a rank. A difference `same` can't see is not reported.
- **Scoring sees everything.** The harness may use every true label to *score*. Treatments only see what their ask allows (for example, 20 labels for `choose`).
- **`wins`** (from ezr) puts a batch on a −100..100 scale. 100 is the best batch in the pool and 0 is an average one, over both goals (`Quality+`, `Alcohol-`).

## Per skill

| Step | Skill | Asks | Score per split | Better | Why this measure |
|---|---|---|---|---|---|
| 1 | remember | 1, 2 | % of the batches `remember` returns (at most 5%, so up to 16 of 320) that really are in the test pool's top 5% by `wins` | higher | Ask 1 is about finding great batches; ask 2 (summary) is checked by the test below, not scored |
| 2 | guess | 3, 4 | Mean absolute error of predicted `Quality` on test batches | lower | Direct; in the panel's own units (score points) |
| 3 | certify | 5, 6 | Pairwise accuracy: on 500 random test pairs, how often "A beats B" is right, counting only pairs it calls a real difference | higher | Ask 5 is a pairwise claim; ask 6 is reported alongside it |
| 4 | flow | 7, 8 | Feed train batches one at a time; after every 100, take MAE on test. Score = mean MAE over the curve | lower | Rewards learning fast, not just ending well |
| 5 | choose | 13, 14 | Label at most 20 train batches, pick one. Score = `wins` of the pick | higher | The ezr "few labels" story, scored with ezr's own measure |
| 6 | blame | 9, 10 | Refit a plain predictor on only the 3 readings it blames; MAE on test | lower | Blame with no ground truth: good blame should keep most of the predictive power |
| 7 | justify | 11, 12 | Fidelity: % of test batches where the stated reason gives the same verdict (good/poor) as the model | higher | An explanation that disagrees with its own model is wrong |
| 8 | fix | 15, 16 | For the worst test batches, apply the suggested change. Score = gain in true `Quality` of the nearest real batch to the changed one | higher | No lab to run the change; the nearest real batch stands in for it |
| 9 | spot | 17, 18 | Corrupt 5% of test batches (×10 or −1 on one reading, swapped columns). F1 of flagging them | higher | Planted anomalies give known answers |
| 10 | race | 19, 20 | Does its verdict ("better than random / than off-the-shelf") match the harness's own `same` verdict? % correct over splits | higher | Race *is* a stats claim; check it against the lecturer's stats |

Every treatment also reports **lines of new code per step** (`scripts/newcode.py`). That's the hypothesis's main measure. The scores above check that shrinking code didn't come from skills that don't work.

## Decisions

The group agreed with the suggestions above:

1. **remember, ask 2** (the summary) is checked by a test, not scored.
2. **blame** is scored by how well the 3 blamed readings alone still predict.
3. **fix** uses the nearest real batch as a stand-in for the lab. Changes can touch any reading, except in ask 16 (sulphates and acidity only).
4. **spot** plants three kinds of bad lab sheet: one reading ×10, a negative reading, or two readings swapped.
5. **`same`** uses ezr.py's default: Cliff's delta + KS, with no Cohen (`eps=0`).
