# How each skill is assessed

**Status:** DRAFT. Per the lecturer, we work through each skill, ask "how would we assess this one?", and don't run any treatments until the group agrees with every method below.

## Ground rules (all skills)

- **One test suite.** `data/splits.json` holds 20 fixed, seeded 80/20 splits (1279 train / 320 test). Every treatment, T1–T7, is scored on exactly these splits. Nothing re-splits.
- **One number per split.** Each skill below yields one score per split, so each treatment gets 20 scores per skill (or fewer if the budget forces fewer repeats).
- **Compare only with `same`.** Treatments are ranked with `harness/stats.py` (`same`, `ranks`), the lecturer's Cliff's delta + KS test. Ties share a rank. A difference `same` can't see is not reported.
- **Scoring sees everything.** The harness may use every true label to *score*. Treatments only see what their ask allows (for example, 20 labels for `choose`).
- **`wins`** (from ezr) puts a batch on a −100..100 scale. 100 is the best batch in the pool and 0 is an average one, over both goals (`Quality+`, `Alcohol-`).

## Per skill

| Step | Skill | Asks | Score per split | Better | Why this measure |
|---|---|---|---|---|---|
| 1 | remember | 1, 2 | Mean `wins` of the 5 test batches it calls "exceptional" | higher | Ask 1 is about finding great batches; ask 2 (summary) is checked by the test below, not scored |
| 2 | guess | 3, 4 | Mean absolute error of predicted `Quality` on test batches | lower | Direct; in the panel's own units (score points) |
| 3 | certify | 5, 6 | Pairwise accuracy: on 500 random test pairs, how often "A beats B" is right, counting only pairs it calls a real difference | higher | Ask 5 is a pairwise claim; ask 6 is reported alongside it |
| 4 | flow | 7, 8 | Feed train batches one at a time; after every 100, take MAE on test. Score = mean MAE over the curve | lower | Rewards learning fast, not just ending well |
| 5 | blame | 9, 10 | Refit a plain predictor on only the 3 readings it blames; MAE on test | lower | Blame with no ground truth: good blame should keep most of the predictive power |
| 6 | justify | 11, 12 | Fidelity: % of test batches where the stated reason gives the same verdict (good/poor) as the model | higher | An explanation that disagrees with its own model is wrong |
| 7 | choose | 13, 14 | Label at most 20 train batches, pick one. Score = `wins` of the pick | higher | The ezr "few labels" story, scored with ezr's own measure |
| 8 | fix | 15, 16 | For the worst test batches, apply the suggested change. Score = gain in true `Quality` of the nearest real batch to the changed one | higher | No lab to run the change; the nearest real batch stands in for it |
| 9 | spot | 17, 18 | Corrupt 5% of test batches (×10 or −1 on one reading, swapped columns). F1 of flagging them | higher | Planted anomalies give known answers |
| 10 | race | 19, 20 | Does its verdict ("better than random / than off-the-shelf") match the harness's own `same` verdict? % correct over splits | higher | Race *is* a stats claim; check it against the lecturer's stats |

Every treatment also reports **lines of new code per step** (`scripts/newcode.sh`). That's the hypothesis's main measure. The scores above check that shrinking code didn't come from skills that don't work.

## Questions for the group

1. **remember, ask 2** (the summary) isn't scored, only checked. Is that OK, or do we want a number for it (e.g. error of the summary's means against the test set)?
2. **blame** has no true answer. Is "the blamed readings still predict well" a fair test?
3. **fix** stands in for a real lab with the nearest real batch. Is that acceptable, or do we restrict changes to the two readings in ask 16?
4. **spot**: are the corruption types (×10, negative, swapped columns) the right kinds of "wrong lab sheet"?
5. Should `same` use `eps` (Cohen) as well? ezr.py leaves it off by default; gloss6 lists all three.
