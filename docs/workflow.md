# Team workflow

**Due Mon Nov 30.** One 3-page ACM PDF to Moodle, with this repo's URL on page one.

## What we're testing

Each new skill should need *less new code* than the one before if it can reuse earlier skills, especially when built on ezr. We build the 10 skills in order, as steps, under 7 treatments, and count the new lines each step adds.

| Treatment | Who writes the code | Background | Told to reuse earlier skills? |
|---|---|---|---|
| T1 / T2 | LLM | plain Python | no / yes |
| T3 / T4 | LLM | scikit-learn | no / yes |
| T5 / T6 | LLM | ezr ([lib/ezr.py](../lib/ezr.py)) | no / yes |
| T7 | **us**, by hand | our choice | yes, as much as possible |

Step order: **1 remember → 2 guess → 3 certify → 4 flow → 5 choose → 6 blame → 7 justify → 8 fix → 9 spot → 10 race.** The asks for each step are in [requirements.md](requirements.md). How each step is scored is in [assessment.md](assessment.md).

## Rules (breaking these spoils the results)

1. **Never add a `CLAUDE.md`** (or any AI memory/context file) to this repo.
2. **LLM trials run outside the repo**: one fresh empty folder per trial, a one-shot call (`claude -p` or the API), memory off. A new chat window is *not* enough. Copy the output into the repo afterwards.
3. **Every treatment is scored on `data/splits.json`.** Never re-split or regenerate it.
4. **Budget: $100 per person, hard stop.** Use Opus 5.5, not Fable 5.2. Log every run's cost in `results/cost.csv`. If you hit the cap, stop and tell the group. Running out is itself a result we report.
5. **No coding before the lecturer approves** [requirements.md](requirements.md), and the group agrees on [assessment.md](assessment.md).

## Who does what

The split follows the treatments: each person runs two LLM treatments and pays for them from their own $100.

| | Brett | Makayla | Tyler |
|---|---|---|---|
| **T7 steps** (write by hand + its scorer) | 1 remember, 2 guess, 3 certify, 4 flow, 5 choose, 6 blame | 7 justify, 8 fix | 9 spot, 10 race |
| **LLM treatments to run** | T1, T2 (plain Python) | T3, T4 (scikit-learn) | T5, T6 (ezr) |
| **Report sections** | 3 Methods | 2 Requirements, 4 Results | 1 Hypotheses, 5 Discussion, AI disclosure |

## Doing a step

**T7 (by hand)**
1. `git pull`. Read the earlier steps' code in `treatments/t7/`, and reuse it wherever you can. That's the point.
2. Write `treatments/t7/stepN.py`. Add its demo command to the README table.
3. Write its scorer (per [assessment.md](assessment.md)) and a test in `tests/`. Run `pytest`.
4. Commit: `T7 stepN: <skill>`. Push.

**LLM treatments (T1–T6)**: once T7 stepN is in:
1. Run the trial runner for your two treatments, stepN, every repeat. (The runner is coming. It sets up the fresh folder and feeds in the background, steps 1..N−1, and the ask.)
2. Copy the outputs to `treatments/tK/rNN/stepN.py` (repeat `NN`). **Don't hand-edit them.** If one is broken, it stays broken and gets reported.
3. Commit: `T3-T4 stepN: <skill>`. Push.

**Tagging**: when *all seven* treatments have stepN committed, one person runs:

```sh
git tag stepN && git push --tags
```

Then `bash scripts/newcode.sh` gives the lines-of-new-code data for the histogram.

**Repeats:** aim for 10 per treatment (full marks on Results), floor of 3. If costs run hot, drop to 3.
