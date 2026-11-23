# CSC491-uproj-wine

CSC491 shrinking-code project. Does a well-factored symbolic library (ezr) mean each new AI skill needs less new code? We test this on one client, a winemaker, using one MOOT dataset, [`data/Wine_quality.csv`](data/Wine_quality.csv).

**Teammates, start here: [docs/workflow.md](docs/workflow.md)** has the rules, who does what, and how to do a step.

- Requirements (20 winemaker asks, tagged by skill): [docs/requirements.md](docs/requirements.md)
- How each skill is scored: [docs/assessment.md](docs/assessment.md)
- Treatments T1–T7: [treatments/](treatments/)
- The one test suite (5 fixed 80/20 splits, shared by every treatment): `data/splits.json`, rebuilt with `python scripts/make_splits.py`
- Stats (the lecturer's `same` and `ranks`): [harness/stats.py](harness/stats.py)
- ezr (the lecturer's library, Python version, pinned copy): [lib/ezr.py](lib/ezr.py)
- Steps are git tags `step1` … `step10`, one skill per step.

## Setup

```sh
pip install -r requirements.txt
```

## Demos

Each demo runs with one command, from the repo root. The demos run our hand-written T7. Swap the `7` for another treatment's number (and add `--repeat N`) to run that one instead.

| Step | Skill | Command |
|------|-------|---------|
| —    | all tests | `pytest` |
| 1    | remember: the exceptional batches, and a normal batch | `python scripts/demo.py 7 1` |
| 2    | guess: kinds of batch, and Quality guesses for unseen batches | `python scripts/demo.py 7 2` |
| 3    | certify: is batch A really better than B, and how often guesses are right | `python scripts/demo.py 7 3` |

## Scoring

```sh
python -m harness.score 1      # or 2, ...
```

This scores every treatment's step N on the shared test splits, using the methods in [docs/assessment.md](docs/assessment.md), and writes the results to `results/scores/stepN.csv`.

## Measuring new code

```sh
python scripts/newcode.py > results/newcode.csv
```

For each `stepN` tag, this counts the lines in every treatment's `stepN.py` as it was at that tag ("cloc per tag"), averaged over repeats, along with that step's share of the treatment's code so far. That data feeds the Results histogram.

## License

MIT. See [LICENSE](LICENSE).
