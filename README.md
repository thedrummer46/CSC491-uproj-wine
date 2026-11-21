# CSC491-uproj-wine

CSC491 shrinking-code project. Does a well-factored symbolic library (ezr) mean each new AI skill needs less new code? We test this on one client, a winemaker, using one MOOT dataset, [`data/Wine_quality.csv`](data/Wine_quality.csv).

- Requirements (20 winemaker asks, tagged by skill): [docs/requirements.md](docs/requirements.md)
- Treatments T1–T7: [treatments/](treatments/)
- Steps are git tags `step1` … `step10`, one skill per step.

## Setup

```sh
pip install -r requirements.txt
```

## Demos

Each demo runs with one command. They are added as each step lands.

| Step | Skill | Command |
|------|-------|---------|
| —    | data sanity tests | `pytest` |

## Measuring new code

```sh
bash scripts/newcode.sh > results/newcode.csv
```

This prints lines added per treatment per step, taken from the git tags. That data feeds the Results histogram.

## License

MIT. See [LICENSE](LICENSE).
