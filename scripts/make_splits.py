"""Make the ONE test suite every treatment is scored on.

Writes data/splits.json: R repeats of a seeded 80/20 shuffle of row indices
(0-based, header excluded). Every treatment, T1..T7, reads this same file;
never re-split per treatment.

    python scripts/make_splits.py            # 5 repeats, seed 1
"""
import csv, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Wine_quality.csv"
OUT = ROOT / "data" / "splits.json"


def main(repeats=5, seed=1, train=0.8):
    with open(DATA, newline="") as f:
        n = sum(1 for _ in csv.reader(f)) - 1
    rng = random.Random(seed)
    cut = int(n * train)
    splits = []
    for _ in range(repeats):
        idx = list(range(n))
        rng.shuffle(idx)
        splits.append({"train": sorted(idx[:cut]), "test": sorted(idx[cut:])})
    OUT.write_text(json.dumps({"data": "data/Wine_quality.csv", "rows": n,
                               "seed": seed, "train": train,
                               "splits": splits}))
    print(f"{OUT.relative_to(ROOT)}: {repeats} splits of {cut}/{n - cut}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
