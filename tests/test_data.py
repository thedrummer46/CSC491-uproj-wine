import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "Wine_quality.csv"


def rows():
    with open(DATA, newline="") as f:
        return list(csv.reader(f))


def test_shape():
    r = rows()
    assert len(r) == 1600  # header + 1599 batches
    assert all(len(x) == 12 for x in r)


def test_goals_tagged():
    head = rows()[0]
    assert head[-2:] == ["Alcohol-", "Quality+"]


def test_all_numeric():
    for x in rows()[1:]:
        [float(v) for v in x]
