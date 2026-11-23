"""T7 (our hand-written control) meets the skills.json interface."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness import score  # noqa: E402  (also puts lib/ on the path)

T7 = ROOT / "treatments" / "t7"


def step(n):
    if not (T7 / f"step{n}.py").exists():
        pytest.skip(f"T7 step{n} not written yet")
    return score.load(T7, n)


#-- step 1: remember ------------------------------------------
def test_remember_at_most_5_percent_best_first():
    m, rows = step(1), score.wine()
    got = m.remember(rows)
    assert 0 < len(got) <= 0.05 * (len(rows) - 1)
    tbl = score.ezr.Tbl(rows)
    d = [score.ezr.ydist(tbl, r) for r in got]
    assert d == sorted(d)                      # best first
    assert max(d) <= min(score.ezr.ydist(tbl, r) for r in tbl.rows
                         if r not in got)      # nothing better was left out


def test_summary_names_every_column():
    m, rows = step(1), score.wine()
    s = m.summary(rows)
    assert list(s) == rows[0]
    assert abs(s["Quality+"] - 5.636) < 0.01


def test_step1_scores_on_every_split():
    m = step(1)
    for train, test in score.splits():
        val, note = score.step1(m, train, test)
        assert val > 0 and note == "summary ok"
