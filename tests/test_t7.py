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


#-- step 2: guess ---------------------------------------------
def test_guess_beats_always_guessing_the_mean():
    m = step(2)
    train, test = score.splits()[0]
    at = test[0].index("Quality+")
    mu = sum(r[at] for r in train[1:]) / (len(train) - 1)
    ours = sum(abs(m.guess(train, score.blind(test[0], r)) - r[at]) for r in test[1:])
    base = sum(abs(mu - r[at]) for r in test[1:])
    assert ours < 0.9 * base


def test_kinds_are_a_few_groups_covering_train_best_first():
    m = step(2)
    train = score.splits()[0][0]
    k = m.kinds(train)
    assert 2 <= len(k) <= 10
    assert sum(len(g) for g, _ in k) == len(train) - 1
    qs = [q for _, q in k]
    assert qs == sorted(qs, reverse=True)


def test_kinds_leaves_ezr_settings_alone():
    m = step(2)
    before = score.ezr.the.Leaf
    m.kinds(score.splits()[0][0])
    assert score.ezr.the.Leaf == before


def test_step1_scores_on_every_split():
    m = step(1)
    for train, test in score.splits():
        val, note = score.step1(m, train, test)
        assert val > 0 and note == "summary ok"
