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


#-- step 3: certify -------------------------------------------
def test_certify_answers_and_is_symmetric():
    m = step(3)
    train, test = score.splits()[0]
    flip = {"A": "B", "B": "A", "same": "same"}
    for a, b in zip(test[1:21], test[21:41]):
        a, b = score.blind(test[0], a), score.blind(test[0], b)
        c = m.certify(train, a, b)
        assert c in flip and m.certify(train, b, a) == flip[c]
        assert m.certify(train, a, a) == "same"


def test_certify_beats_a_coin():
    m = step(3)
    train, test = score.splits()[0]
    val, note = score.step3(m, train, test)
    assert val > 45 and "WRONG" not in note  # ties make 50% hard; a coin gets ~30%


def test_accuracy_is_a_share_better_than_always_6():
    m = step(3)
    train, test = score.splits()[0]
    at = test[0].index("Quality+")
    always6 = sum(r[at] == 6 for r in test[1:]) / (len(test) - 1)
    assert always6 < m.accuracy(train, test) <= 1


#-- step 4: flow ----------------------------------------------
def test_flow_learns_in_place_and_starts_fresh():
    m = step(4)
    train = score.splits()[0][0]
    model = m.start(train[0])
    for r in train[1:51]:
        assert m.learn(model, r) is model     # updated, not rebuilt
    assert len(m.start(train[0]).rows) == 0   # a new start is empty


def test_flow_gets_better_with_more_batches():
    m = step(4)
    train, test = score.splits()[0]
    names, at = test[0], test[0].index("Quality+")
    rows = test[1:101]
    def err(model):
        return sum(abs(m.predict(model, score.blind(names, r)) - r[at]) for r in rows)
    model = m.start(names)
    for r in train[1:101]: m.learn(model, r)
    early = err(model)
    for r in train[101:]: m.learn(model, r)
    assert err(model) < early


def test_worth_means_predicted_6_or_more():
    m = step(4)
    train, test = score.splits()[0]
    model = m.start(train[0])
    for r in train[1:]: m.learn(model, r)
    for r in test[1:31]:
        b = score.blind(test[0], r)
        assert m.worth(model, b) == (m.predict(model, b) >= 6)


def test_step1_scores_on_every_split():
    m = step(1)
    for train, test in score.splits():
        val, note = score.step1(m, train, test)
        assert val > 0 and note == "summary ok"
