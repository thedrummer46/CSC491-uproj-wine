import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.stats import same, ranks


def sample(mu, n=50, seed=1):
    r = random.Random(seed)
    return [r.gauss(mu, 1) for _ in range(n)]


def test_same_on_same():
    xs = sample(10)
    assert same(xs, list(reversed(xs)))
    assert same(xs, [x + 0.01 for x in xs])


def test_different_on_shift():
    assert not same(sample(10), sample(12))


def test_ranks_ties_share_rank():
    xs = sample(10)
    d = {"a": xs, "b": [x + 0.01 for x in xs], "c": sample(5)}
    r = ranks(d)
    assert r["a"] == r["b"] == 0 and r["c"] == 1
