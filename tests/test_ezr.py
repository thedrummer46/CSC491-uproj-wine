import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lib import ezr


def wine():
    return ezr.Tbl(ezr.csv(str(ROOT / "data" / "Wine_quality.csv")))


def test_loads_wine():
    t = wine()
    assert len(t.rows) == 1599
    assert len(t.x) == 10  # 12 columns - 2 goals
    assert [c.txt for c in t.y] == ["Alcohol-", "Quality+"]


def test_holdout_beats_average():
    random.seed(ezr.the.Seed)
    t = wine()
    assert ezr.wins(t)(ezr.holdout(t)) > 0
