"""Step 3 of the winemaker's assistant.

Two last questions from the cellar:

  "When you say batch A will beat batch B, tell me whether that's a real
   difference or just tasting noise."               -> certify(train, a, b)

  "How often are your score estimates right on batches you've never seen
   before?"                                         -> accuracy(train, test)

Input convention (same as steps 1 and 2)
----------------------------------------
`train` and `test` are lists of lists.  Row 0 is the header (strings);
every later row holds numbers.  A column name ending in '+' is a goal to
maximize, one ending in '-' is a goal to minimize, anything else is a lab
reading.  Only the readings are used to predict; Quality+ is the outcome
being guessed.

How certify() decides
---------------------
Both batches are scored the way step 2 scores them: a distance-weighted
average of their nearest past batches on the z-scored, correlation-
weighted readings.  That average comes with a spread -- the past batches
that look like this one did not all get the same score from the panel --
so each estimate carries a standard error, computed from the weighted
scatter of its own neighbours over their effective count (Kish's
formula, since the neighbours are weighted, not counted).

A verdict of 'A' or 'B' is only given when the gap between the two
estimates clears two bars:

  * a two-sided 95% Welch-style t interval on the difference, so a gap
    that the neighbour scatter could easily have produced is called
    noise; and
  * half the panel's own step size (half a point on a whole-number
    score), so a gap too small for the panel to ever express is called
    noise even when the neighbourhoods are unusually tight.

The neighbour scatter is floored by the model's own out-of-sample error
(leave-one-out over a bounded sample of the training rows), so a pocket
of past batches that happened to agree perfectly cannot be mistaken for
certainty about a new batch.

Cost
----
Everything derived from one train list -- column statistics, the
neighbour table, the leave-one-out error, the panel's step size -- is
built once and cached on that list, so the 500 certify() pairs and the
320 accuracy() rows each cost one scan of the training rows: well under
a second in total for a 320-row split, and a few seconds for a cellar of
thousands.  Only the Python standard library is used.
"""

import math
import random
import sys

import step1
import step2

# Never trust a local neighbourhood to be quieter than this share of the
# model's own out-of-sample error variance.
MIN_VAR_SHARE = 0.25

# Rows used for the leave-one-out error estimate (bounds the setup cost).
LOO_SAMPLE = 200

SEED = step2.SEED

# Two-sided 95% critical values of Student's t, by degrees of freedom.
# The first entry at or above the df is used, which errs on the cautious
# side (a slightly wider interval, so slightly more 'same' verdicts).
_T95 = [(1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571),
        (6, 2.447), (7, 2.365), (8, 2.306), (9, 2.262), (10, 2.228),
        (12, 2.179), (15, 2.131), (20, 2.086), (25, 2.060), (30, 2.042)]


def _critical(df):
    """Critical value for a two-sided 95% test with `df` degrees of freedom."""
    d = int(df)
    if d < 1:
        return _T95[0][1]
    for cut, t in _T95:
        if d <= cut:
            return t
    return 1.96


# ----------------------------------------------------------- neighbours --

def _neighbours(model, row):
    """The k nearest past batches as [(distance^2, index), ...], nearest first.

    Same geometry as step 2's guess(): z-scored readings, each weighted by
    how strongly it tracks Quality+.  Rows beyond the current shortlist are
    abandoned part-way through, so the scan stays cheap.
    """
    want = step2._encode(model, row)
    weight = model.weight
    k = model.k

    best = []
    worst = float("inf")
    for idx, point in enumerate(model.points):
        d = 0.0
        for pos, a in enumerate(point):
            diff = want[pos] - a
            d += weight[pos] * diff * diff
            if d >= worst:
                break                     # cannot make the shortlist
        else:
            if len(best) < k:
                best.append((d, idx))
                if len(best) == k:
                    best.sort()
                    worst = best[-1][0]
            else:
                best[-1] = (d, idx)
                best.sort()
                worst = best[-1][0]
    best.sort()
    return best


def _weigh(best):
    """Turn a neighbour shortlist into [(weight, score), ...]."""
    out = []
    for d, idx in best:
        out.append((1.0 / (math.sqrt(d) + 1e-9), idx))
    return out


def _predict(model, row):
    """(estimated score, standard error, effective neighbour count).

    The estimate is step 2's answer.  The standard error comes from how
    much the neighbours themselves disagreed: if similar past batches got
    similar scores the estimate is firm, if they were all over the place
    it is not.
    """
    best = _neighbours(model, row)
    if not best:
        return float(model.fallback), float("inf"), 0.0

    mass = 0.0
    sqmass = 0.0
    total = 0.0
    pairs = []
    for d, idx in best:
        w = 1.0 / (math.sqrt(d) + 1e-9)
        y = model.ys[idx]
        pairs.append((w, y))
        mass += w
        sqmass += w * w
        total += w * y
    if mass <= 0.0 or sqmass <= 0.0:
        return float(model.fallback), float("inf"), 0.0

    mean = total / mass
    var = sum(w * (y - mean) ** 2 for w, y in pairs) / mass
    neff = mass * mass / sqmass            # Kish's effective sample size

    # the neighbourhood can look quieter than the model really is
    floor = _extras(model)["resid_var"] * MIN_VAR_SHARE
    if var < floor:
        var = floor

    if neff > 1.0:
        var *= neff / (neff - 1.0)         # unbiased-ish for the weights used
        se = math.sqrt(var / neff)
    else:
        se = math.sqrt(var) if var > 0.0 else 0.0
    return float(mean), float(se), float(neff)


# ---------------------------------------------------- per-train extras --

_EXTRAS = {}


def _extras(model):
    """Leave-one-out error and panel step size for one model, built once."""
    key = id(model)
    hit = _EXTRAS.get(key)
    if hit is not None and hit[0] is model:
        return hit[1]
    if len(_EXTRAS) > 4:
        _EXTRAS.clear()
    built = {"resid_var": _loo_var(model), "step": _step(model.ys)}
    _EXTRAS[key] = (model, built)
    return built


def _step(ys):
    """The smallest score difference the panel actually expresses.

    For a whole-number quality score this is 1, so half of it -- half a
    point -- is the smallest gap worth calling real.
    """
    distinct = sorted(set(ys))
    if len(distinct) < 2:
        return 1.0
    gap = min(b - a for a, b in zip(distinct, distinct[1:]))
    return gap if gap > 0.0 else 1.0


def _loo_var(model):
    """Mean squared leave-one-out error of the step 2 estimate on train.

    Each sampled training batch is predicted from the others, so this is
    the model's honest error on batches it has not been told about.  It
    acts as the floor under every neighbourhood's own scatter.
    """
    points = model.points
    ys = model.ys
    weight = model.weight
    k = model.k
    n = len(points)
    if n < 3:
        return 0.0

    idxs = list(range(n))
    if n > LOO_SAMPLE:
        random.Random(SEED).shuffle(idxs)
        idxs = idxs[:LOO_SAMPLE]

    total = 0.0
    counted = 0
    for i in idxs:
        want = points[i]
        best = []
        worst = float("inf")
        for idx, point in enumerate(points):
            if idx == i:
                continue                   # the batch may not vote on itself
            d = 0.0
            for pos, a in enumerate(point):
                diff = want[pos] - a
                d += weight[pos] * diff * diff
                if d >= worst:
                    break
            else:
                if len(best) < k:
                    best.append((d, idx))
                    if len(best) == k:
                        best.sort()
                        worst = best[-1][0]
                else:
                    best[-1] = (d, idx)
                    best.sort()
                    worst = best[-1][0]
        if not best:
            continue
        mass = 0.0
        acc = 0.0
        for d, idx in best:
            w = 1.0 / (math.sqrt(d) + 1e-9)
            mass += w
            acc += w * ys[idx]
        if mass <= 0.0:
            continue
        err = ys[i] - acc / mass
        total += err * err
        counted += 1
    return total / counted if counted else 0.0


# ----------------------------------------------------------- certify --

def certify(train, a, b):
    """Which of two batches will score better -- or 'same' if it is noise.

    Returns 'A' if batch `a` is expected to beat batch `b`, 'B' for the
    other way round, and 'same' when the gap between the two estimates is
    inside what the past batches' own disagreement could explain, or is
    too small for the panel to ever show.
    """
    model = step2._model(train)
    if not model.points:
        return "same"

    pa, sea, na = _predict(model, a)
    pb, seb, nb = _predict(model, b)
    diff = pa - pb

    spread = math.sqrt(sea * sea + seb * seb)
    if spread != spread or spread == float("inf"):
        return "same"

    df = na + nb - 2.0
    need = _critical(df) * spread

    floor = 0.5 * _extras(model)["step"]   # half a point: below this the
    if need < floor:                       # panel could not tell either
        need = floor

    if diff > need:
        return "A"
    if -diff > need:
        return "B"
    return "same"


# ---------------------------------------------------------- accuracy --

def _round_half_up(x):
    """Round to the nearest whole number, .5 going up (not banker's rounding)."""
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def accuracy(train, test):
    """Share of unseen batches whose guessed score rounds to the true one.

    The estimate for each test batch comes from the training rows alone
    (nothing in `test` is used to predict), is rounded to the nearest
    whole score, and is counted right when it matches the panel's score.
    Test rows with no recorded score are skipped; with nothing to check,
    the answer is 0.0.
    """
    if not test or len(test) < 2 or not train or len(train) < 2:
        return 0.0

    header = train[0]
    goals, _readings = step1._split_columns(header)
    target = step2._target_index(header, goals)
    if target is None:
        target = step2._target_index(test[0], step1._split_columns(test[0])[0])
    if target is None:
        return 0.0

    hits = 0
    n = 0
    for row in test[1:]:
        actual = step1._value(row, target)
        if actual is None:
            continue
        n += 1
        if _round_half_up(step2.guess(train, row)) == _round_half_up(actual):
            hits += 1
    return float(hits) / n if n else 0.0


# ---------------------------------------------------------------- CLI --

def main(argv):
    if len(argv) != 2:
        print("usage: python step3.py data.csv")
        return 1

    rows = step1.load_csv(argv[1])
    header = rows[0]
    train, test = step2._split(rows)
    goals, _readings = step1._split_columns(header)
    target = step2._target_index(header, goals)
    if target is None:
        print("no Quality+ column in %s" % argv[1])
        return 1

    print("train %d batches, test %d batches" % (len(train) - 1, len(test) - 1))

    share = accuracy(train, test)
    print("\nscore estimates on batches never seen before:")
    print("  right to the nearest point   %.1f%%" % (100.0 * share))

    scored = [r for r in test[1:] if step1._value(r, target) is not None]
    if len(scored) < 2:
        return 0

    rng = random.Random(SEED)
    pairs = 500
    calls = {"A": 0, "B": 0, "same": 0}
    committed = 0
    correct = 0
    real_gaps = 0
    real_caught = 0
    ties = 0
    ties_called_same = 0

    for _ in range(pairs):
        a = scored[rng.randrange(len(scored))]
        b = scored[rng.randrange(len(scored))]
        verdict = certify(train, a, b)
        calls[verdict] += 1

        ya = step1._value(a, target)
        yb = step1._value(b, target)
        if ya > yb:
            truth = "A"
        elif yb > ya:
            truth = "B"
        else:
            truth = "same"

        if truth == "same":
            ties += 1
            if verdict == "same":
                ties_called_same += 1
        else:
            real_gaps += 1
            if verdict == truth:
                real_caught += 1

        if verdict != "same":
            committed += 1
            if verdict == truth:
                correct += 1

    print("\n%d pairs of unseen batches put to certify():" % pairs)
    print("  called A %d, called B %d, called noise %d"
          % (calls["A"], calls["B"], calls["same"]))
    if committed:
        print("  when it names a winner, right  %.1f%% (%d calls)"
              % (100.0 * correct / committed, committed))
    if real_gaps:
        print("  pairs that really differed, called %.1f%% of them"
              % (100.0 * real_caught / real_gaps))
    if ties:
        print("  pairs that really tied, called noise %.1f%%"
              % (100.0 * ties_called_same / ties))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
