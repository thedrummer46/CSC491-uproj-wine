"""Step 4 of the winemaker's assistant.

The last two requests from the cellar:

  "As each new batch comes off the line, tell me right away whether it's
   worth sending to the panel."                      -> worth(model, row)

  "Keep sharpening your sense of what makes a good batch as this
   season's scores come in, without starting over."  -> learn(model, row)

and the two helpers those rest on:

  start(names)          -- an empty model that knows only the header
  predict(model, row)   -- the Quality+ this batch looks likely to get

Input convention (same as steps 1-3)
------------------------------------
`names` is the header row (strings); a column name ending in '+' is a
goal to maximize, one ending in '-' is a goal to minimize, anything else
is a lab reading.  `row` is a list of numbers in the same column order.
A row handed to learn() is a scored batch, so it carries its goals; a
row handed to predict() or worth() only needs its readings -- the goal
cells are ignored even when present, because Quality+ is the thing being
guessed and the other goals are judged alongside it.

What the model is
-----------------
A ridge regression of Quality+ on the z-scored lab readings, held in the
form of its sufficient statistics: the count, the running sums of the
readings, their sums of cross-products, and their sums against the score.
One scored batch adds itself into those sums -- a fixed handful of
multiply-adds, no re-reading of anything seen earlier -- and the
coefficients are re-solved from the sums only when a prediction is next
asked for.  So learn() never refits from scratch, yet the coefficients
after 1279 batches are the same ones a from-scratch fit on all 1279
would give.  (The one order-dependence: a reading that arrives missing is
filled in with the running mean at that moment.)

Why this shape and not step 2's neighbours
------------------------------------------
Step 2 could afford to hold the whole cellar and search it.  Here the
answer has to come back "right away" while the line is running, and the
model has to absorb a score the moment the panel reports it.  A linear
model in sufficient-statistic form does both in constant time per batch
and constant memory: the cost of learn() does not grow as the season
goes on, and predict() is one dot product over eleven readings.

Cost
----
learn()   -- O(d^2) with d = 11 readings: ~120 multiply-adds per batch.
predict() -- O(d), plus one Cholesky solve (d = 11) the first time a
             prediction follows new data, cached until more arrives.
The stated workload (1279 learns, 320 predictions after every 100 of
them, so ~4000 predictions and 13 solves) is milliseconds of work, not
seconds.  Only the Python standard library is used.
"""

import math
import sys
import time

import step1
import step2

# Ridge penalty, applied on the correlation scale (readings are z-scored
# before the fit), so one value is sensible for every column.  Small
# enough not to flatten a real signal, large enough to keep two readings
# that move together from producing wild opposing coefficients early in
# the season.
RIDGE = 0.05

# Below this many scored batches a fitted slope is noise; the running
# average score is the honest answer.
MIN_ROWS = 8

# With nothing seen at all: the middle of a 0-10 panel scale.
DEFAULT_SCORE = 5.5

# "Worth sending to the panel" means a predicted Quality+ of 6 or more.
WORTH = 6.0
_EPS = 1e-9


# ------------------------------------------------------------------ model --

class _Model(object):
    """Sufficient statistics for the ridge fit, plus the cached solution."""

    __slots__ = ("names", "xcols", "target",
                 "n", "sx", "sxx", "sy", "sxy", "ymin", "ymax",
                 "dirty", "base", "cols", "mu", "sd", "coef")

    def seen(self):
        """How many scored batches have gone into the model."""
        return self.n


def start(names):
    """An empty model that knows only the header row.

    It can be handed to predict() straight away -- with nothing learned
    it answers the middle of the scale -- and to learn() one batch at a
    time thereafter.
    """
    header = [str(s).strip() for s in (names or [])]
    goals, readings = step1._split_columns(header)
    target = step2._target_index(header, goals)

    model = _Model()
    model.names = header
    model.xcols = list(readings)          # data column index of each reading
    model.target = target

    d = len(model.xcols)
    model.n = 0
    model.sx = [0.0] * d                  # sum of each reading
    model.sxx = [[0.0] * d for _ in range(d)]   # sum of reading i * reading j
    model.sy = 0.0
    model.sxy = [0.0] * d                 # sum of reading i * score
    model.ymin = None
    model.ymax = None

    model.dirty = True                    # coefficients need (re-)solving
    model.base = DEFAULT_SCORE
    model.cols = []                       # readings actually used in the fit
    model.mu = []
    model.sd = []
    model.coef = []
    return model


# -------------------------------------------------------------------- learn --

def _readings(model, row):
    """This batch's readings, missing cells filled with the running mean.

    Filling with the mean lets a batch with a blank cell still teach the
    model about its other readings without pulling the fit in any
    direction on the blank one.
    """
    n = model.n
    out = []
    for pos, col in enumerate(model.xcols):
        v = step1._value(row, col)
        if v is None:
            v = model.sx[pos] / n if n else 0.0
        out.append(v)
    return out


def learn(model, row):
    """The model after seeing one more scored batch.

    The batch is added into the running sums -- a fixed amount of work,
    whatever the season has already brought in -- and the coefficients
    are marked stale, to be re-solved the next time anyone asks for a
    prediction.  A batch with no recorded Quality+ teaches nothing and is
    passed over.  The same model object is returned, so callers may write
    either `learn(m, row)` or `m = learn(m, row)`.
    """
    if model is None or not model.xcols or model.target is None:
        return model

    y = step1._value(row, model.target)
    if y is None:
        return model                      # unscored: nothing to learn from

    x = _readings(model, row)

    model.n += 1
    model.sy += y
    sx = model.sx
    sxy = model.sxy
    sxx = model.sxx
    for i, xi in enumerate(x):
        sx[i] += xi
        sxy[i] += xi * y
        rowi = sxx[i]
        for j in range(i, len(x)):
            rowi[j] += xi * x[j]          # upper triangle only; it is symmetric

    if model.ymin is None or y < model.ymin:
        model.ymin = y
    if model.ymax is None or y > model.ymax:
        model.ymax = y

    model.dirty = True
    return model


# --------------------------------------------------------------- the fit --

def _cross(model, i, j):
    """Sum of reading i times reading j, from the stored upper triangle."""
    return model.sxx[i][j] if i <= j else model.sxx[j][i]


def _cholesky_solve(a, b):
    """Solve a symmetric positive-definite system; None if it is not."""
    m = len(b)
    lower = [[0.0] * m for _ in range(m)]
    for i in range(m):
        rowi = lower[i]
        for j in range(i + 1):
            rowj = lower[j]
            s = a[i][j]
            for k in range(j):
                s -= rowi[k] * rowj[k]
            if i == j:
                if s <= 1e-12:
                    return None
                rowi[j] = math.sqrt(s)
            else:
                rowi[j] = s / rowj[j]

    y = [0.0] * m
    for i in range(m):
        s = b[i]
        rowi = lower[i]
        for k in range(i):
            s -= rowi[k] * y[k]
        y[i] = s / rowi[i]

    x = [0.0] * m
    for i in range(m - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, m):
            s -= lower[k][i] * x[k]
        x[i] = s / lower[i][i]
    return x


def _refit(model):
    """Re-solve the coefficients from the running sums.

    This is not a second pass over the batches -- the batches are long
    gone.  It reads the eleven-by-eleven table of sums the batches left
    behind, which is why it costs the same after one batch and after ten
    thousand.
    """
    model.dirty = False
    model.cols = []
    model.mu = []
    model.sd = []
    model.coef = []

    n = model.n
    if n <= 0:
        model.base = DEFAULT_SCORE
        return
    model.base = model.sy / n
    if n < MIN_ROWS:
        return                            # too early to trust any slope

    # pick the readings that actually vary, and put them on a z-scale
    use = []
    for pos in range(len(model.xcols)):
        mean = model.sx[pos] / n
        var = _cross(model, pos, pos) / n - mean * mean
        if var > 1e-12:
            use.append((pos, mean, math.sqrt(var)))
    if not use:
        return

    m = len(use)
    ybar = model.base
    a = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for p in range(m):
        pi, mi, si = use[p]
        b[p] = (model.sxy[pi] / n - mi * ybar) / si
        for q in range(p, m):
            qj, mj, sj = use[q]
            c = (_cross(model, pi, qj) / n - mi * mj) / (si * sj)
            a[p][q] = c
            a[q][p] = c

    # ridge: nudge the diagonal, growing the nudge if the system is still
    # too close to singular for a clean solve
    lam = RIDGE
    coef = None
    for _ in range(6):
        for p in range(m):
            a[p][p] += lam
        coef = _cholesky_solve(a, b)
        if coef is not None:
            break
        lam *= 10.0                       # the bump stays in a[][]: cumulative,
        # which is exactly the stronger penalty we want on the retry
    if coef is None:
        return                            # fall back to the running average

    model.cols = [u[0] for u in use]
    model.mu = [u[1] for u in use]
    model.sd = [u[2] for u in use]
    model.coef = coef


# ------------------------------------------------------------------ predict --

def predict(model, row):
    """Predicted Quality+ for one batch, from what the model has seen.

    Early in the season, before there are enough scored batches to read a
    slope from, this is the average score so far (and the middle of the
    scale before even that).  Once the fit is in, it is that average plus
    each reading's departure from normal times what that departure has
    been worth, clipped to the range of scores the panel has actually
    given -- the model has no business predicting a 12.
    """
    if model is None:
        return float(DEFAULT_SCORE)
    if model.dirty:
        _refit(model)

    out = model.base
    if model.coef:
        xcols = model.xcols
        mu = model.mu
        sd = model.sd
        coef = model.coef
        for k, pos in enumerate(model.cols):
            v = step1._value(row, xcols[pos])
            if v is None:
                continue                  # a blank reading says "normal"
            out += coef[k] * (v - mu[k]) / sd[k]

    if model.ymin is not None:
        if out < model.ymin:
            out = model.ymin
        elif out > model.ymax:
            out = model.ymax
    return float(out)


# -------------------------------------------------------------------- worth --

def worth(model, row):
    """True if this batch looks worth sending to the panel.

    The bar is a predicted Quality+ of 6 or more.  One dot product over
    the readings, so it can be answered as the batch comes off the line.
    """
    return predict(model, row) >= WORTH - _EPS


# ---------------------------------------------------------------- CLI --

def _round_half_up(x):
    """Nearest whole score, .5 going up (not banker's rounding)."""
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def _check(model, batches, target):
    """(share rounding to the right score, mean absolute error)."""
    hits = 0
    err = 0.0
    for row in batches:
        actual = step1._value(row, target)
        got = predict(model, row)
        err += abs(got - actual)
        if _round_half_up(got) == _round_half_up(actual):
            hits += 1
    n = len(batches)
    if not n:
        return 0.0, 0.0
    return float(hits) / n, err / n


def main(argv):
    if len(argv) != 2:
        print("usage: python step4.py data.csv")
        return 1

    rows = step1.load_csv(argv[1])
    header = rows[0]
    train, test = step2._split(rows)
    goals, _readings = step1._split_columns(header)
    target = step2._target_index(header, goals)
    if target is None:
        print("no Quality+ column in %s" % argv[1])
        return 1

    scored = [r for r in test[1:] if step1._value(r, target) is not None]
    print("train %d batches, test %d scored batches"
          % (len(train) - 1, len(scored)))

    began = time.time()
    model = start(header)
    marks = []
    for n, row in enumerate(train[1:], 1):
        learn(model, row)
        if n % 100 == 0:
            marks.append((n, _check(model, scored, target)))
    final = _check(model, scored, target)
    spent = time.time() - began

    print("\nlearning as the season comes in (test batches never learned from):")
    print("  %-10s %-10s %s" % ("batches", "right", "mean error"))
    for n, (share, err) in marks:
        print("  %-10d %-10s %.3f" % (n, "%.1f%%" % (100.0 * share), err))
    print("  %-10d %-10s %.3f" % (model.seen(), "%.1f%%" % (100.0 * final[0]),
                                  final[1]))

    sent = 0
    good_sent = 0
    good = 0
    for row in scored:
        actual = step1._value(row, target)
        really_good = actual >= WORTH
        if really_good:
            good += 1
        if worth(model, row):
            sent += 1
            if really_good:
                good_sent += 1

    print("\nsending to the panel (bar: predicted Quality+ %.0f or more):" % WORTH)
    print("  flagged %d of %d batches" % (sent, len(scored)))
    if sent:
        print("  of those flagged, really %.0f+   %.1f%%"
              % (WORTH, 100.0 * good_sent / sent))
    if good:
        print("  of the really %.0f+ batches, caught %.1f%%"
              % (WORTH, 100.0 * good_sent / good))

    print("\n%d learns and %d predictions in %.2f seconds"
          % (model.seen(), (len(marks) + 1) * len(scored), spent))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
