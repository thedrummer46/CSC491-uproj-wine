"""Step 6 of the winemaker's assistant.

Two questions about what the lab sheet is telling us:

  "Which two or three readings matter most for whether a batch scores
   well?"                                            -> blame(train, 3)

  "Our low-scoring batches: which lab readings are dragging them
   down?"                                            -> dragging(train)

Input convention (same as steps 1-5)
------------------------------------
`train` is a list of lists.  train[0] is the header (strings); every
later row holds numbers.  A column name ending in '+' is a goal to
maximize, one ending in '-' is a goal to minimize, anything else is a
lab reading.

Both answers name lab readings only.  A goal column is never returned:
'Quality+' is the thing being explained and 'Alcohol-' is judged
alongside it, so neither is an explanation of anything -- blaming the
score on the score would tell the cellar nothing it can act on.

How blame() decides
-------------------
Every column (readings and the score) is put on a rank scale first, so
one stuck sensor or a single freak barrel cannot create a slope, and a
relationship that bends but never turns around still counts.  Then the
readings are chosen one at a time: the reading that explains most of
the score on its own, then the reading that adds most on top of it, and
so on.

That greedy order, rather than a plain list of the strongest
correlations, is the honest answer to "which two or three matter".
Several readings here move together; sorted by correlation alone the
top three can easily be three views of one fact, and the winemaker
would walk away thinking three dials matter when only one does.
Picking the next reading by what it adds keeps the shortlist from
repeating itself.

How dragging() decides
----------------------
The poor batches (Quality+ below 6) are held up against the rest, one
reading at a time, and compared by rank -- the chance that a poor batch
reads higher than a good one.  A reading is flagged only when that
contrast is both large enough to matter (an effect size past the usual
'small' bar) and past a 95% test, because with 1279 batches almost
every reading differs by a hair and a list of eleven complaints is not
a list of complaints.  'high' means the poor batches run above the good
ones on that reading, 'low' means below; worst gap first.

This is a contrast, not a cause: it says where the poor batches sit
differently, which is what the question asked, and the readings that
also appear in blame() are the ones most worth acting on.

Cost
----
One pass over the training rows per column to build the rank scales and
the 11-by-11 table of cross-products (~80k multiply-adds), then 66
solves of a system no bigger than 11-by-11 for the full greedy order,
and one sort per column for the rank contrast.  Both answers come back
in well under a second on 1279 rows.  Only the Python standard library
is used.
"""

import math
import sys

import step1
import step2
import step4

# A batch scoring below this was not good enough; those are the ones the
# winemaker is asking about.
POOR_BELOW = 6.0

# Ridge penalty on the correlation scale, so the greedy fit stays stable
# when two readings are near-copies of each other.
RIDGE = 0.01

# Fewer scored batches than this and nothing here is worth reading.
MIN_ROWS = 10

# A flagged reading has to clear both bars: a rank effect size of at
# least this (the conventional 'small but real' line) and a two-sided
# 95% test.
MIN_DELTA = 0.15
MIN_Z = 1.96


# ------------------------------------------------------------- rank scales --

def _rank_column(data, i):
    """Each row's percentile position in column i, in 0..1.

    Ties share a position, and a missing or unusable cell becomes 0.5 --
    the middle of the column, which pulls the answer in no direction.
    """
    scale = step1._rank_scale(step1._column(data, i))
    out = []
    for row in data:
        v = step1._value(row, i)
        p = scale.get(v) if v is not None else None
        out.append(0.5 if p is None else p)
    return out


def _standardize(values):
    """Values with mean 0 and spread 1, or None if the column never varies."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    if var <= 1e-12:
        return None
    sd = math.sqrt(var)
    return [(v - mean) / sd for v in values]


# ------------------------------------------------------------------ frame --

def _frame(train):
    """Everything blame() needs, or None if the question cannot be answered.

    Returns a dict with the reading names, the correlation matrix of the
    rank-scaled readings, and each reading's correlation with the
    rank-scaled score.
    """
    if not train or len(train) < 2:
        return None

    header = [str(s).strip() for s in train[0]]
    goals, readings = step1._split_columns(header)
    target = step2._target_index(header, goals)
    if target is None or not readings:
        return None

    rows = [r for r in train[1:] if step1._value(r, target) is not None]
    if len(rows) < MIN_ROWS:
        return None

    y = _standardize(_rank_column(rows, target))
    if y is None:
        return None                       # every batch scored the same

    names = []
    xs = []
    for i in readings:
        z = _standardize(_rank_column(rows, i))
        if z is None:
            continue                      # a constant reading explains nothing
        names.append(header[i])
        xs.append(z)
    if not names:
        return None

    n = len(y)
    d = len(xs)
    cross = [[0.0] * d for _ in range(d)]
    against = [0.0] * d
    for p in range(d):
        xp = xs[p]
        against[p] = sum(xp[t] * y[t] for t in range(n)) / n
        for q in range(p, d):
            xq = xs[q]
            c = sum(xp[t] * xq[t] for t in range(n)) / n
            cross[p][q] = c
            cross[q][p] = c

    return {"names": names, "cross": cross, "against": against, "rows": n}


def _explained(frame, subset):
    """Share of the score's variation a set of readings accounts for.

    A ridge-shrunk least-squares fit on the rank scale; because both
    sides are standardized, the fit's explained variance is just the
    coefficients read against the correlations.
    """
    if not subset:
        return 0.0
    cross = frame["cross"]
    against = frame["against"]
    m = len(subset)
    b = [against[p] for p in subset]

    lam = RIDGE
    coef = None
    for _ in range(6):
        a = [[cross[p][q] + (lam if p == q else 0.0) for q in subset]
             for p in subset]
        coef = step4._cholesky_solve(a, b)
        if coef is not None:
            break
        lam *= 10.0                       # still singular: lean harder
    if coef is None:
        return 0.0
    total = sum(coef[t] * b[t] for t in range(m))
    if total < 0.0:
        total = 0.0
    elif total > 1.0:
        total = 1.0
    return total


def _ordered(frame):
    """All readings, most important first, as [(name, what it added), ...].

    Each step takes the reading that adds most to what the readings
    already picked explain, so the order reads as a shortlist rather
    than as a pile of restatements of the same fact.
    """
    names = frame["names"]
    left = list(range(len(names)))
    chosen = []
    out = []
    base = 0.0
    while left:
        pick = None
        gain = None
        for p in left:
            got = _explained(frame, chosen + [p]) - base
            if gain is None or got > gain + 1e-12:
                gain = got
                pick = p                  # header order breaks exact ties
        chosen.append(pick)
        left.remove(pick)
        base += gain
        out.append((names[pick], gain))
    return out


# ------------------------------------------------------------------ blame --

def blame(train, k=3):
    """The k lab readings that matter most for Quality+, most important first.

    Only reading columns are named, never a goal.  Fewer than k come back
    only when the cellar has fewer usable readings than that, or when
    there is nothing to go on at all (no scores, or every batch scored
    the same), in which case the answer is empty.
    """
    k = int(k)
    if k <= 0:
        return []
    frame = _frame(train)
    if frame is None:
        return []
    return [name for name, _gain in _ordered(frame)[:k]]


# -------------------------------------------------------------- contrast --

def _delta(poor, good):
    """(rank effect size in -1..1, normal score) for poor against good.

    The effect size is the chance a poor batch reads higher than a good
    one minus the chance it reads lower, computed from mid-ranks so that
    repeated readings are handled properly.  Positive means the poor
    batches run high on this reading.
    """
    n1 = len(poor)
    n2 = len(good)
    if n1 < 1 or n2 < 1:
        return 0.0, 0.0

    merged = [(v, 0) for v in poor] + [(v, 1) for v in good]
    merged.sort(key=lambda pair: pair[0])
    n = n1 + n2

    rank_sum = 0.0
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and merged[j + 1][0] == merged[i][0]:
            j += 1
        size = j - i + 1
        mid = (i + j) / 2.0 + 1.0         # shared 1-based rank of this run
        for t in range(i, j + 1):
            if merged[t][1] == 0:
                rank_sum += mid
        tie_term += size * size * size - size
        i = j + 1

    u = rank_sum - n1 * (n1 + 1) / 2.0
    delta = 2.0 * u / (n1 * n2) - 1.0

    var = 0.0
    if n > 1:
        var = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1.0)))
    z = (u - n1 * n2 / 2.0) / math.sqrt(var) if var > 0.0 else 0.0
    return delta, z


def _split_by_score(train):
    """(header, readings, poor rows, good rows) or None if there is no contrast.

    Poor means Quality+ below 6.  If the cellar is all poor or all good
    there is nothing to hold up against anything, so the split falls back
    to the median score; if even that leaves one side empty, there is no
    question to answer.
    """
    if not train or len(train) < 2:
        return None

    header = [str(s).strip() for s in train[0]]
    goals, readings = step1._split_columns(header)
    target = step2._target_index(header, goals)
    if target is None or not readings:
        return None

    pairs = []
    for row in train[1:]:
        y = step1._value(row, target)
        if y is not None:
            pairs.append((row, y))
    if len(pairs) < MIN_ROWS:
        return None

    poor = [row for row, y in pairs if y < POOR_BELOW]
    good = [row for row, y in pairs if y >= POOR_BELOW]
    if not poor or not good:
        mid = step1._median([y for _row, y in pairs])
        poor = [row for row, y in pairs if y < mid]
        good = [row for row, y in pairs if y >= mid]
    if not poor or not good:
        return None
    return header, readings, poor, good


def _contrast(train):
    """Every reading, worst gap first, as

        [(name, 'high'/'low', effect size, normal score,
          typical in poor, typical in good, flagged), ...]

    `flagged` marks the ones dragging() reports: a gap big enough to
    matter and unlikely enough to be chance.
    """
    split = _split_by_score(train)
    if split is None:
        return []
    header, readings, poor, good = split

    out = []
    for i in readings:
        lo = step1._column(poor, i)
        hi = step1._column(good, i)
        if not lo or not hi:
            continue
        delta, z = _delta(lo, hi)
        direction = "high" if delta > 0.0 else "low"
        flagged = abs(delta) >= MIN_DELTA and abs(z) >= MIN_Z
        out.append((header[i], direction, delta, z,
                    step1._median(lo), step1._median(hi), flagged))

    # worst first; header order breaks ties, so the answer is deterministic
    order = {name: pos for pos, name in enumerate(header)}
    out.sort(key=lambda rec: (-abs(rec[2]), order.get(rec[0], 0)))
    return out


# ----------------------------------------------------------------- dragging --

def dragging(train):
    """Readings that sit wrong in the poor batches, worst first.

    Each entry is (column name, 'high' or 'low'): 'high' means the
    batches scoring below 6 run above the better batches on that
    reading, 'low' means they run below it.  Only lab readings are
    named, never a goal column, and only readings whose gap is both
    sizeable and unlikely to be chance -- so an empty list is a real
    answer, meaning the poor batches do not stand apart on any single
    reading.
    """
    return [(name, direction)
            for name, direction, _d, _z, _plo, _phi, flagged in _contrast(train)
            if flagged]


# --------------------------------------------------------------------- CLI --

def main(argv):
    if len(argv) != 2:
        print("usage: python step6.py data.csv")
        return 1

    rows = step1.load_csv(argv[1])
    train, _test = step2._split(rows)
    print("train %d batches" % (len(train) - 1))

    frame = _frame(train)
    top = blame(train, 3)
    print("\nreadings that matter most for the score:")
    if not frame:
        print("  (nothing to go on)")
    else:
        running = []
        for pos, (name, gain) in enumerate(_ordered(frame), 1):
            running.append(name)
            mark = "*" if name in top else " "
            print("  %s %d. %-18s adds %.3f, together %.3f"
                  % (mark, pos, name, gain, _explained(
                      frame, [frame["names"].index(h) for h in running])))
        print("\n  blame(train, 3) -> %s" % ", ".join(top))

    print("\npoor batches (Quality+ below %.0f) against the rest:" % POOR_BELOW)
    records = _contrast(train)
    if not records:
        print("  (no contrast to draw)")
    else:
        print("  %-18s %-6s %-7s %-9s %-9s" %
              ("reading", "runs", "effect", "poor", "good"))
        for name, direction, delta, _z, plo, phi, flagged in records:
            print("  %-18s %-6s %+.3f  %-9s %-9s %s"
                  % (name, direction, delta, plo, phi,
                     "<- dragging" if flagged else ""))
        flagged = dragging(train)
        print("\n  dragging(train) -> %s" % (
            ", ".join("%s too %s" % (n, d) for n, d in flagged) or "nothing"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
