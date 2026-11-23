"""Step 5 of the winemaker's assistant.

The two remaining requests, and this time tasting costs money:

  "The panel can only taste 20 batches this month.  Which 20 should they
   taste to find our best wine?"            -> taste(rows, label, 20)

  "Find me a batch that scores high but stays low in alcohol, while
   tasting as few batches as possible."     -> choose(rows, label, 20)

Input convention (same as steps 1-4, with one change)
-----------------------------------------------------
`rows` is a list of lists.  rows[0] is the header (strings).  A column
name ending in '+' is a goal to maximize, one ending in '-' is a goal to
minimize, anything else is a lab reading.

The change: in this step the goal cells of every data row are '?'.  The
lab readings are on the sheet for all 1279 batches -- they cost nothing,
the lab already measured them -- but nobody knows a batch's Quality+ or
its Alcohol- until the panel sits down with it.  `label(row)` returns
that same batch with its goals filled in, and each call is one tasting.
The budget is a hard cap: label() is called at most `budget` times.

So this is not a prediction problem, it is a shopping problem: of 1279
batches we may open 20, and we want the best wine in the cellar to be
one of the 20.

How the choosing works
----------------------
A batch is judged by its distance to heaven: each goal is put on a 0..1
scale using the spread of the batches tasted so far, 0 being the ideal
end (highest Quality+, lowest Alcohol-), and the two are combined as a
root-mean-square.  Low is good.  That single number is what the search
tries to minimize, and it is why choose() will not hand back a 9-point
wine at 14% alcohol: the alcohol half of the distance punishes it.

The search itself is the standard sequential recipe:

  1. A first look, spread out.  A quarter of the budget goes on a
     farthest-first sample of the z-scored lab sheets, so the first
     tastings cover the cellar instead of five barrels from one corner.

  2. Then, one tasting at a time, a Gaussian process on the z-scored
     readings (squared-exponential kernel, length scale set to the
     median distance between batches) is fitted to the batches tasted so
     far and asked about all the untasted ones.  It answers with a mean
     and an honest spread, and the next batch to open is the one with
     the largest expected improvement over the best tasted so far.

     Expected improvement is used rather than "whichever looks best"
     because with five tastings in hand the surrogate is nearly blind:
     EI will happily spend a tasting on a batch it is merely unsure
     about, which is how the search finds the good corner at all, and it
     tightens into pure exploitation by itself as the budget runs down.

  3. The spread of the goal scales is re-read after every tasting, so a
     first look that happened to find only mediocre wine does not fix
     the yardstick for the rest of the month.

taste() spends the whole budget -- the panel is sitting this month
either way, and the question was which 20 to pour.  choose() is allowed
to stop early, once expected improvement everywhere has gone
negligible, because there the winemaker asked for as few tastings as
possible.  Both share one engine; choose() returns the best batch the
engine actually tasted, never an untasted guess.

Cost
----
Per tasting: one O(n^3) Cholesky on the tasted-by-tasted kernel (n <= 20,
so a few hundred flops) and one O(n^2) triangular solve per untasted
batch.  Kernel values against each new tasting are computed once for
every batch and kept, so nothing is recomputed.  Over 20 tastings on
1279 batches that is a couple of million flops -- a second or two of
Python, well inside the 90 allowed for a split.  Only the Python
standard library is used.
"""

import math
import random
import sys
import time

import step1
import step2

# The panel's capacity for the month.
BUDGET = 20

# Share of the budget spent on the spread-out first look, and the floor
# under it: fewer than four tastings is not enough to fit anything.
WARM_FRACTION = 0.25
WARM_MIN = 4

# Kernel noise: two batches with identical lab sheets still get slightly
# different scores from the panel, so the fit is not forced through
# every point.
NOISE = 0.02
SIGNAL = 1.0            # targets are standardized, so unit signal variance

# choose() may stop once the best expected improvement anywhere is below
# this (in standardized units), but never before MIN_TASTES tastings.
STOP_EI = 1e-3
MIN_TASTES = 10

LS_SAMPLE = 60          # batches sampled to set the kernel length scale
SEED = step2.SEED


# --------------------------------------------------------------- geometry --

def _sqdist(a, b):
    d = 0.0
    for pos, x in enumerate(a):
        diff = x - b[pos]
        d += diff * diff
    return d


def _features(rows):
    """(goals, reading columns, one z-scored point per data row).

    Only the lab readings go in: the goals are '?' everywhere, and even
    once tasted they are the thing being searched for, not a clue to it.
    Readings that never vary are dropped -- they cannot tell two batches
    apart -- and a blank reading becomes 0, the column's own mean.
    """
    header = rows[0]
    data = rows[1:]
    goals, readings = step1._split_columns(header)

    cols = []
    mu = []
    sd = []
    for i in readings:
        values = step1._column(data, i)
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        if var <= 1e-12:
            continue
        cols.append(i)
        mu.append(mean)
        sd.append(math.sqrt(var))

    points = []
    for row in data:
        p = []
        for k, i in enumerate(cols):
            v = step1._value(row, i)
            p.append(0.0 if v is None else (v - mu[k]) / sd[k])
        points.append(tuple(p))
    return goals, cols, points


def _length_scale(points):
    """Median distance between batches: the scale at which they differ.

    Taken over a bounded sample of pairs, so the setup stays cheap.  A
    kernel this wide calls two batches similar when they are closer than
    a typical pair, which is the behaviour wanted from a surrogate that
    has only a handful of tastings to go on.
    """
    n = len(points)
    if n < 2 or not points[0]:
        return 1.0
    idxs = list(range(n))
    if n > LS_SAMPLE:
        random.Random(SEED).shuffle(idxs)
        idxs = idxs[:LS_SAMPLE]
    gaps = []
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            gaps.append(math.sqrt(_sqdist(points[idxs[a]], points[idxs[b]])))
    if not gaps:
        return 1.0
    gaps.sort()
    med = gaps[len(gaps) // 2]
    return med if med > 1e-9 else 1.0


def _spread(points, count):
    """`count` batches that cover the cellar: farthest-first sampling.

    Starts at the most typical batch (nearest the centroid, so the first
    tasting is a representative one) and each time adds the batch
    farthest from everything picked so far.  Deterministic, so the same
    cellar gives the same first look.
    """
    n = len(points)
    if n == 0 or count <= 0:
        return []
    count = min(count, n)
    width = len(points[0])

    centre = [0.0] * width
    for p in points:
        for k, x in enumerate(p):
            centre[k] += x
    centre = tuple(x / n for x in centre)

    first = 0
    closest = None
    for i, p in enumerate(points):
        d = _sqdist(p, centre)
        if closest is None or d < closest:
            closest = d
            first = i

    used = [False] * n
    used[first] = True
    chosen = [first]
    gaps = [_sqdist(points[i], points[first]) for i in range(n)]

    while len(chosen) < count:
        pick = -1
        best = -1.0
        for i in range(n):
            if used[i]:
                continue
            if gaps[i] > best:
                best = gaps[i]
                pick = i
        if pick < 0:
            break
        used[pick] = True
        chosen.append(pick)
        p = points[pick]
        for i in range(n):
            d = _sqdist(points[i], p)
            if d < gaps[i]:
                gaps[i] = d
    return chosen


# ------------------------------------------------------- distance to heaven --

def _ranges(labelled, goals):
    """The low and high of each goal among the batches tasted so far."""
    out = []
    for i, _w in goals:
        values = []
        for row in labelled:
            v = step1._value(row, i)
            if v is not None:
                values.append(v)
        out.append((min(values), max(values)) if values else (None, None))
    return out


def _d2h(row, goals, lohi):
    """How far this tasted batch is from the perfect batch, in 0..1.

    Each goal contributes its own distance from the ideal end of the
    range seen so far -- high Quality+ is good, low Alcohol- is good --
    and the goals are combined root-mean-square, so a batch has to do
    well on both.  0 is heaven; 1 is the worst corner.
    """
    total = 0.0
    count = 0
    for pos, (i, w) in enumerate(goals):
        v = step1._value(row, i)
        lo, hi = lohi[pos]
        if v is None or hi is None:
            continue
        span = hi - lo
        if span <= 0.0:
            d = 0.0                       # one value seen: no spread to judge
        elif w > 0:
            d = (hi - v) / span           # maximizing: the top is heaven
        else:
            d = (v - lo) / span           # minimizing: the bottom is heaven
        if d < 0.0:
            d = 0.0
        elif d > 1.0:
            d = 1.0
        total += d * d
        count += 1
    if not count:
        return 1.0                        # nothing known: assume the worst
    return math.sqrt(total / count)


# ------------------------------------------------------------- surrogate --

def _chol(a):
    """Lower Cholesky factor of a symmetric matrix; None if not definite."""
    m = len(a)
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
    return lower


def _forward(lower, b):
    """Solve L v = b for a lower-triangular L."""
    m = len(b)
    v = [0.0] * m
    for i in range(m):
        s = b[i]
        rowi = lower[i]
        for k in range(i):
            s -= rowi[k] * v[k]
        v[i] = s / rowi[i]
    return v


def _cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _pdf(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _farthest(kcols, done, n):
    """The untasted batch least like anything tasted: pure exploration.

    Used when the tasted batches are all alike, or when the kernel
    matrix will not factor -- cases where expected improvement has
    nothing to say and another look somewhere new is the honest move.
    """
    pick = None
    best = None
    for j in range(len(done)):
        if done[j]:
            continue
        near = max(kcols[j][:n]) if n else 0.0
        if best is None or near < best:
            best = near
            pick = j
    return pick


def _next(kcols, done, tasted, tasted_idx, goals):
    """(index of the next batch to taste, its expected improvement).

    Fits the Gaussian process to the tastings so far and scores every
    untasted batch by how much it is expected to beat the best tasting
    yet.  A batch the surrogate thinks is merely promising and a batch
    it is simply unsure about both score well, which is what keeps the
    search from settling into the first decent corner it finds.
    """
    n = len(tasted_idx)
    if n == 0:
        return _farthest(kcols, done, 0), 0.0

    lohi = _ranges(tasted, goals)
    ys = [_d2h(row, goals, lohi) for row in tasted]
    mean = sum(ys) / n
    var = sum((y - mean) ** 2 for y in ys) / n
    sd = math.sqrt(var) if var > 0.0 else 0.0
    if sd <= 1e-12:
        # every tasting so far scored the same: nothing to interpolate
        return _farthest(kcols, done, n), 0.0
    z = [(y - mean) / sd for y in ys]

    big = []
    for a, ia in enumerate(tasted_idx):
        rowk = list(kcols[ia][:n])
        rowk[a] += NOISE
        big.append(rowk)
    lower = _chol(big)
    if lower is None:
        return _farthest(kcols, done, n), 0.0

    u = _forward(lower, z)
    floor = min(z)                        # the best tasting so far

    pick = None
    best = -1.0
    for j in range(len(done)):
        if done[j]:
            continue
        v = _forward(lower, kcols[j][:n])
        mu = 0.0
        vv = SIGNAL
        for pos, x in enumerate(v):
            mu += x * u[pos]
            vv -= x * x
        if vv < 1e-12:
            vv = 1e-12
        spread = math.sqrt(vv)
        gain = floor - mu
        t = gain / spread
        ei = gain * _cdf(t) + spread * _pdf(t)
        if ei > best:
            best = ei
            pick = j
    if pick is None:
        return None, 0.0
    return pick, best


# ---------------------------------------------------------------- engine --

def _search(rows, label, budget, stop_early):
    """(batches tasted in order, the best of them).

    Shared by taste() and choose(); the only difference is whether the
    search is allowed to hand the budget back unspent.
    """
    if not rows or len(rows) < 2:
        return [], None

    data = list(rows[1:])
    goals, _cols, points = _features(rows)

    budget = int(budget)
    if budget > len(data):
        budget = len(data)
    if budget < 1:
        # nothing may be tasted: hand back the most typical batch as it is
        spread = _spread(points, 1)
        return [], (data[spread[0]] if spread else data[0])

    ell = _length_scale(points)
    twoll = 2.0 * ell * ell

    tasted = []
    tasted_idx = []
    done = [False] * len(data)
    kcols = [[] for _ in range(len(data))]

    def take(idx):
        """Spend one tasting on batch `idx` and fold it into the surrogate."""
        got = label(data[idx])
        if got is None:
            got = data[idx]               # a label() that fills in place
        done[idx] = True
        tasted_idx.append(idx)
        tasted.append(got)
        p = points[idx]
        for j in range(len(data)):
            kcols[j].append(math.exp(-_sqdist(points[j], p) / twoll))

    warm = max(WARM_MIN, int(round(budget * WARM_FRACTION)))
    warm = min(warm, budget)

    if not goals:
        # no goal columns to search for: a spread-out sample is the most
        # informative set of batches we can name
        for idx in _spread(points, budget):
            take(idx)
        return tasted, (tasted[0] if tasted else None)

    for idx in _spread(points, warm):
        take(idx)

    while len(tasted) < budget:
        pick, ei = _next(kcols, done, tasted, tasted_idx, goals)
        if pick is None:
            break
        if stop_early and len(tasted) >= MIN_TASTES and ei < STOP_EI:
            break                         # no tasting left worth paying for
        take(pick)

    return tasted, _best(tasted, goals)


def _best(tasted, goals):
    """The tasted batch nearest heaven, judged on the full tasting record."""
    if not tasted:
        return None
    if not goals:
        return tasted[0]
    lohi = _ranges(tasted, goals)
    best = tasted[0]
    score = _d2h(best, goals, lohi)
    for row in tasted[1:]:
        d = _d2h(row, goals, lohi)
        if d < score:                     # ties keep the earlier tasting
            score = d
            best = row
    return best


# ----------------------------------------------------------------- taste --

def taste(rows, label, budget=BUDGET):
    """The batches to pour for the panel this month, in the order tasted.

    At most `budget` of them, each returned with its goals filled in.
    The first few cover the cellar; after that each choice is the batch
    with the most to tell us about where the best wine is, given
    everything poured so far.  The whole budget is spent: the panel is
    sitting either way, so an unpoured glass is information thrown out.
    """
    tasted, _best_row = _search(rows, label, budget, stop_early=False)
    return tasted


# ---------------------------------------------------------------- choose --

def choose(rows, label, budget=BUDGET):
    """The best batch found: high Quality+ with low Alcohol-.

    Same search as taste(), but it stops as soon as no untasted batch is
    expected to beat the best one found -- the winemaker asked for as
    few tastings as possible -- and it returns the batch itself, tasted
    and scored, rather than a batch the model merely hopes about.
    """
    _tasted, best = _search(rows, label, budget, stop_early=True)
    return best


# ------------------------------------------------------------------- CLI --

def _mask(rows):
    """A copy of the data with every goal cell set to '?', plus the truth.

    This is the step-5 situation made out of a finished dataset: the lab
    sheets stay visible, the panel's verdicts are taken away, and a
    label() hands one back at a time.
    """
    header = [str(s).strip() for s in rows[0]]
    goals, _readings = step1._split_columns(header)
    hidden = set(i for i, _w in goals)

    masked = [header]
    truth = {}
    for row in rows[1:]:
        copy = ["?" if i in hidden else v for i, v in enumerate(row)]
        masked.append(copy)
        truth[id(copy)] = list(row)
    return masked, truth, goals


def _labeller(truth, tally):
    """A label() that costs one tasting per call."""
    def label(row):
        tally[0] += 1
        full = truth.get(id(row))
        if full is None:
            return row
        for i, v in enumerate(full):
            row[i] = v                    # the same batch, goals filled in
        return row
    return label


def _cellar_scores(rows, goals):
    """Every batch's true distance to heaven, on the whole cellar's scale."""
    data = rows[1:]
    lohi = _ranges(data, goals)
    return [_d2h(row, goals, lohi) for row in data], lohi


def main(argv):
    if len(argv) != 2:
        print("usage: python step5.py data.csv")
        return 1

    rows = step1.load_csv(argv[1])
    header = rows[0]
    train, _test = step2._split(rows)
    goals, _readings = step1._split_columns(header)
    if not goals:
        print("no goal columns ('+' / '-') in %s" % argv[1])
        return 1

    truths, _lohi = _cellar_scores(train, goals)
    order = sorted(range(len(truths)), key=lambda k: truths[k])
    perfect = truths[order[0]]
    print("cellar: %d batches, %d of them tasteable this month"
          % (len(train) - 1, BUDGET))
    print("best batch in the whole cellar: distance to heaven %.3f" % perfect)

    def report(name, tasted, chosen, used, spent):
        print("\n%s: %d tastings in %.2f seconds" % (name, used, spent))
        if not tasted:
            return
        lohi = _ranges([truth for truth in train[1:]], goals)
        found = min(_d2h(r, goals, lohi) for r in tasted)
        rank = sum(1 for t in truths if t < found)
        print("  best batch poured: distance to heaven %.3f" % found)
        print("  it is batch %d of %d in the cellar (top %.1f%%)"
              % (rank + 1, len(truths), 100.0 * (rank + 1) / len(truths)))
        if chosen is not None:
            print("  returned batch: " + ", ".join(
                "%s=%s" % (h, chosen[i]) for i, h in enumerate(header)
                if any(i == g for g, _w in goals)))

    masked, truth, _g = _mask(train)
    tally = [0]
    began = time.time()
    poured = taste(masked, _labeller(truth, tally), BUDGET)
    report("taste()", poured, None, tally[0], time.time() - began)

    masked, truth, _g = _mask(train)
    tally = [0]
    began = time.time()
    picked = choose(masked, _labeller(truth, tally), BUDGET)
    report("choose()", [picked] if picked is not None else [], picked,
           tally[0], time.time() - began)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
