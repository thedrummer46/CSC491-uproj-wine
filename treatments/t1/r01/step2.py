"""Step 2 of the winemaker's assistant.

Two more questions from the cellar:

  "Before the panel sits, estimate what score this new batch will get
   from its lab sheet alone."                        -> guess(train, row)

  "Group our batches into a few kinds, and tell me what score to expect
   from each kind."                                  -> kinds(train)

Input convention (same as step 1)
---------------------------------
`train` is a list of lists.  train[0] is the header (strings); every
later row holds numbers.  A column name ending in '+' is a goal to
maximize, one ending in '-' is a goal to minimize, anything else is a
lab reading.

What counts as "the lab sheet"
------------------------------
Only the plain reading columns are used to predict.  The goal columns
('Quality+', 'Alcohol-') are outcomes, not inputs: the panel's score is
what we are guessing, and the other goals are judged alongside it, so
feeding them in would be answering the question with the answer.

How it works
------------
guess  -- nearest-neighbour averaging.  Readings are put on a common
          z-score scale and each reading is weighted by how strongly it
          tracks Quality+ in the training rows, so a reading that says
          nothing about the score barely moves the distance.  The score
          comes back as a distance-weighted average of the k closest
          past batches.  Nothing is fitted beyond column statistics, so
          a small cellar does not get an overconfident straight line.

kinds  -- k-means on the z-scored readings, k picked from 2..6 by the
          silhouette score, then each group reports the mean Quality+ of
          its own batches.  Groups come back best-expected-score first.

Cost
----
The per-train setup (column statistics, correlations) happens once and
is cached on the train list, so 320 calls to guess() cost one scan of
the training rows each: milliseconds in total.  kinds() runs a handful
of bounded k-means fits and scores them on a sample of at most 300
rows, which is well inside a second for a few thousand batches.  Only
the Python standard library is used.
"""

import math
import random
import sys

import step1

# --- guess() ---------------------------------------------------------------
KNN_MIN = 3             # never average fewer than this many neighbours
KNN_MAX = 12            # ... nor more, or distinct batches blur together
WEIGHT_FLOOR = 0.05     # a reading with zero correlation still counts a little

# --- kinds() ---------------------------------------------------------------
MIN_KINDS = 2
MAX_KINDS = 6           # "a few kinds": a winemaker has to act on these
KMEANS_RESTARTS = 4
KMEANS_ITERS = 50
SILHOUETTE_SAMPLE = 300
SEED = 1                # fixed, so the same cellar gives the same groups


# ------------------------------------------------------------------ model --

class _Model(object):
    """Everything derived from one set of training rows."""

    __slots__ = ("xcols", "mean", "span", "weight", "points", "ys",
                 "rows", "fallback", "k")


def _pearson(data, i, j):
    """Correlation between columns i and j over rows where both are usable."""
    xs = []
    ys = []
    for row in data:
        a = step1._value(row, i)
        b = step1._value(row, j)
        if a is None or b is None:
            continue
        xs.append(a)
        ys.append(b)
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sxx = syy = 0.0
    for a, b in zip(xs, ys):
        da = a - mx
        db = b - my
        sxy += da * db
        sxx += da * da
        syy += db * db
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def _target_index(header, goals):
    """Index of the Quality+ column (the score the panel will give)."""
    for i, raw in enumerate(header):
        if str(raw).strip().lower() == "quality+":
            return i
    # no column by that name: fall back to the last column to maximize
    best = None
    for i, w in goals:
        if w > 0:
            best = i
    return best


def _build(train):
    """Column statistics, reading weights and the neighbour table."""
    model = _Model()
    model.xcols = []
    model.mean = []
    model.span = []
    model.weight = []
    model.points = []
    model.ys = []
    model.rows = []
    model.fallback = 0.0
    model.k = 1

    if not train or len(train) < 2:
        return model

    header = train[0]
    data = list(train[1:])
    goals, readings = step1._split_columns(header)
    target = _target_index(header, goals)
    if target is None:
        return model

    ys = step1._column(data, target)
    model.fallback = step1._median(ys) if ys else 0.0

    # One z-score scale per reading column.  Constant columns are dropped:
    # they cannot tell two batches apart.
    for i in readings:
        values = step1._column(data, i)
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        span = math.sqrt(var)
        if span <= 0.0:
            continue
        model.xcols.append(i)
        model.mean.append(mean)
        model.span.append(span)
        # how much this reading tracks the score the panel gives
        model.weight.append(abs(_pearson(data, i, target)) + WEIGHT_FLOOR)

    if not model.xcols:
        return model

    for row in data:
        y = step1._value(row, target)
        if y is None:
            continue                      # an unscored batch teaches nothing
        model.points.append(_encode(model, row))
        model.ys.append(y)
        model.rows.append(row)

    n = len(model.points)
    if n:
        k = int(round(math.sqrt(n) / 2.0))
        model.k = max(KNN_MIN, min(KNN_MAX, max(1, k), n))
    return model


def _encode(model, row):
    """One batch as a tuple of z-scores; a missing reading becomes 0 (mean)."""
    out = []
    for pos, i in enumerate(model.xcols):
        v = step1._value(row, i)
        if v is None:
            out.append(0.0)
        else:
            out.append((v - model.mean[pos]) / model.span[pos])
    return tuple(out)


# A tiny cache so 320 calls to guess() with the same train list do the
# setup once.  Keyed on the list's identity and length, and the list
# itself is held, so a different or re-read split rebuilds.
_CACHE = {}


def _model(train):
    key = id(train)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is train and hit[1] == len(train):
        return hit[2]
    if len(_CACHE) > 4:
        _CACHE.clear()
    built = _build(train)
    _CACHE[key] = (train, len(train), built)
    return built


# ------------------------------------------------------------------ guess --

def guess(train, row):
    """Predicted Quality+ for one new batch, from its lab readings alone.

    The k most similar past batches vote, each weighted by 1/distance, so
    a near-twin of this batch counts for more than a vaguely similar one.
    With no usable training rows the answer is the typical past score.
    """
    model = _model(train)
    if not model.points:
        return float(model.fallback)

    want = _encode(model, row)
    weight = model.weight

    # distance to every past batch, keeping only the k smallest
    best = []                                  # [(distance^2, index), ...]
    worst = float("inf")
    k = model.k
    for idx, point in enumerate(model.points):
        d = 0.0
        for pos, a in enumerate(point):
            diff = want[pos] - a
            d += weight[pos] * diff * diff
            if d >= worst:
                break                          # cannot make the shortlist
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

    total = 0.0
    mass = 0.0
    for d, idx in best[:k]:
        w = 1.0 / (math.sqrt(d) + 1e-9)
        total += w * model.ys[idx]
        mass += w
    if mass <= 0.0:
        return float(model.fallback)
    return float(total / mass)


# ------------------------------------------------------------------ kinds --

def _sqdist(a, b):
    d = 0.0
    for pos, x in enumerate(a):
        diff = x - b[pos]
        d += diff * diff
    return d


def _init_centers(points, k, rng):
    """k-means++ : spread the first centres out instead of guessing blind."""
    centers = [points[rng.randrange(len(points))]]
    while len(centers) < k:
        gaps = [min(_sqdist(p, c) for c in centers) for p in points]
        total = sum(gaps)
        if total <= 0.0:
            break                              # every point already covered
        mark = rng.random() * total
        run = 0.0
        pick = points[-1]
        for p, g in zip(points, gaps):
            run += g
            if run >= mark:
                pick = p
                break
        centers.append(pick)
    return centers


def _kmeans(points, k, rng):
    """Return (assignment, within-cluster sum of squares)."""
    centers = _init_centers(points, k, rng)
    k = len(centers)
    width = len(points[0])
    assign = [0] * len(points)

    for _ in range(KMEANS_ITERS):
        moved = False
        for idx, p in enumerate(points):
            best = 0
            bestd = _sqdist(p, centers[0])
            for c in range(1, k):
                d = _sqdist(p, centers[c])
                if d < bestd:
                    bestd = d
                    best = c
            if assign[idx] != best:
                assign[idx] = best
                moved = True

        sums = [[0.0] * width for _ in range(k)]
        counts = [0] * k
        for idx, p in enumerate(points):
            c = assign[idx]
            counts[c] += 1
            row = sums[c]
            for pos, x in enumerate(p):
                row[pos] += x
        for c in range(k):
            if counts[c]:
                centers[c] = tuple(x / counts[c] for x in sums[c])
            else:
                # an empty group: restart it on a random batch
                centers[c] = points[rng.randrange(len(points))]
                moved = True
        if not moved:
            break

    sse = sum(_sqdist(p, centers[assign[i]]) for i, p in enumerate(points))
    return assign, sse


def _silhouette(points, assign, k, rng):
    """How cleanly separated the groups are, on a sample, in -1..1."""
    n = len(points)
    idxs = list(range(n))
    if n > SILHOUETTE_SAMPLE:
        rng.shuffle(idxs)
        idxs = idxs[:SILHOUETTE_SAMPLE]

    groups = [[] for _ in range(k)]
    for i in idxs:
        groups[assign[i]].append(i)
    if sum(1 for g in groups if g) < 2:
        return -1.0

    total = 0.0
    counted = 0
    for i in idxs:
        own = assign[i]
        if len(groups[own]) < 2:
            continue
        inside = 0.0
        for j in groups[own]:
            if j != i:
                inside += math.sqrt(_sqdist(points[i], points[j]))
        inside /= len(groups[own]) - 1

        outside = None
        for c in range(k):
            if c == own or not groups[c]:
                continue
            d = sum(math.sqrt(_sqdist(points[i], points[j]))
                    for j in groups[c]) / len(groups[c])
            if outside is None or d < outside:
                outside = d
        if outside is None:
            continue
        spread = max(inside, outside)
        if spread > 0.0:
            total += (outside - inside) / spread
        counted += 1
    return total / counted if counted else -1.0


def kinds(train):
    """Group the batches into a few kinds and score each kind.

    Returns a list of (group_rows, expected_quality) pairs, best expected
    score first.  `group_rows` holds the training rows of that kind (the
    header is not included) and `expected_quality` is the mean Quality+
    of those rows -- what to expect from the next batch of that kind.
    """
    model = _model(train)
    points = model.points
    if not points:
        return []
    if len(points) < MIN_KINDS * 2:
        # too small a cellar to split: one kind, which is the whole cellar
        return [(list(model.rows),
                 float(sum(model.ys) / len(model.ys)))]

    distinct = len(set(points))
    top = max(MIN_KINDS, min(MAX_KINDS, distinct, len(points) // 2))

    best_assign = None
    best_k = 1
    best_score = None
    for k in range(MIN_KINDS, top + 1):
        rng = random.Random(SEED + k)
        fit_assign = None
        fit_sse = None
        for _ in range(KMEANS_RESTARTS):
            assign, sse = _kmeans(points, k, rng)
            if fit_sse is None or sse < fit_sse:
                fit_sse = sse
                fit_assign = assign
        score = _silhouette(points, fit_assign, k, random.Random(SEED))
        if best_score is None or score > best_score:
            best_score = score
            best_assign = fit_assign
            best_k = k

    if best_assign is None:
        return [(list(model.rows),
                 float(sum(model.ys) / len(model.ys)))]

    buckets = [[] for _ in range(best_k)]
    scores = [[] for _ in range(best_k)]
    for idx, c in enumerate(best_assign):
        buckets[c].append(model.rows[idx])
        scores[c].append(model.ys[idx])

    out = []
    for c in range(best_k):
        if not buckets[c]:
            continue
        out.append((buckets[c], float(sum(scores[c]) / len(scores[c]))))
    out.sort(key=lambda pair: -pair[1])
    return out


# -------------------------------------------------------------------- CLI --

def _split(rows, seed=SEED, hold=0.2):
    """Shuffle the data rows into (train, test), header kept on both."""
    header = rows[0]
    data = list(rows[1:])
    random.Random(seed).shuffle(data)
    cut = int(len(data) * (1.0 - hold))
    return [header] + data[:cut], [header] + data[cut:]


def main(argv):
    if len(argv) != 2:
        print("usage: python step2.py data.csv")
        return 1

    rows = step1.load_csv(argv[1])
    header = rows[0]
    train, test = _split(rows)
    goals, _readings = step1._split_columns(header)
    target = _target_index(header, goals)
    if target is None:
        print("no Quality+ column in %s" % argv[1])
        return 1

    print("train %d batches, test %d batches" % (len(train) - 1, len(test) - 1))

    err = 0.0
    base = 0.0
    n = 0
    typical = step1._median(step1._column(train[1:], target))
    for row in test[1:]:
        actual = step1._value(row, target)
        if actual is None:
            continue
        err += abs(guess(train, row) - actual)
        base += abs(typical - actual)
        n += 1
    if n:
        print("\nguessing the panel's score (mean error over %d batches):" % n)
        print("  from the lab sheet   %.3f" % (err / n))
        print("  always say %-8s %.3f" % (typical, base / n))

    groups = kinds(train)
    print("\n%d kinds of batch in the cellar:" % len(groups))
    for num, (group, expected) in enumerate(groups, 1):
        print("\n  kind %d: %d batches, expect Quality+ %.2f"
              % (num, len(group), expected))
        picture = step1.summary([header] + group)
        for name, value in picture.items():
            print("    %-18s %s" % (name, value))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
