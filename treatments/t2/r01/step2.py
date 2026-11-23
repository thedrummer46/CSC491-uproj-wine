"""Step 2 of the winemaker's assistant.

Answers two more requests about the cellar:

  guess(train, row) -- "before the panel sits, estimate what score this new
                       batch will get from its lab sheet alone"
  kinds(train)      -- "group our batches into a few kinds, and tell me what
                       score to expect from each kind"

Both lean on step 1 for reading the header row and for the small numeric
helpers, so the meaning of ``+``/``-`` column markers is defined in exactly
one place.

The score being predicted is the goal column the panel awards (``Quality+``).
Every other column -- the plain lab readings and any other declared goal,
such as ``Alcohol-`` -- is something the lab sheet already shows before the
panel sits, so all of them are used as evidence.

Method: nearest past batches.  A new batch is scored like the handful of old
batches whose lab sheets look most like it, which is close to how a cellar
master actually reasons ("this one looks like the autumn lot").  The kinds
are found with k-means on the same lab-sheet picture, and each kind reports
the median panel score of its members.

Plain Python 3, standard library only.
"""

from random import Random

import step1
from step1 import _classify, _column, _median, _number, _split

# Candidate neighbourhood sizes; the best one is picked on held-out
# training rows, so the choice adapts to the cellar's own data.
NEIGHBOUR_CHOICES = (1, 3, 5, 9, 15, 25)

# Every HOLDOUT_STEP-th training row is set aside to judge those candidates.
HOLDOUT_STEP = 4

# Clustering limits -- "a few kinds" means a few, and small enough to stay fast.
MAX_KINDS = 5
KMEANS_RESTARTS = 2
KMEANS_ROUNDS = 15
MIN_KIND_GAIN = 0.10  # keep splitting while scatter drops at least this much
SEED = 20250101

# Guards against division by zero.
_TINY = 1e-12


# --------------------------------------------------------------- header roles

def _target_index(header, goals):
    """Which column the tasting panel fills in.

    Prefer a goal column actually named "quality"; otherwise fall back to the
    first maximized column, then to the last column in the sheet.
    """
    for i, name in enumerate(header):
        text = str(name).strip()
        base = text.rstrip("+-").strip().lower()
        if base == "quality":
            return i
    for i, direction in goals:
        if direction > 0:
            return i
    if goals:
        return goals[0][0]
    return len(header) - 1 if header else 0


def _evidence_indices(header, goals, target):
    """Columns known from the lab sheet: everything except the panel score."""
    _, readings = _classify(header)
    columns = set(readings)
    for i, _ in goals:
        columns.add(i)
    columns.discard(target)
    return sorted(columns)


# ------------------------------------------------------- the lab-sheet picture

def _vector(row, indices, fallbacks, lo, span):
    """One batch as a list of 0..1 numbers, one per evidence column.

    A missing or unreadable cell falls back to the column's typical value, so
    an incomplete lab sheet still gets an answer instead of an error.
    """
    out = []
    for k, index in enumerate(indices):
        x = _number(row[index]) if index < len(row) else None
        if x is None:
            x = fallbacks[k]
        out.append((x - lo[k]) / span[k])
    return out


def _distance2(a, b):
    total = 0.0
    for x, y in zip(a, b):
        diff = x - y
        total += diff * diff
    return total


# ------------------------------------------------------------------ the model

def _neighbour_guess(points, targets, order, vec, k, base):
    """Distance-weighted average score of the k closest past batches.

    ``order`` is the list of (distance, index) pairs, already sorted; it is
    computed once per query and reused for every candidate k.
    """
    if not order:
        return base
    weight_sum = 0.0
    total = 0.0
    for dist, i in order[:k]:
        if dist <= _TINY:
            # An essentially identical past batch: trust it outright.
            return targets[i]
        w = 1.0 / dist
        weight_sum += w
        total += w * targets[i]
    if weight_sum <= 0.0:
        return base
    return total / weight_sum


def _sorted_neighbours(points, vec):
    order = [(_distance2(p, vec), i) for i, p in enumerate(points)]
    order.sort()
    return order


def _choose_k(points, targets, base):
    """Pick a neighbourhood size using held-out training rows."""
    n = len(points)
    if n < 4:
        return 1

    holdout = list(range(0, n, HOLDOUT_STEP))
    keep = [i for i in range(n) if i % HOLDOUT_STEP]
    if not holdout or len(keep) < 2:
        return min(NEIGHBOUR_CHOICES[0], n)

    fit_points = [points[i] for i in keep]
    fit_targets = [targets[i] for i in keep]
    candidates = [k for k in NEIGHBOUR_CHOICES if k <= len(fit_points)]
    if not candidates:
        candidates = [len(fit_points)]

    errors = [0.0] * len(candidates)
    for i in holdout:
        order = _sorted_neighbours(fit_points, points[i])
        truth = targets[i]
        for c, k in enumerate(candidates):
            errors[c] += abs(_neighbour_guess(
                fit_points, fit_targets, order, points[i], k, base) - truth)

    best = 0
    for c in range(1, len(candidates)):
        if errors[c] < errors[best]:
            best = c
    return candidates[best]


def _build(train):
    """Everything derived from the training rows, computed once."""
    header, data = _split(train)
    goals, _ = _classify(header)
    target = _target_index(header, goals)
    indices = _evidence_indices(header, goals, target)

    scores = _column(data, target)
    base = _median(scores)
    if base is None:
        base = 0.0

    fallbacks, lo, span = [], [], []
    for index in indices:
        column = _column(data, index)
        typical = _median(column)
        fallbacks.append(0.0 if typical is None else typical)
        low = min(column) if column else 0.0
        high = max(column) if column else 0.0
        width = high - low
        lo.append(low)
        span.append(width if width > _TINY else 1.0)

    # Only rows with a real panel score can teach us anything.
    points, targets, kept = [], [], []
    for row in data:
        y = _number(row[target]) if target < len(row) else None
        if y is None:
            continue
        points.append(_vector(row, indices, fallbacks, lo, span))
        targets.append(y)
        kept.append(list(row))

    k = _choose_k(points, targets, base) if (points and indices) else 1

    return {
        "indices": indices,
        "fallbacks": fallbacks,
        "lo": lo,
        "span": span,
        "points": points,
        "targets": targets,
        "rows": kept,
        "base": base,
        "k": min(k, len(points)) if points else 1,
    }


# guess() is called once per test batch with the same train rows, so the model
# is built on the first call and reused afterwards.
_CACHE = {}


def _model(train):
    key = (id(train), len(train))
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is train:
        return hit[1]
    if len(_CACHE) > 4:
        _CACHE.clear()
    model = _build(train)
    _CACHE[key] = (train, model)
    return model


# ---------------------------------------------------------------- the clusters

def _kmeans(points, k, rng):
    """k-means with k-means++ seeding; returns (labels, within-cluster scatter)."""
    n = len(points)
    dim = len(points[0])

    centres = [list(points[rng.randrange(n)])]
    nearest = [None] * n
    while len(centres) < k:
        c = centres[-1]
        for i, p in enumerate(points):
            d = _distance2(p, c)
            if nearest[i] is None or d < nearest[i]:
                nearest[i] = d
        total = sum(nearest)
        if total <= _TINY:
            centres.append(list(points[rng.randrange(n)]))
            continue
        cut = rng.random() * total
        acc = 0.0
        pick = n - 1
        for i, d in enumerate(nearest):
            acc += d
            if acc >= cut:
                pick = i
                break
        centres.append(list(points[pick]))

    labels = [-1] * n
    for _ in range(KMEANS_ROUNDS):
        moved = False
        for i, p in enumerate(points):
            best_j, best_d = 0, None
            for j, c in enumerate(centres):
                d = _distance2(p, c)
                if best_d is None or d < best_d:
                    best_d, best_j = d, j
            if labels[i] != best_j:
                labels[i] = best_j
                moved = True
        sums = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for i, p in enumerate(points):
            j = labels[i]
            counts[j] += 1
            s = sums[j]
            for d in range(dim):
                s[d] += p[d]
        for j in range(k):
            if counts[j]:
                centres[j] = [v / counts[j] for v in sums[j]]
        if not moved:
            break

    scatter = 0.0
    for i, p in enumerate(points):
        scatter += _distance2(p, centres[labels[i]])
    return labels, scatter


def _best_kmeans(points, k, seed):
    best = None
    for attempt in range(KMEANS_RESTARTS):
        labels, scatter = _kmeans(points, k, Random(seed + attempt))
        if best is None or scatter < best[1]:
            best = (labels, scatter)
    return best


def _grouping(points):
    """Labels for a sensible small number of kinds.

    Start at two kinds and keep splitting only while the extra kind buys a
    real reduction in within-cluster scatter -- the usual elbow rule, so the
    winemaker gets a few meaningful groups instead of arbitrary slices.
    """
    n = len(points)
    if n < 2:
        return [0] * n

    dim = len(points[0])
    centre = [sum(p[d] for p in points) / n for d in range(dim)]
    previous = sum(_distance2(p, centre) for p in points)
    if previous <= _TINY:
        return [0] * n  # every batch looks the same

    chosen = None
    for k in range(2, min(MAX_KINDS, n) + 1):
        labels, scatter = _best_kmeans(points, k, SEED + 1000 * k)
        gain = (previous - scatter) / previous if previous > _TINY else 0.0
        if chosen is not None and gain < MIN_KIND_GAIN:
            break
        chosen = labels
        previous = scatter
    return chosen if chosen is not None else [0] * n


# --------------------------------------------------------------- the two asks

def guess(train, row):
    """Predicted panel score for one new batch, from its lab sheet alone.

    The answer is the distance-weighted average score of the most similar
    past batches.  With no usable history, the typical past score is the
    honest answer.
    """
    model = _model(train)
    points = model["points"]
    if not points:
        return float(model["base"])
    if not model["indices"]:
        return float(model["base"])

    vec = _vector(row, model["indices"], model["fallbacks"],
                  model["lo"], model["span"])
    order = _sorted_neighbours(points, vec)
    return float(_neighbour_guess(points, model["targets"], order, vec,
                                  model["k"], model["base"]))


def kinds(train):
    """The cellar's batches split into a few kinds, best kind first.

    Each entry is ``(group_rows, expected_quality)``: the batches of that
    kind (copies of the original rows) and the score to expect from it, taken
    as the median panel score of its members.
    """
    model = _model(train)
    rows = model["rows"]
    points = model["points"]
    targets = model["targets"]
    if not rows:
        return []
    if not model["indices"]:
        return [([list(r) for r in rows], float(model["base"]))]

    labels = _grouping(points)
    buckets = {}
    for i, label in enumerate(labels):
        buckets.setdefault(label, []).append(i)

    out = []
    for label in sorted(buckets):
        members = buckets[label]
        group = [list(rows[i]) for i in members]
        expected = _median([targets[i] for i in members])
        if expected is None:
            expected = model["base"]
        out.append((group, float(expected)))

    out.sort(key=lambda pair: (-pair[1], -len(pair[0])))
    return out
