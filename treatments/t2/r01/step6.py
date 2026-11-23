"""Step 6 of the winemaker's assistant.

Answers the two requests the winemaker makes when the panel's verdict is
already in and the question is *why*:

  blame(train, k=3)  -- "which two or three readings matter most for whether
                        a batch scores well?"
  dragging(train)    -- "our low-scoring batches: which lab readings are
                        dragging them down?"

Both answers name lab readings only.  The goal columns (``Quality+``,
``Alcohol-``) are what we are explaining, never an explanation, so they are
excluded by construction: the roles come from step 1's reading of the
``+``/``-`` header markers, and the panel's own column is step 2's
``_target_index``, so "which column the panel fills in" is still decided in
exactly one place.  "A poor batch" means the same thing it means in step 4 --
a score below :data:`step4.WORTH_THRESHOLD`, i.e. below 6.

How the two answers are reached.

``blame`` asks which readings carry information about the score, not which
ones happen to drift with it.  Each reading is cut into quantile bands and
judged by how much of the score's remaining scatter the band averages
explain -- a measure that notices a reading mattering in a bent or
humped way, which a plain correlation would miss.  The strongest reading is
taken, its band averages are subtracted off, and the search repeats on what
is left.  So a second reading is only credited for what the first one did
not already account for, and the winemaker gets two or three readings that
each say something new instead of three names for the same effect.

``dragging`` compares the poor batches with the rest, reading by reading,
using the chance that a poor batch sits higher on that reading than a good
one (Cliff's delta).  That is a rank comparison, so it does not care about
units and is not dragged around by one freak batch, and its sign says
directly whether the reading runs too **high** or too **low** in poor wine.
Readings whose shift is no bigger than a negligible effect are left out --
reporting them would be reading tea leaves.

Plain Python 3, standard library only.
"""

from bisect import bisect_right

import step2
import step4
from step1 import _classify, _median, _number, _split
from step2 import _target_index

# Quantile bands per reading when measuring how much of the score it explains.
BANDS = 5

# Never cut a reading so finely that a band holds fewer batches than this.
MIN_BAND_ROWS = 10

# Smallest rank shift worth telling the winemaker about: below this, the
# reading is not meaningfully different in poor batches (Romano's "small").
SMALL_EFFECT = 0.147

# A poor batch is one the panel scored below this -- step 4's own threshold.
POOR_BELOW = step4.WORTH_THRESHOLD

_TINY = step2._TINY


# ------------------------------------------------------------ reading the table

def _aligned(data, index):
    """One column, one value per row, gaps filled with the typical value.

    Keeping the column aligned with the rows matters here: every reading is
    compared against the same list of panel scores, so a row may not simply
    vanish from one column and not another.
    """
    values, missing = [], []
    for position, row in enumerate(data):
        x = _number(row[index]) if index < len(row) else None
        if x is None:
            missing.append(position)
        values.append(x)
    if missing:
        typical = _median([v for v in values if v is not None])
        if typical is None:
            typical = 0.0
        for position in missing:
            values[position] = typical
    return values


def _bands(values):
    """(band of each batch, number of bands) for one reading.

    Bands are quantile cuts, so each holds a comparable share of the cellar
    whatever shape the reading's spread has.  Returns ``(None, 0)`` for a
    reading that never varies, since it can explain nothing.
    """
    n = len(values)
    ordered = sorted(values)
    distinct = len(set(ordered))
    if distinct < 2 or n < 2:
        return None, 0

    count = min(BANDS, distinct, max(2, n // MIN_BAND_ROWS))
    edges = []
    for i in range(1, count):
        edge = ordered[(i * n) // count]
        if not edges or edge > edges[-1]:
            edges.append(edge)
    if not edges:
        return None, 0

    labels = [bisect_right(edges, v) for v in values]
    return labels, len(edges) + 1


def _prepare(train):
    """Everything both answers need from the training table, read once."""
    header, rows = _split(train)
    goals, readings = _classify(header)
    target = _target_index(header, goals)

    # Only batches the panel has actually scored can explain anything.
    data, scores = [], []
    for row in rows:
        y = _number(row[target]) if target < len(row) else None
        if y is None:
            continue
        data.append(row)
        scores.append(y)

    columns = []
    for index in readings:
        if index == target:
            continue  # defensive: the panel's own column is never a reading
        values = _aligned(data, index) if data else []
        labels, count = _bands(values) if values else (None, 0)
        columns.append({
            "index": index,
            "name": header[index],
            "values": values,
            "labels": labels,
            "count": count,
        })

    return {
        "header": header,
        "data": data,
        "scores": scores,
        "columns": columns,
    }


# blame() and dragging() are each called once per split, but both read the
# same table, so the reading is done once and kept.
_CACHE = {}


def _context(train):
    key = (id(train), len(train))
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is train:
        return hit[1]
    if len(_CACHE) > 4:
        _CACHE.clear()
    prepared = _prepare(train)
    _CACHE[key] = (train, prepared)
    return prepared


# ------------------------------------------------- what a reading accounts for

def _explained(column, residual):
    """(share of scatter the reading's bands explain, band averages).

    The share is the between-band sum of squares over the total: 0 when the
    bands all average the same, 1 when knowing the band tells you the score.
    """
    labels = column["labels"]
    count = column["count"]
    if not labels or count < 2:
        return 0.0, None

    n = len(residual)
    totals = [0.0] * count
    counts = [0] * count
    for i in range(n):
        b = labels[i]
        totals[b] += residual[i]
        counts[b] += 1

    overall = sum(residual) / n
    total_ss = 0.0
    for r in residual:
        diff = r - overall
        total_ss += diff * diff
    if total_ss <= _TINY:
        return 0.0, None

    means = [0.0] * count
    between = 0.0
    for b in range(count):
        if not counts[b]:
            continue
        means[b] = totals[b] / counts[b]
        diff = means[b] - overall
        between += counts[b] * diff * diff

    return between / total_ss, means


def _ranked_readings(prepared, wanted):
    """Reading names in order of what each adds, strongest first.

    After a reading is credited, its band averages are subtracted from the
    scores being explained, so the next reading is judged on the part of the
    story still untold.  Readings that explain nothing at all are appended in
    header order, only to fill the requested count honestly.
    """
    scores = prepared["scores"]
    pool = [c for c in prepared["columns"] if c["labels"]]
    if not scores or not pool:
        return []

    residual = list(scores)
    chosen = []
    while len(chosen) < wanted and pool:
        best = None
        for slot, column in enumerate(pool):
            share, means = _explained(column, residual)
            if share <= _TINY or means is None:
                continue
            if best is None or share > best[0]:
                best = (share, slot, means)
        if best is None:
            break
        _share, slot, means = best
        column = pool.pop(slot)
        chosen.append(column["name"])
        labels = column["labels"]
        for i in range(len(residual)):
            residual[i] -= means[labels[i]]
    return chosen


# ------------------------------------------------- poor batches versus the rest

def _cliff_delta(poor, good):
    """Chance a poor batch reads higher than a good one, on a -1..+1 scale.

    ``+1`` means every poor batch reads higher than every good one, ``-1``
    the reverse, ``0`` that the two are indistinguishable.  Computed from
    mid-ranks, so ties are handled and the cost is a single sort.
    """
    n_poor, n_good = len(poor), len(good)
    if not n_poor or not n_good:
        return 0.0

    merged = [(v, 0) for v in poor]
    merged.extend((v, 1) for v in good)
    merged.sort(key=lambda pair: pair[0])

    total = n_poor + n_good
    rank_sum = 0.0
    i = 0
    while i < total:
        j = i
        while j + 1 < total and merged[j + 1][0] == merged[i][0]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            if merged[t][1] == 0:
                rank_sum += rank
        i = j + 1

    u = rank_sum - n_poor * (n_poor + 1) / 2.0
    return 2.0 * (u / float(n_poor * n_good)) - 1.0


# --------------------------------------------------------------- the two asks

def blame(train, k=3):
    """The readings that matter most for the panel's score, strongest first.

    Returns at most ``k`` lab-reading column names, exactly as they appear in
    the header -- never a goal column.  Each name after the first is there
    for what it adds beyond the ones already named, so three readings that
    merely echo one another are never all reported.
    """
    prepared = _context(train)
    names = [c["name"] for c in prepared["columns"]]
    if not names:
        return []

    try:
        wanted = int(k)
    except (TypeError, ValueError):
        wanted = 3
    wanted = max(0, min(wanted, len(names)))
    if wanted == 0:
        return []

    chosen = _ranked_readings(prepared, wanted)
    if len(chosen) < wanted:
        taken = set(chosen)
        for name in names:
            if len(chosen) >= wanted:
                break
            if name not in taken:
                chosen.append(name)
                taken.add(name)
    return chosen[:wanted]


def dragging(train):
    """Readings that run too high or too low in the poor batches, worst first.

    A poor batch is one the panel scored below 6.  Each reading is compared
    between the poor batches and the rest; the pairs returned are
    ``(column name, 'high')`` when poor wine reads higher on that reading and
    ``(column name, 'low')`` when it reads lower, ordered by how big the
    shift is.  Readings whose shift is negligible are left out, and an empty
    list is the honest answer when nothing separates the poor batches -- or
    when the cellar has no poor batches, or only poor batches, to compare.
    """
    prepared = _context(train)
    scores = prepared["scores"]
    if not scores:
        return []

    poor_rows = [i for i, y in enumerate(scores) if y < POOR_BELOW]
    good_rows = [i for i, y in enumerate(scores) if y >= POOR_BELOW]
    if not poor_rows or not good_rows:
        return []

    found = []
    for order, column in enumerate(prepared["columns"]):
        values = column["values"]
        if not values or column["count"] < 2:
            continue
        delta = _cliff_delta([values[i] for i in poor_rows],
                             [values[i] for i in good_rows])
        if abs(delta) < SMALL_EFFECT:
            continue
        found.append((abs(delta), order, column["name"],
                      "high" if delta > 0.0 else "low"))

    found.sort(key=lambda item: (-item[0], item[1]))
    return [(name, direction) for _size, _order, name, direction in found]
