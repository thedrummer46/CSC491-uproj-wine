"""Step 4 of the winemaker's assistant.

Answers the two requests that come up while the harvest is running:

  worth(model, row)   -- "as each new batch comes off the line, tell me right
                         away whether it's worth sending to the panel"
  learn(model, row)   -- "keep sharpening your sense of what makes a good
                         batch as this season's scores come in, without
                         starting over"

with the two supporting calls the winemaker needs around them:

  start(names)        -- an assistant that knows only the header row
  predict(model, row) -- the panel score we expect for one batch

Everything about the header row -- which column the panel fills in, which
columns the lab sheet already shows -- is read with step 1 and step 2's
helpers, so the meaning of the ``+``/``-`` markers is still defined in
exactly one place.  Batches are compared with the same nearest-past-batches
reasoning as step 2 (``_vector``, ``_sorted_neighbours``,
``_neighbour_guess``), so the online assistant and the batch one agree about
what "looks like the autumn lot" means.

Nothing here is refitted from scratch.  One scored batch arrives, and:

  * the running column stats (typical value, lowest, highest) take it in;
  * the least-squares normal equations take it in as a rank-one update, so
    the straight-line read on the whole season is exact without ever
    revisiting an old batch;
  * the batch joins the remembered cellar, which is what the neighbour
    reasoning consults;
  * and -- before any of that -- the batch is first used as an honest test:
    we predict it while it is still unseen and file the error.  Those filed
    errors are what lets the assistant sharpen its judgement, because they
    say which neighbourhood size, and whether neighbours, the straight line,
    or the two averaged, has actually been serving this cellar best.

Plain Python 3, standard library only.
"""

from bisect import bisect_left, insort
from random import Random

import step2
from step1 import _classify, _median, _number
from step2 import (NEIGHBOUR_CHOICES, _evidence_indices, _neighbour_guess,
                   _sorted_neighbours, _target_index, _vector)

# A batch is worth the panel's time at this predicted score or better.
WORTH_THRESHOLD = 6.0

# Rounding slack, so a prediction of exactly 6 is never lost to float noise.
WORTH_TOL = 1e-9

# Neighbourhood size used before enough batches have been judged to know
# which size suits this cellar.
DEFAULT_K = 5

# A way of answering must have been tested on this many unseen batches
# before its track record is allowed to decide anything.
MIN_EVAL = 30

# Conditioning nudge on the normal equations, relative to their own scale.
RIDGE = 1e-6

# The cached lab-sheet pictures are rebuilt once a column's observed range
# has shifted by this much; a bookkeeping rescale, not a refit.
RESCALE_TOL = 0.05

# How many past batches the neighbour reasoning keeps on hand.  Well above a
# season's worth, so in normal use nothing is ever dropped; the cap only
# keeps a runaway stream from slowing the answers down.
MEMORY_LIMIT = 2000

SEED = 20250101

_TINY = step2._TINY


# ------------------------------------------------------------- the empty model

def start(names):
    """An assistant that knows the header row and nothing else.

    It can already be asked anything -- it simply has no opinion yet, and
    says so by predicting 0.0 until the first scored batch arrives.
    """
    header = list(names) if names else []
    goals, _ = _classify(header)
    target = _target_index(header, goals)
    indices = _evidence_indices(header, goals, target)
    width = len(indices)
    size = width + 1  # evidence columns plus the intercept

    return {
        "header": header,
        "target": target,
        "indices": indices,

        # running per-column stats over every row seen
        "counts": [0] * width,
        "sums": [0.0] * width,
        "lo": [0.0] * width,
        "hi": [0.0] * width,
        "fallbacks": [0.0] * width,
        "span": [1.0] * width,

        # the scale the remembered pictures were drawn on
        "scale_lo": [0.0] * width,
        "scale_span": [1.0] * width,

        # the remembered cellar
        "rows": [],
        "points": [],
        "targets": [],
        "ysorted": [],
        "ymin": None,
        "ymax": None,
        "scored": 0,

        # normal equations for the straight-line read
        "xtx": [[0.0] * size for _ in range(size)],
        "xty": [0.0] * size,
        "weights": None,
        "dirty": True,

        # track records, each over the batches it was actually tested on
        "knn_err": [0.0] * len(NEIGHBOUR_CHOICES),
        "knn_n": [0] * len(NEIGHBOUR_CHOICES),
        "mix_err": [0.0] * len(NEIGHBOUR_CHOICES),
        "mix_n": [0] * len(NEIGHBOUR_CHOICES),
        "lin_err": 0.0,
        "lin_n": 0,

        "cache": {},
        "rng": Random(SEED),
    }


# ------------------------------------------------------------- running column stats

def _features(model, row):
    """The lab-sheet numbers of one batch, gaps filled with typical values."""
    fallbacks = model["fallbacks"]
    out = []
    for k, index in enumerate(model["indices"]):
        x = _number(row[index]) if index < len(row) else None
        out.append(fallbacks[k] if x is None else x)
    return out


def _observe(model, row):
    """Fold one batch into the running per-column stats."""
    counts = model["counts"]
    sums = model["sums"]
    lo = model["lo"]
    hi = model["hi"]
    for k, index in enumerate(model["indices"]):
        x = _number(row[index]) if index < len(row) else None
        if x is None:
            continue
        if counts[k] == 0:
            lo[k] = x
            hi[k] = x
        else:
            if x < lo[k]:
                lo[k] = x
            if x > hi[k]:
                hi[k] = x
        counts[k] += 1
        sums[k] += x
        model["fallbacks"][k] = sums[k] / counts[k]
        width = hi[k] - lo[k]
        model["span"][k] = width if width > _TINY else 1.0


def _drifted(model):
    """Has a column stretched far enough to spoil the remembered pictures?"""
    for k in range(len(model["indices"])):
        span = model["span"][k]
        used = model["scale_span"][k]
        if abs(model["lo"][k] - model["scale_lo"][k]) > RESCALE_TOL * span:
            return True
        if abs(span - used) > RESCALE_TOL * max(span, used, _TINY):
            return True
    return False


def _rescale(model):
    """Redraw the remembered pictures on the current column ranges."""
    model["scale_lo"] = list(model["lo"])
    model["scale_span"] = list(model["span"])
    model["points"] = [_picture(model, r) for r in model["rows"]]


def _picture(model, row):
    """One batch as step 2 would draw it, on the remembered scale."""
    return _vector(row, model["indices"], model["fallbacks"],
                   model["scale_lo"], model["scale_span"])


# ------------------------------------------------------- the straight-line read

def _update_linear(model, row, y):
    """Fold one scored batch into the normal equations (a rank-one update)."""
    x = _features(model, row)
    x.append(1.0)
    xtx = model["xtx"]
    xty = model["xty"]
    size = len(x)
    for i in range(size):
        xi = x[i]
        if xi == 0.0:
            continue
        upper = xtx[i]
        for j in range(i, size):
            upper[j] += xi * x[j]
        xty[i] += xi * y
    model["dirty"] = True


def _solve(matrix, rhs):
    """Gaussian elimination with partial pivoting; None when singular."""
    n = len(rhs)
    work = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = col
        best = abs(work[col][col])
        for r in range(col + 1, n):
            v = abs(work[r][col])
            if v > best:
                best, pivot = v, r
        if best <= _TINY:
            return None
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
        prow = work[col]
        pv = prow[col]
        for r in range(col + 1, n):
            rowr = work[r]
            factor = rowr[col] / pv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                rowr[c] -= factor * prow[c]
    answer = [0.0] * n
    for i in range(n - 1, -1, -1):
        rowi = work[i]
        total = rowi[n]
        for j in range(i + 1, n):
            total -= rowi[j] * answer[j]
        answer[i] = total / rowi[i]
    return answer


def _weights(model):
    """Least-squares weights from the stored sums, solved only when stale."""
    if not model["dirty"]:
        return model["weights"]
    model["dirty"] = False
    model["weights"] = None

    size = len(model["indices"]) + 1
    if size <= 1 or len(model["targets"]) < size + 2:
        return None

    xtx = model["xtx"]
    trace = sum(xtx[i][i] for i in range(size))
    ridge = RIDGE * (trace / size) if trace > 0.0 else RIDGE

    full = []
    for i in range(size):
        line = [xtx[i][j] if j >= i else xtx[j][i] for j in range(size)]
        line[i] += ridge
        full.append(line)

    model["weights"] = _solve(full, model["xty"])
    return model["weights"]


def _linear(model, row):
    """The straight line's read on one batch, or None while it has no read."""
    weights = _weights(model)
    if weights is None:
        return None
    x = _features(model, row)
    total = weights[-1]
    for k, value in enumerate(x):
        total += weights[k] * value
    return total


# ------------------------------------------------------- what has been working

def _candidates(model):
    """Neighbourhood sizes worth considering against the cellar we remember."""
    n = len(model["points"])
    sizes = [(c, k) for c, k in enumerate(NEIGHBOUR_CHOICES) if k <= n]
    if not sizes and n:
        sizes = [(0, n)]
    return sizes


def _choice(model):
    """(way of answering, neighbourhood size) with the best track record.

    Each way is judged on its mean error over the unseen batches it was
    actually tested on, and only once it has a long enough record; until
    then the assistant sticks to a sensible default neighbourhood.
    """
    sizes = _candidates(model)
    if not sizes:
        return "knn", 0

    default = min(sizes, key=lambda pair: (abs(pair[1] - DEFAULT_K), pair[1]))
    best = None
    for c, k in sizes:
        if model["knn_n"][c] >= MIN_EVAL:
            mean = model["knn_err"][c] / model["knn_n"][c]
            if best is None or mean < best[0]:
                best = (mean, "knn", k)
        if model["mix_n"][c] >= MIN_EVAL:
            mean = model["mix_err"][c] / model["mix_n"][c]
            if best is None or mean < best[0]:
                best = (mean, "mix", k)
    if model["lin_n"] >= MIN_EVAL:
        mean = model["lin_err"] / model["lin_n"]
        if best is None or mean < best[0]:
            best = (mean, "lin", default[1])

    if best is None:
        return "knn", default[1]
    return best[1], best[2]


def _combine(model, order, vec, base, way, k, lin):
    """One prediction, given a way of answering and a lab-sheet picture."""
    neighbours = base
    if k and order:
        neighbours = _neighbour_guess(model["points"], model["targets"],
                                      order, vec, k, base)
    if way == "lin":
        return neighbours if lin is None else lin
    if way == "mix" and lin is not None:
        return 0.5 * (neighbours + lin)
    return neighbours


def _base(model):
    """The score a typical remembered batch was awarded."""
    middle = _median(model["ysorted"])
    return 0.0 if middle is None else middle


def _clamp(model, value):
    """Keep a prediction inside the range of scores the panel has awarded."""
    low, high = model["ymin"], model["ymax"]
    if low is not None and value < low:
        return low
    if high is not None and value > high:
        return high
    return value


# --------------------------------------------------------------- remembering

def _remember(model, row, point, y):
    """Add one scored batch to the cellar the neighbour reasoning consults.

    Past the memory cap a batch takes the place of a randomly chosen older
    one (reservoir sampling), so the remembered cellar stays a fair sample
    of the season and the answers stay fast.  The straight-line read is
    unaffected either way -- its sums already hold every batch ever seen.
    """
    model["scored"] += 1
    ysorted = model["ysorted"]

    if len(model["rows"]) < MEMORY_LIMIT:
        model["rows"].append(list(row))
        model["points"].append(point)
        model["targets"].append(y)
        insort(ysorted, y)
        return

    j = model["rng"].randrange(model["scored"])
    if j >= MEMORY_LIMIT:
        return
    gone = model["targets"][j]
    spot = bisect_left(ysorted, gone)
    if spot < len(ysorted) and ysorted[spot] == gone:
        del ysorted[spot]
    model["rows"][j] = list(row)
    model["points"][j] = point
    model["targets"][j] = y
    insort(ysorted, y)


# --------------------------------------------------------------- the two asks

def learn(model, row):
    """Take in one more scored batch and sharpen the assistant on it.

    The batch is first predicted while it is still unseen, and the error
    filed against every way of answering; only then is it folded in.  Each
    update touches just this batch -- the running stats, a rank-one update of
    the normal equations, and one more entry in the remembered cellar -- so
    nothing is ever refitted from scratch.
    """
    if not row:
        return model

    target = model["target"]
    y = _number(row[target]) if target < len(row) else None
    if y is None:
        # No score on the sheet: it still tells us about the lab readings.
        _observe(model, row)
        if _drifted(model):
            _rescale(model)
        if model["cache"]:
            model["cache"] = {}
        return model

    points = model["points"]
    if points and model["indices"]:
        # Honest test first: this batch has not been folded in yet.
        base = _base(model)
        vec = _picture(model, row)
        order = _sorted_neighbours(points, vec)
        lin = _linear(model, row)
        for c, k in _candidates(model):
            guess = _neighbour_guess(points, model["targets"], order, vec,
                                     k, base)
            model["knn_err"][c] += abs(guess - y)
            model["knn_n"][c] += 1
            if lin is not None:
                model["mix_err"][c] += abs(0.5 * (guess + lin) - y)
                model["mix_n"][c] += 1
        if lin is not None:
            model["lin_err"] += abs(lin - y)
            model["lin_n"] += 1

    _update_linear(model, row, y)

    point = _picture(model, row)
    _remember(model, row, point, y)
    if model["ymin"] is None or y < model["ymin"]:
        model["ymin"] = y
    if model["ymax"] is None or y > model["ymax"]:
        model["ymax"] = y

    _observe(model, row)
    if _drifted(model):
        _rescale(model)

    if model["cache"]:
        model["cache"] = {}
    return model


def predict(model, row):
    """The panel score we expect for one batch, on today's experience.

    With nothing learned yet there is no opinion to give and the answer is
    0.0; with scored batches but no usable lab sheet, the typical past score
    is the honest answer.
    """
    if not model["targets"]:
        return 0.0

    base = _base(model)
    if not model["indices"] or not model["points"]:
        return float(base)

    try:
        key = tuple(row)
    except TypeError:
        key = None
    if key is not None:
        hit = model["cache"].get(key)
        if hit is not None:
            return hit

    way, k = _choice(model)
    vec = _picture(model, row)
    order = _sorted_neighbours(model["points"], vec) if k else []
    lin = _linear(model, row) if way in ("lin", "mix") else None
    answer = float(_clamp(model, _combine(model, order, vec, base,
                                          way, k, lin)))

    if key is not None:
        model["cache"][key] = answer
    return answer


def worth(model, row):
    """Is this batch worth the panel's time?

    True when the score we expect for it is 6 or better -- the same number
    the predictor reports, so the recommendation never disagrees with the
    estimate it rests on.
    """
    return predict(model, row) >= WORTH_THRESHOLD - WORTH_TOL
