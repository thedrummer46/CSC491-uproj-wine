"""Step 3 of the winemaker's assistant.

Answers the last two requests about the cellar:

  certify(train, a, b) -- "when you say batch A will beat batch B, tell me
                           whether that is a real difference or just tasting
                           noise"
  accuracy(train, test) -- "how often are your score estimates right on
                            batches you have never seen before?"

Both reuse step 2's model of the cellar (nearest past batches, built once per
train table and cached there) and step 1's header reading and numeric
helpers, so the meaning of the ``+``/``-`` markers, the choice of the panel
score column and the way a batch is turned into a lab-sheet picture are all
still defined in exactly one place.

How a difference is judged real.  Two things have to be true before we are
willing to say A beats B:

  1. The gap has to be big enough to show up on the panel's own scale.  The
     panel awards scores on a grid (whole points, for ``Quality+``), so a gap
     smaller than half a step could never change the score a batch is given.
  2. The gap has to be bigger than the wobble in our own two estimates.  Each
     estimate carries a standard error, taken from how much the similar past
     batches disagree among themselves; when only one past batch is close
     enough to speak, we fall back on the typical error our predictions make
     on held-out past batches.  A gap inside one standard error of itself is
     tasting noise.

Anything that fails either test is reported as ``'same'``.

Plain Python 3, standard library only.
"""

from math import floor, sqrt

import step2
from step1 import _classify, _median, _number, _split
from step2 import (_model, _neighbour_guess, _sorted_neighbours,
                   _target_index, _vector)

# A gap must be at least this many standard errors of the gap to count.
CONFIDENCE_Z = 1.0

# Half a step on the panel's scale: smaller gaps cannot change a score.
FLOOR_SHARE = 0.5

# How many past batches to re-predict when measuring our own typical error.
# Enough to pin the error down, few enough to stay fast on a big cellar.
PROBE_LIMIT = 150

# Turns a median absolute deviation into a standard-deviation-like number.
MAD_TO_SIGMA = 1.4826

# Two scores count as the same awarded score within this slack.
SAME_SCORE_TOL = 1e-9

# A grid step this small relative to the range of scores is no grid at all;
# treat the scale as whole points, which is what "rounds to" means.
MIN_GRID_SHARE = 0.02

_TINY = step2._TINY


# ------------------------------------------------------------- the panel scale

def _spread(values):
    """Robust standard-deviation-like spread of a list of numbers."""
    middle = _median(values)
    if middle is None:
        return 0.0
    return MAD_TO_SIGMA * _median([abs(v - middle) for v in values])


def _scale(scores):
    """(origin, step) of the grid of scores the panel actually awards.

    For a whole-point scale such as ``Quality+`` this is (0.0, 1.0), i.e.
    ordinary rounding.  A cellar that records half points gets a half-point
    grid.  A score that is really continuous gets whole points, since that is
    what "rounds to the true score" means.
    """
    if not scores:
        return 0.0, 1.0
    if all(abs(s - floor(s + 0.5)) <= SAME_SCORE_TOL for s in scores):
        return 0.0, 1.0

    grid = sorted(set(scores))
    if len(grid) < 2:
        return 0.0, 1.0
    gaps = [grid[i + 1] - grid[i] for i in range(len(grid) - 1)]
    step = _median(gaps)
    span = grid[-1] - grid[0]
    if step is None or step <= _TINY or step < span * MIN_GRID_SHARE:
        return 0.0, 1.0
    return grid[0], step


def _snap(value, origin, step):
    """The nearest score on the panel's grid (ties go up, as rounding does)."""
    if step <= _TINY:
        return value
    return origin + step * floor((value - origin) / step + 0.5)


# ------------------------------------------------- our own typical error

def _residual_sigma(model):
    """Typical size of our prediction error, measured on past batches.

    Each probe batch is predicted from the *other* past batches, so the
    number reflects performance on unseen wine rather than memory.
    """
    points = model["points"]
    targets = model["targets"]
    n = len(points)
    if n < 3 or not model["indices"]:
        return _spread(targets)

    stride = max(1, n // PROBE_LIMIT)
    residuals = []
    for i in range(0, n, stride):
        order = [pair for pair in _sorted_neighbours(points, points[i])
                 if pair[1] != i]
        if not order:
            continue
        guessed = _neighbour_guess(points, targets, order, points[i],
                                   model["k"], model["base"])
        residuals.append(abs(guessed - targets[i]))

    if not residuals:
        return _spread(targets)
    sigma = MAD_TO_SIGMA * _median(residuals)
    if sigma <= _TINY:
        # Predictions are nearly exact; fall back on the mean error so the
        # noise test is not degenerate.
        sigma = sum(residuals) / len(residuals)
    return sigma


# ------------------------------------------------------------------- context

# certify() is called for hundreds of pairs and accuracy() for hundreds of
# rows, always with the same train table, so everything derived from the
# training data -- and every per-batch estimate -- is computed once and kept.
_CONTEXT = {}


def _build_context(train):
    model = _model(train)
    origin, step = _scale(model["targets"])
    return {
        "model": model,
        "sigma": _residual_sigma(model),
        "origin": origin,
        "step": step,
        "floor": FLOOR_SHARE * step,
        "estimates": {},
    }


def _context(train):
    key = (id(train), len(train))
    hit = _CONTEXT.get(key)
    if hit is not None and hit[0] is train:
        return hit[1]
    if len(_CONTEXT) > 4:
        _CONTEXT.clear()
    ctx = _build_context(train)
    _CONTEXT[key] = (train, ctx)
    return ctx


# --------------------------------------------------------- one batch estimated

def _estimate(ctx, row):
    """(predicted score, standard error) for one batch.

    The predicted score is exactly what ``step2.guess`` returns -- the same
    model, the same neighbourhood size -- so certifying a pair and scoring
    accuracy can never disagree about what we think a batch will get.

    The standard error comes from how much the similar past batches disagree:
    a weighted spread divided by the effective number of batches that had a
    say.  When too few batches speak up to measure that (or when an identical
    past batch is quoted outright) our typical held-out error stands in.
    """
    model = ctx["model"]
    points = model["points"]
    if not points or not model["indices"]:
        return float(model["base"]), ctx["sigma"]

    try:
        key = tuple(row)
    except TypeError:
        key = None
    if key is not None:
        hit = ctx["estimates"].get(key)
        if hit is not None:
            return hit

    targets = model["targets"]
    vec = _vector(row, model["indices"], model["fallbacks"],
                  model["lo"], model["span"])
    order = _sorted_neighbours(points, vec)
    mean = float(_neighbour_guess(points, targets, order, vec,
                                  model["k"], model["base"]))

    error = ctx["sigma"]
    chosen = order[:model["k"]]
    if chosen and all(dist > _TINY for dist, _ in chosen):
        weight_sum = 0.0
        square_sum = 0.0
        for dist, _ in chosen:
            w = 1.0 / dist
            weight_sum += w
            square_sum += w * w
        if weight_sum > 0.0 and square_sum > 0.0:
            variance = 0.0
            for dist, i in chosen:
                diff = targets[i] - mean
                variance += (1.0 / dist) * diff * diff
            variance /= weight_sum
            effective = (weight_sum * weight_sum) / square_sum
            if effective >= 2.0:
                error = sqrt(variance / effective)

    answer = (mean, error)
    if key is not None:
        ctx["estimates"][key] = answer
    return answer


# --------------------------------------------------------------- the two asks

def certify(train, a, b):
    """Which of two batches will score better, or ``'same'`` if it is noise.

    Returns ``'A'``, ``'B'`` or ``'same'``.  A winner is only named when the
    predicted gap clears both the panel's own half-step resolution and one
    standard error of the gap itself; otherwise the honest answer is that the
    two batches are indistinguishable.
    """
    ctx = _context(train)
    mean_a, error_a = _estimate(ctx, a)
    mean_b, error_b = _estimate(ctx, b)

    gap = mean_a - mean_b
    size = abs(gap)
    if size <= _TINY:
        return "same"

    wobble = sqrt(error_a * error_a + error_b * error_b)
    threshold = max(ctx["floor"], CONFIDENCE_Z * wobble)
    if size < threshold:
        return "same"
    return "A" if gap > 0.0 else "B"


def accuracy(train, test):
    """Share of unseen batches whose guessed score rounds to the true one.

    The guess is rounded onto the scale the panel actually awards (whole
    points for ``Quality+``) and compared with the recorded score.  Test rows
    with no recorded score teach nothing and are left out of the tally; with
    no scored test rows at all the share is 0.0.
    """
    ctx = _context(train)
    header, data = _split(test)
    if not data:
        return 0.0

    goals, _ = _classify(header)
    target = _target_index(header, goals)
    origin, step = ctx["origin"], ctx["step"]

    right = 0
    total = 0
    for row in data:
        truth = _number(row[target]) if target < len(row) else None
        if truth is None:
            continue
        total += 1
        mean, _error = _estimate(ctx, row)
        if abs(_snap(mean, origin, step) - truth) <= SAME_SCORE_TOL:
            right += 1

    if not total:
        return 0.0
    return right / float(total)
