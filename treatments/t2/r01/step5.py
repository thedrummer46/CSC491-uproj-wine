"""Step 5 of the winemaker's assistant.

Answers the two requests that come up when tasting itself is the scarce
resource -- the goal cells (``Alcohol-``, ``Quality+``) of every batch are
``'?'`` until the panel actually tastes it, and each ``label(row)`` call
spends one of the month's few tastings:

  taste(rows, label, budget)  -- "the panel can only taste 20 batches this
                                 month; which 20 should they taste to find
                                 our best wine?"
  choose(rows, label, budget) -- "find me a batch that scores high but stays
                                 low in alcohol, tasting as few as possible"

Column meaning is still read with step 1, so the ``+``/``-`` markers are
defined in exactly one place, and "how good is this batch" is still step 1's
notion of exceptional: every declared goal turned so that bigger is better,
min-max normalized and averaged, with the Pareto front preferred.  A batch is
still compared with another batch by step 2's lab-sheet picture
(``_vector``, ``_sorted_neighbours``, ``_neighbour_guess``) -- except that
here only the plain lab readings may be used, since the goal columns are
exactly what we are not allowed to look at until we pay for a tasting.

How the tastings are spent:

  1. A spread-out opening.  The first few batches are chosen by
     farthest-first traversal of the lab-sheet pictures, starting from the
     most ordinary batch in the cellar.  That buys a cheap map of what kinds
     of wine are in the shed before any guessing begins.
  2. Then one batch at a time, chosen on the evidence so far.  Every tasted
     batch is scored on its goals; a nearest-past-batches read predicts that
     score for each untasted batch, and the batch picked is the one with the
     best mix of a high predicted score and being somewhere we have not
     tasted yet.  The weight on "somewhere new" fades as the budget runs
     down, so the assistant explores early and cashes in late.

Plain Python 3, standard library only.
"""

from math import ceil

import step1
import step2
from step1 import _classify, _column, _median, _split
from step2 import (_distance2, _neighbour_guess, _sorted_neighbours, _vector)

# Share of the budget spent on the spread-out opening.
WARMUP_SHARE = 0.4

# How many tasted batches speak when predicting an untasted one.
NEIGHBOURS = 3

# Weight on "somewhere new" at the very start of the guided phase; it decays
# to nothing as the last tasting approaches.
EXPLORE = 1.0

_TINY = step2._TINY


# ------------------------------------------------------- the lab-sheet picture

def _pictures(header, data):
    """(reading columns, one 0..1 picture per batch).

    Only the plain lab readings are used: the goal cells are unknown until a
    batch is tasted, so looking at them here would be cheating.  Gaps fall
    back to the column's typical value, exactly as in step 2.
    """
    _, readings = _classify(header)
    fallbacks, lo, span = [], [], []
    for index in readings:
        column = _column(data, index)
        typical = _median(column)
        fallbacks.append(0.0 if typical is None else typical)
        low = min(column) if column else 0.0
        high = max(column) if column else 0.0
        width = high - low
        lo.append(low)
        span.append(width if width > _TINY else 1.0)

    points = [_vector(row, readings, fallbacks, lo, span) for row in data]
    return readings, points


# ----------------------------------------------------------- the opening moves

def _spread_out(points, count):
    """Indices of ``count`` batches that cover the cellar as widely as possible.

    Farthest-first traversal: start from the most ordinary batch (closest to
    the middle of the shed), then repeatedly take whichever batch is least
    like anything chosen so far.
    """
    n = len(points)
    if count >= n:
        return list(range(n))
    dim = len(points[0]) if points else 0
    if not dim:
        return list(range(count))

    centre = [sum(p[d] for p in points) / float(n) for d in range(dim)]
    first = min(range(n), key=lambda i: (_distance2(points[i], centre), i))

    chosen = [first]
    taken = set(chosen)
    nearest = [_distance2(p, points[first]) for p in points]

    while len(chosen) < count:
        pick, best = None, None
        for i in range(n):
            if i in taken:
                continue
            if best is None or nearest[i] > best:
                best, pick = nearest[i], i
        if pick is None:
            break
        chosen.append(pick)
        taken.add(pick)
        here = points[pick]
        for i in range(n):
            d = _distance2(points[i], here)
            if d < nearest[i]:
                nearest[i] = d
    return chosen


# ---------------------------------------------------------- scoring what we know

def _goal_scores(goals, tasted):
    """Step 1's combined goal score for each tasted batch (bigger is better)."""
    if not tasted:
        return []
    if not goals:
        return [0.5] * len(tasted)
    table = step1._oriented(tasted, goals)
    return step1._scores(table, len(goals))


def _unit(values):
    """Min-max normalize a list to 0..1; a flat list becomes all 0.5."""
    if not values:
        return []
    low, high = min(values), max(values)
    span = high - low
    if span <= _TINY:
        return [0.5] * len(values)
    return [(v - low) / span for v in values]


def _next_batch(points, candidates, tasted_points, scores, weight):
    """Which untasted batch to spend the next tasting on.

    Each candidate gets a predicted goal score from the tasted batches that
    look most like it, and a novelty -- how far it sits from anything tasted.
    Both are put on a 0..1 scale across the candidates, and the winner is the
    best ``predicted + weight * novelty``.
    """
    k = min(NEIGHBOURS, len(tasted_points))
    base = _median(scores)
    if base is None:
        base = 0.0

    preds, novelty = [], []
    for i in candidates:
        vec = points[i]
        order = _sorted_neighbours(tasted_points, vec)
        preds.append(_neighbour_guess(tasted_points, scores, order, vec,
                                      k, base))
        novelty.append(order[0][0] ** 0.5 if order else 1.0)

    preds = _unit(preds)
    novelty = _unit(novelty)

    best, pick = None, candidates[0]
    for c, i in enumerate(candidates):
        acq = preds[c] + weight * novelty[c]
        if best is None or acq > best:
            best, pick = acq, i
    return pick


# ------------------------------------------------------------- spending tastings

def _ask(label, row):
    """One tasting: hand the batch to the panel and keep what comes back."""
    answer = label(row)
    if answer is None:
        answer = row
    return list(answer)


def _search(rows, label, budget):
    """Spend the budget on tastings; return the labelled rows, in order tasted."""
    header, data = _split(rows)
    if not data:
        return []
    try:
        budget = int(budget)
    except (TypeError, ValueError):
        return []
    if budget <= 0:
        return []
    budget = min(budget, len(data))

    goals, _ = _classify(header)
    readings, points = _pictures(header, data)

    if not readings or not goals:
        # Nothing to reason with: no lab readings to compare batches by, or
        # no goals to aim at.  Taste a plain prefix rather than guess.
        return [_ask(label, row) for row in data[:budget]]

    warm = max(1, min(budget, int(ceil(budget * WARMUP_SHARE))))
    tasted, tasted_points, done = [], [], set()

    for i in _spread_out(points, warm):
        tasted.append(_ask(label, data[i]))
        tasted_points.append(points[i])
        done.add(i)

    while len(tasted) < budget:
        candidates = [i for i in range(len(data)) if i not in done]
        if not candidates:
            break
        left = budget - len(tasted)
        weight = EXPLORE * (left / float(budget))
        pick = _next_batch(points, candidates, tasted_points,
                           _goal_scores(goals, tasted), weight)
        tasted.append(_ask(label, data[pick]))
        tasted_points.append(points[pick])
        done.add(pick)

    return tasted


# --------------------------------------------------------------- the two asks

def taste(rows, label, budget=20):
    """The batches to send the panel this month, in the order to taste them.

    At most ``budget`` tastings are spent.  The opening few map out the kinds
    of wine in the shed; after that each choice is guided by what the earlier
    tastings revealed, so the list closes in on the cellar's best wine
    instead of sampling blindly.
    """
    return _search(rows, label, budget)


def choose(rows, label, budget=20):
    """The best batch found: high ``Quality+``, low ``Alcohol-``.

    The same tastings as :func:`taste` are spent, then step 1 picks the
    winner among them -- Pareto front first, combined goal score breaking the
    tie -- so "best" means the same thing here as it does for past batches.
    """
    tasted = _search(rows, label, budget)
    if not tasted:
        return []

    header, _ = _split(rows)
    goals, _ = _classify(header)
    if not goals:
        return tasted[0]

    best = step1.remember([header] + tasted)
    if not best:
        return tasted[0]

    winner = best[0]
    for row in tasted:
        if list(row) == list(winner):
            return row
    return winner
