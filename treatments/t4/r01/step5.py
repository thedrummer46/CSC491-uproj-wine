"""step5.py -- deciding what the panel actually tastes, when tasting is the cost.

Answers the last two requests from the winemaker:

  "The panel can only taste 20 batches this month.  Which 20 should they
   taste to find our best wine?"
        ->  taste(rows, label, budget=20) -> the chosen batches, labelled,
                                             in the order they were tasted

  "Find me a batch that scores high but stays low in alcohol, while tasting
   as few batches as possible."
        ->  choose(rows, label, budget=20) -> the best batch found

Data convention is step1's, with one change that drives everything here: the
goal cells ('Alcohol-', 'Quality+') of every data row read '?' until that
batch is tasted.  `label(row)` is the tasting -- it returns the batch with
its goals filled in -- and every call costs one of the 20 slots the panel
has this month.  So the lab readings are free and known for all 1279 rows,
and the two numbers we care about are unknown for all of them.

What the columns mean is not re-derived: `step1._column_kinds` says which
columns are goals and which way each one points, `step1._number` coerces
cells, `step1._median` fills blank readings, and the final pick is made by
`step1.remember`, which shells the tasted batches into Pareto fronts over
the goals and then orders by distance to the ideal corner.  At a budget of
20 its 5% cap resolves to exactly one batch -- the answer `choose` owes.

Why a Gaussian process and not step2/step4's models
---------------------------------------------------
step2 fits a forest to a few hundred scored rows and step4 learns a ridge
line plus neighbours from a whole season.  Here the model never sees more
than 20 labels, and what it is asked for is not a point estimate but *where
to look next* -- which needs calibrated uncertainty on batches nothing has
been tasted near.  A forest grown on 8 rows has no honest error bars and a
ridge line on 11 correlated readings has none at all.  A GP over the
standardized lab sheet does, and 20 points in 11 dimensions is exactly the
regime it was built for.  One GP per goal, so the two goals keep their own
length scales: alcohol is far more predictable from the lab sheet than the
panel's score is.

How the 20 slots are spent
--------------------------
* The first 8 go on a spread of styles: k-means on the standardized readings
  with k = 8, tasting the batch nearest each centre (biggest style first),
  topped up greedily by farthest-point if a cluster comes back empty.  No
  surrogate can point anywhere useful before it has seen the range of wine
  the cellar makes, and a diverse 8 is the cheapest way to buy that.

* The remaining 12 are chosen one at a time by optimism in goal space.  Each
  goal's posterior is pushed kappa standard deviations in its own good
  direction, scaled into the same [0, 1] units step1 uses, and the candidate
  with the smallest distance-to-heaven on those optimistic values is tasted
  next.  kappa decays 2.0 -> 0.25 across the 12, so the month opens by
  exploring and closes by confirming -- which is what both requests want,
  since `taste` is judged on the best batch in its 20 and `choose` on the
  single batch it hands over.

Cost
----
Per adaptive step: two GP fits on at most 20 points and two predictions over
1279 rows.  A whole call is a dozen such steps plus one k-means, well under
a second; `taste` and `choose` together are a small fraction of the 90s
budget, and the 20-call ceiling is enforced by construction -- `label` is
invoked only from `_take`, which appends once per call.
"""

from __future__ import annotations

import warnings

import numpy as np

from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

import step1

BUDGET = 20

# Share of the budget spent on a diverse opening, before any model is
# trusted to aim.  0.4 of 20 is 8 batches.
SEED_FRACTION = 0.4
MIN_SEEDS = 3

# Optimism on the posterior, from the first adaptive pick to the last.
KAPPA_HI = 2.0
KAPPA_LO = 0.25

# Below this many tastings of a goal, a GP has nothing to say about it and
# the search keeps spreading out instead.
MIN_FOR_GP = 4

RANDOM_STATE = 0
_TINY = 1e-12


# ----------------------------------------------------------------- columns

def _column_roles(header):
    """(kinds, reading indexes, goal indexes) for a header row.

    Classification comes from step1, so '+' / '-' / reading means here
    exactly what it means in every earlier step.
    """
    kinds = step1._column_kinds(header)
    readings = [i for i, kind in enumerate(kinds) if kind == step1._READING]
    goals = [i for i, kind in enumerate(kinds) if kind != step1._READING]
    return kinds, readings, goals


def _matrix(data, readings):
    """The free part of the table: standardized lab readings, all rows.

    Readings are known before tasting, so this is built once and never
    changes.  A blank or non-numeric reading falls back to its column
    median; a column with no spread is dropped to zero rather than blown up
    by the division.
    """
    if not readings:
        return np.zeros((len(data), 1), dtype=float)

    medians = {i: step1._median(step1._column(data, i)) for i in readings}

    raw = np.empty((len(data), len(readings)), dtype=float)
    for r, row in enumerate(data):
        for c, index in enumerate(readings):
            value = step1._number(row[index]) if index < len(row) else None
            raw[r, c] = medians[index] if value is None else value

    centre = raw.mean(axis=0)
    spread = raw.std(axis=0)
    scale = np.where(spread > _TINY, 1.0 / np.maximum(spread, _TINY), 0.0)
    return (raw - centre) * scale


# --------------------------------------------------------------- spreading

def _next_diverse(X, picked, available):
    """The untasted batch farthest from everything tasted so far.

    With nothing tasted yet it returns the most typical batch -- the one
    nearest the centre of the cellar's chemistry -- which is a better first
    tasting than an outlier.
    """
    if not available.any():
        return None

    if not picked:
        score = -((X - X.mean(axis=0)) ** 2).sum(axis=1)
    else:
        score = np.full(len(X), np.inf)
        for i in picked:
            score = np.minimum(score, ((X - X[i]) ** 2).sum(axis=1))

    score = np.where(available, score, -np.inf)
    return int(np.argmax(score))


def _seed_order(X, count):
    """`count` opening tastings that cover the range of styles on hand.

    k-means on the readings, one batch per cluster (largest cluster first,
    so the styles the cellar actually makes most of are tasted first),
    topped up by farthest-point if clustering returns fewer distinct
    representatives than asked for.
    """
    n = len(X)
    count = max(0, min(count, n))
    if count == 0:
        return []

    picks = []
    available = np.ones(n, dtype=bool)

    if 1 < count < n:
        try:
            model = KMeans(n_clusters=count, n_init=10,
                           random_state=RANDOM_STATE).fit(X)
            labels = model.labels_
            sizes = [(int(np.count_nonzero(labels == c)), c)
                     for c in range(count)]
            for _size, c in sorted(sizes, reverse=True):
                order = np.argsort(((X - model.cluster_centers_[c]) ** 2)
                                   .sum(axis=1))
                for i in order:
                    i = int(i)
                    if available[i]:
                        picks.append(i)
                        available[i] = False
                        break
        except Exception:               # degenerate table: spread instead
            picks = []
            available = np.ones(n, dtype=bool)

    while len(picks) < count:
        i = _next_diverse(X, picks, available)
        if i is None:
            break
        picks.append(i)
        available[i] = False

    return picks[:count]


# -------------------------------------------------------------- surrogates

def _kernel():
    """Smooth-but-not-analytic trend, plus a term for tasting noise.

    Matern(nu=2.5) rather than an RBF: panel scores are not infinitely
    smooth in chemistry, and the white noise term is what stops the GP from
    claiming zero uncertainty at a batch two tasters would score apart.
    """
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(1e-2, (1e-6, 1e1))
    )


class _Surrogate:
    """One goal's posterior over the untasted batches.

    Falls back to a constant mean with the observed spread as its error bar
    whenever there are too few tastings, or too little variation among them,
    to fit anything -- which keeps the acquisition step purely exploratory
    early on instead of chasing a fitted artefact.
    """

    def __init__(self, X, y):
        self.mean = float(np.mean(y)) if len(y) else 0.0
        spread = float(np.std(y)) if len(y) else 0.0
        self.spread = spread if spread > _TINY else 1.0
        self.gp = None

        if len(y) < MIN_FOR_GP or spread <= _TINY:
            return

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp = GaussianProcessRegressor(
                    kernel=_kernel(),
                    normalize_y=True,
                    n_restarts_optimizer=1,
                    random_state=RANDOM_STATE,
                )
                gp.fit(X, y)
            self.gp = gp
        except Exception:
            self.gp = None

    def posterior(self, X):
        """(mean, standard deviation) for every candidate batch."""
        if self.gp is None:
            return (np.full(len(X), self.mean),
                    np.full(len(X), self.spread))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sd = self.gp.predict(X, return_std=True)
        mu = np.where(np.isfinite(mu), mu, self.mean)
        sd = np.where(np.isfinite(sd), sd, self.spread)
        return mu, np.maximum(sd, 0.0)


def _goal_scale(values):
    """(floor, span) mapping a goal's tasted values onto step1's [0, 1].

    The span is the observed range, or the spread, or 1.0 -- whatever is the
    first thing that is not zero, so a month that has so far tasted one
    score still produces usable units.
    """
    lo = float(min(values))
    hi = float(max(values))
    span = hi - lo
    if span > _TINY:
        return lo, span
    spread = float(np.std(values))
    return lo, (spread if spread > _TINY else 1.0)


def _acquire(X, tasted, data, kinds, goals, available, kappa):
    """The untasted batch that looks best once optimism is allowed.

    Each goal is modelled on the batches tasted so far, its posterior pushed
    kappa standard deviations the way the winemaker wants that goal to move,
    and the result scaled into the same units step1 normalizes goals into.
    The pick is the smallest distance-to-heaven over those optimistic
    values -- step1's own definition, vectorized over all 1279 candidates.
    """
    rows = [row for _i, row in tasted]
    units = []

    for index in goals:
        values = []
        keep = []
        for position, row in enumerate(rows):
            value = (step1._number(row[index])
                     if index < len(row) else None)
            if value is not None:
                values.append(value)
                keep.append(tasted[position][0])
        if not values:
            continue

        lo, span = _goal_scale(values)
        surrogate = _Surrogate(X[keep], np.asarray(values, dtype=float))
        mu, sd = surrogate.posterior(X)

        if kinds[index] == step1._MINIMIZE:
            optimistic = 1.0 - (mu - kappa * sd - lo) / span
        else:
            optimistic = (mu + kappa * sd - lo) / span
        units.append(optimistic)

    if not units:
        return _next_diverse(X, [i for i, _row in tasted], available)

    U = np.vstack(units)
    gap = np.sqrt(((1.0 - U) ** 2).mean(axis=0))
    gap = np.where(np.isfinite(gap), gap, np.inf)
    gap = np.where(available, gap, np.inf)
    if not np.isfinite(gap).any():
        return _next_diverse(X, [i for i, _row in tasted], available)
    return int(np.argmin(gap))


# ------------------------------------------------------------------ search

def _take(data, index, label, tasted, available):
    """Spend one tasting on one batch and record what came back.

    The original row object is handed to `label`, not a copy, so a `label`
    that works by identity or mutates in place behaves as its author
    intended; if it returns nothing, the row itself is kept (it may have
    been filled in place).
    """
    row = data[index]
    try:
        result = label(row)
    except Exception:
        result = None
    if result is None:
        result = row
    tasted.append((index, list(result)))
    available[index] = False


def _search(rows, label, budget):
    """(header, labelled rows in tasting order, their row positions).

    Never calls `label` more than `budget` times, and never twice on the
    same batch.
    """
    header, data = step1._split(rows)
    if not data or label is None:
        return header, [], []

    try:
        budget = int(budget)
    except (TypeError, ValueError):
        budget = BUDGET
    budget = min(max(budget, 0), len(data))
    if budget == 0:
        return header, [], []

    kinds, readings, goals = _column_roles(header)
    X = _matrix(data, readings)

    seeds = min(budget, max(MIN_SEEDS, int(SEED_FRACTION * budget)))
    if not goals:
        # Nothing to optimize: the most informative 20 batches are simply a
        # spread across the cellar's chemistry.
        seeds = budget

    tasted = []
    available = np.ones(len(data), dtype=bool)

    for index in _seed_order(X, seeds):
        _take(data, index, label, tasted, available)

    remaining = budget - len(tasted)
    for step in range(max(remaining, 0)):
        if not available.any():
            break
        kappa = (KAPPA_HI if remaining <= 1 else
                 KAPPA_HI + (KAPPA_LO - KAPPA_HI) * step / (remaining - 1.0))
        index = _acquire(X, tasted, data, kinds, goals, available, kappa)
        if index is None:
            break
        _take(data, index, label, tasted, available)

    return header, [row for _i, row in tasted], [i for i, _row in tasted]


def _best(header, labelled):
    """The one batch to hand over: high Quality+, low Alcohol-.

    Decided by step1.remember over the tasted batches, which fronts them by
    Pareto dominance in goal space -- so a batch is not punished for trading
    score against alcohol -- and then orders by distance to the ideal
    corner.  At a budget of 20 its 5% cap returns exactly one batch.
    """
    if not labelled:
        return None

    _kinds, _readings, goals = _column_roles(header)

    scored = [
        row for row in labelled
        if all(index < len(row) and step1._number(row[index]) is not None
               for index in goals)
    ]
    pool = scored if scored else labelled

    picked = step1.remember([list(header)] + [list(row) for row in pool])
    if picked:
        return picked[0]
    return list(pool[0])


# ------------------------------------------------------------------ public

def taste(rows, label, budget=BUDGET):
    """The batches to send the panel this month, in the order to taste them.

    Opens with a spread of styles, then lets the two goal models aim the
    rest of the month at wines that score well while staying low in alcohol.
    Returns the labelled rows -- at most `budget` of them, one per tasting.
    """
    _header, labelled, _indexes = _search(rows, label, budget)
    return labelled


def choose(rows, label, budget=BUDGET):
    """The best batch found inside the tasting budget.

    Runs the same search `taste` does and hands back its single best wine:
    high Quality+, low Alcohol-.  Returns None only when there is nothing
    to taste at all.
    """
    header, labelled, _indexes = _search(rows, label, budget)
    return _best(header, labelled)


# ------------------------------------------------------------------ extras

def taste_report(rows, label, budget=BUDGET):
    """`taste` with the bookkeeping shown, for explaining a month's plan.

    {'tasted', 'indexes', 'best', 'seeds'} -- the labelled batches in order,
    their 0-based positions among the data rows, the batch `choose` would
    hand over, and how many of the tastings went on the diverse opening.
    """
    header, labelled, indexes = _search(rows, label, budget)
    seeds = 0
    if labelled:
        _kinds, _readings, goals = _column_roles(header)
        seeds = (len(labelled) if not goals else
                 min(len(labelled),
                     max(MIN_SEEDS, int(SEED_FRACTION * len(labelled)))))
    return {
        "tasted": labelled,
        "indexes": indexes,
        "best": _best(header, labelled),
        "seeds": seeds,
    }
