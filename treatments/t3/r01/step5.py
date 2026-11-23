"""step5.py -- active tasting for the winemaker's AI assistant.

Data convention (same as :mod:`step1` .. :mod:`step4`)
------------------------------------------------------
``rows`` is a list of lists.  ``rows[0]`` is the header (strings).  A column
name ending in ``+`` is a goal to be maximized, one ending in ``-`` is a goal
to be minimized, and anything else is a lab reading.

What is different here
----------------------
The goal cells of every data row -- ``Alcohol-`` and ``Quality+`` -- read
``'?'``.  Nobody has tasted these batches yet.  The lab readings are free (the
bench already measured them); a score is expensive, because it costs the panel
a glass and an afternoon.  ``label(row)`` is that afternoon: it hands back the
same batch with its goal cells filled in, and each call is one tasting.

Two requests are answered:

* "The panel can only taste 20 batches this month -- which 20?"   -> taste()
* "Find a batch that scores high but stays low in alcohol, while
   tasting as few batches as possible."                           -> choose()

How the 20 are picked
---------------------
Not all at once.  A tasting plan written in advance can only spread itself
evenly over the cellar; a plan that reacts can spend its later glasses near
whatever the earlier ones found.  So the budget is spent in two phases:

1. **Look around** (about the first 40%).  k-means on the standardized lab
   readings carves the cellar into as many kinds as we have opening glasses,
   and the batch nearest each centre is tasted.  This buys one honest reading
   of every corner of the cellar, including corners a greedy search would
   never have visited.
2. **Close in** (the rest).  After each glass, a Gaussian process is fitted to
   the tasted batches -- one per goal -- and asked about all 1279 batches at
   once.  It answers with both an estimate and an honest admission of how
   unsure it is.  The next glass goes to the batch with the highest
   ``estimate + beta * uncertainty``: good enough to be worth tasting, or
   unknown enough to be worth learning.  ``beta`` is wound down as the month
   runs out, so early glasses explore and the last ones simply chase the best
   wine.

Combining the two goals
-----------------------
High score and low alcohol are different units, so each goal is min-max
normalized over the batches tasted so far and flipped where the cellar wants
it small (exactly the convention :func:`step1.remember` uses).  A batch's
worth is the mean of those 0..1 numbers, 1 being best-in-cellar on everything.
Because that blend is a positively weighted sum, the per-goal estimates and
uncertainties fold into it directly, which is what makes one acquisition
score out of two models.

Cost
----
Each adaptive glass costs two Gaussian-process fits on at most 20 points plus
one prediction sweep over 1279 rows -- milliseconds.  ``taste`` and ``choose``
are each one such run of about a dozen glasses, so a whole split lands in well
under a second of compute, far inside the 90 second budget.  The tastings
themselves are capped: ``label`` is called at most ``budget`` times, never
twice on the same batch.
"""

from __future__ import annotations

import math

import step1
import step2

try:  # numpy carries the distances and the linear algebra
    import numpy as np

    _HAVE_NUMPY = True
except Exception:  # pragma: no cover - pure-Python path below still answers
    _HAVE_NUMPY = False

try:  # scikit-learn supplies the opening design and the surrogate
    from sklearn.cluster import KMeans
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        Matern,
        WhiteKernel,
    )

    _HAVE_SKLEARN = _HAVE_NUMPY
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False

DEFAULT_BUDGET = 20
WARM_FRACTION = 0.4  # share of the budget spent looking around first
MIN_WARM = 3
MAX_WARM = 10
BETA_HIGH = 1.5  # how much unexplored cellar is worth, early on
BETA_LOW = 0.25  # ... and at the end of the month
MIN_FIT = 3  # tasted batches needed before a goal can be modelled
DUP_PENALTY = 1e6  # never spend a glass on a lab sheet we already tasted


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------
class _Search:
    """One month of tasting: what has been poured, and what to pour next."""

    def __init__(self, rows):
        self.order = []  # labelled rows, in the order they were tasted
        self.tasted = {}  # row index -> labelled row
        self.values = {}  # row index -> {goal column index: score}
        self.keys = set()  # lab sheets already tasted (duplicate guard)
        self.calls = 0  # tastings spent
        self.header = []
        self.goals = []
        self.x_index = []
        self.y_index = None
        self.data = []
        self.F = []  # standardized lab sheets, as plain lists
        self.X = None  # the same thing as a numpy array, when available

        if not rows or len(rows) < 2:
            return

        header = [str(h).strip() for h in rows[0]]
        goals, readings = step1._split_header(header)
        self.header = header
        self.goals = goals
        self.y_index = step2._target_index(header)
        self.x_index = [i for i, _n, _d in readings]
        self.data = [row for row in rows[1:] if row is not None]
        if not self.data:
            return

        self._build_features()

    # -- the lab sheets --------------------------------------------------
    def _build_features(self):
        """Standardize the readings so every one counts for about as much."""
        width = len(self.x_index)
        if not width:
            self.F = [[] for _ in self.data]
            return

        fill = {i: step1._median(step1._column(self.data, i)) for i in self.x_index}
        raw = []
        for row in self.data:
            vector = []
            for i in self.x_index:
                value = step2._cell(row, i)
                vector.append(fill[i] if value is None else value)
            raw.append(vector)

        columns = list(zip(*raw))
        n = len(raw)
        means, scales = [], []
        for column in columns:
            mean = sum(column) / n
            spread = sum((v - mean) ** 2 for v in column) / max(1, n - 1)
            means.append(mean)
            scales.append(math.sqrt(spread) if spread > 1e-18 else 1.0)

        self.F = [
            [(v - means[p]) / scales[p] for p, v in enumerate(vector)]
            for vector in raw
        ]
        if _HAVE_NUMPY:
            try:
                self.X = np.asarray(self.F, dtype=float)
            except Exception:  # pragma: no cover
                self.X = None

    def _key(self, index):
        return tuple(round(v, 9) for v in self.F[index])

    def _d2(self, i, j):
        a, b = self.F[i], self.F[j]
        return sum((x - y) ** 2 for x, y in zip(a, b))

    # -- pouring a glass -------------------------------------------------
    def _pour(self, index, label):
        """Spend one tasting on `index` and fold in whatever comes back."""
        row = self.data[index]
        self.calls += 1
        try:
            labelled = label(row)
        except Exception:
            labelled = None
        if labelled is None:
            labelled = row

        self.tasted[index] = labelled
        self.order.append(labelled)
        self.keys.add(self._key(index))

        found = {}
        for i, _name, _direction in self.goals:
            value = step2._cell(labelled, i)
            if value is not None:
                found[i] = value
        if found:
            self.values[index] = found

    # -- scoring what we have --------------------------------------------
    def _bounds(self):
        """Observed low/high for each goal, over the batches tasted so far."""
        out = {}
        for found in self.values.values():
            for i, value in found.items():
                lo, hi = out.get(i, (value, value))
                out[i] = (min(lo, value), max(hi, value))
        return out

    def _worth(self, found, bounds=None):
        """A tasted batch as one 0..1 number: 1 is best-in-cellar all round."""
        if not found:
            return 0.0
        if bounds is None:
            bounds = self._bounds()
        total, used = 0.0, 0
        for i, _name, direction in self.goals:
            if i not in found or i not in bounds:
                continue
            lo, hi = bounds[i]
            span = hi - lo
            fraction = 0.5 if span <= 0 else (found[i] - lo) / span
            total += fraction if direction > 0 else 1.0 - fraction
            used += 1
        return total / used if used else 0.0

    # -- the opening glasses ---------------------------------------------
    def _warm_indices(self, budget):
        """A space-filling opening design: one batch per kind of wine."""
        count = int(round(budget * WARM_FRACTION))
        count = max(MIN_WARM, min(MAX_WARM, count))
        count = max(1, min(count, budget, len(self.data)))

        if not self.F or not self.F[0]:
            return list(range(count))

        if _HAVE_SKLEARN and self.X is not None and 2 <= count < len(self.data):
            picks = self._kmeans_indices(count)
            if picks:
                return picks
        return self._farthest(count)

    def _kmeans_indices(self, count):
        try:
            centers = KMeans(
                n_clusters=count, n_init=10, random_state=0
            ).fit(self.X).cluster_centers_
        except Exception:
            return []

        picks, used = [], set()
        for center in centers:
            try:
                d2 = ((self.X - center) ** 2).sum(axis=1)
                order = np.argsort(d2, kind="stable")
            except Exception:
                return []
            for j in order:
                j = int(j)
                if j not in used:
                    used.add(j)
                    picks.append(j)
                    break
        return picks

    def _farthest(self, count, pool=None):
        """Greedy max-min design: each pick as far as possible from the rest."""
        pool = list(range(len(self.data))) if pool is None else list(pool)
        if not pool:
            return []
        if not self.F or not self.F[0]:
            return pool[:count]

        seed = min(pool, key=lambda i: sum(v * v for v in self.F[i]))
        picks, chosen = [seed], {seed}
        nearest = {i: self._d2(i, seed) for i in pool}

        while len(picks) < count and len(chosen) < len(pool):
            best, best_d = None, -1.0
            for i in pool:
                if i in chosen:
                    continue
                if nearest[i] > best_d:
                    best, best_d = i, nearest[i]
            if best is None:
                break
            picks.append(best)
            chosen.add(best)
            for i in pool:
                if i in chosen:
                    continue
                d = self._d2(i, best)
                if d < nearest[i]:
                    nearest[i] = d
        return picks

    # -- the surrogate ---------------------------------------------------
    @staticmethod
    def _kernel():
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(1e-2, (1e-6, 1e1))
        )

    def _gp_scores(self, candidates, beta):
        """``estimate + beta * uncertainty`` of the blended worth, per batch.

        One Gaussian process per goal, each fitted only to the batches already
        tasted.  Because the blend is a positively weighted sum of normalized
        goals, the per-goal estimates add and the per-goal variances add, so
        two models still give one acquisition score.
        """
        if self.X is None or not self.values:
            return None

        try:
            Xc = self.X[candidates]
        except Exception:
            return None

        mu_sum = np.zeros(len(candidates), dtype=float)
        var_sum = np.zeros(len(candidates), dtype=float)
        used = 0

        for i, _name, direction in self.goals:
            indices = [k for k, found in self.values.items() if i in found]
            if len(indices) < MIN_FIT:
                continue
            y = np.asarray([self.values[k][i] for k in indices], dtype=float)
            lo, hi = float(y.min()), float(y.max())
            span = hi - lo
            if span <= 0:
                continue  # every glass agreed: this goal says nothing yet

            try:
                gp = GaussianProcessRegressor(
                    kernel=self._kernel(),
                    normalize_y=True,
                    n_restarts_optimizer=1,
                    random_state=0,
                )
                gp.fit(self.X[indices], y)
                mu, sd = gp.predict(Xc, return_std=True)
            except Exception:
                continue
            if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sd))):
                continue

            fraction = (mu - lo) / span if direction > 0 else (hi - mu) / span
            mu_sum += fraction
            var_sum += (sd / span) ** 2
            used += 1

        if used == 0:
            return None
        return (mu_sum + beta * np.sqrt(var_sum)) / used

    def _near_scores(self, candidates, beta):
        """Fallback surrogate: nearby tasted batches, plus a novelty bonus."""
        if self.X is None or not self.values:
            return None

        indices = list(self.values.keys())
        bounds = self._bounds()
        y = np.asarray(
            [self._worth(self.values[k], bounds) for k in indices], dtype=float
        )
        try:
            Xt = self.X[indices]
            Xc = self.X[candidates]
            d = np.sqrt(
                np.maximum(
                    ((Xc[:, None, :] - Xt[None, :, :]) ** 2).sum(axis=2), 0.0
                )
            )
        except Exception:
            return None

        weights = 1.0 / (d + 1e-6)
        mu = (weights * y[None, :]).sum(axis=1) / weights.sum(axis=1)
        nearest = d.min(axis=1)
        scale = float(np.median(nearest)) + 1e-6
        return mu + beta * nearest / (nearest + scale)

    # -- choosing the next glass -----------------------------------------
    def _next(self, beta):
        candidates = [i for i in range(len(self.data)) if i not in self.tasted]
        if not candidates:
            return None

        scores = None
        if _HAVE_SKLEARN:
            scores = self._gp_scores(candidates, beta)
        if scores is None and _HAVE_NUMPY:
            scores = self._near_scores(candidates, beta)
        if scores is None:
            picks = self._farthest(1, candidates) if self.tasted else candidates[:1]
            return picks[0] if picks else candidates[0]

        best, best_score = None, None
        for position, index in enumerate(candidates):
            score = float(scores[position])
            if self._key(index) in self.keys:
                score -= DUP_PENALTY  # same lab sheet: a glass learns nothing
            if best_score is None or score > best_score:
                best, best_score = index, score
        return best

    # -- the month -------------------------------------------------------
    def run(self, label, budget):
        """Spend up to `budget` tastings, best-informed glass each time."""
        if not self.data or not callable(label):
            return self.order

        budget = DEFAULT_BUDGET if budget is None else int(budget)
        budget = max(0, min(budget, len(self.data)))
        if budget == 0:
            return self.order

        for index in self._warm_indices(budget):
            if self.calls >= budget:
                break
            if index not in self.tasted:
                self._pour(index, label)

        adaptive = max(1, budget - self.calls)
        while self.calls < budget:
            remaining = budget - self.calls
            share = min(1.0, remaining / adaptive)
            beta = BETA_LOW + (BETA_HIGH - BETA_LOW) * share
            index = self._next(beta)
            if index is None:
                break
            self._pour(index, label)

        return self.order

    # -- the answer ------------------------------------------------------
    def best(self):
        """The finest batch tasted: highest blended worth, ties to quality."""
        if not self.order:
            return None
        if not self.values:
            return self.order[0]

        bounds = self._bounds()
        best, best_key = None, None
        for index, found in self.values.items():
            quality = found.get(self.y_index)
            key = (
                self._worth(found, bounds),
                quality if quality is not None else -math.inf,
                -index,
            )
            if best_key is None or key > best_key:
                best, best_key = index, key
        return self.tasted[best] if best is not None else self.order[0]


# --------------------------------------------------------------------------
# "which 20 should the panel taste?"
# --------------------------------------------------------------------------
def taste(rows, label, budget=DEFAULT_BUDGET):
    """Return the batches chosen for the panel, in the order they were tasted.

    ``label`` is called at most `budget` times and never twice on the same
    batch.  The first glasses are spread across the kinds of wine in the
    cellar; each later glass goes where the batches tasted so far suggest the
    best wine is hiding -- or where we are still most in the dark.
    """
    return _Search(rows).run(label, budget)


# --------------------------------------------------------------------------
# "find a batch that scores high but stays low in alcohol"
# --------------------------------------------------------------------------
def choose(rows, label, budget=DEFAULT_BUDGET):
    """Return the best batch found: high ``Quality+``, low ``Alcohol-``.

    The same budgeted search as :func:`taste` is run, then the winner is the
    tasted batch with the best blend of the two goals -- each normalized over
    everything tasted, so neither a panel point nor a percent of alcohol gets
    to dominate just by being the bigger number.  Returns ``None`` only if
    there is nothing to taste or no budget to taste it with.
    """
    search = _Search(rows)
    search.run(label, budget)
    return search.best()
