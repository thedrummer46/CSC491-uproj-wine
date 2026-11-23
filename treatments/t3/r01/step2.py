"""step2.py -- panel-score prediction for the winemaker's AI assistant.

Data convention (same as :mod:`step1`)
--------------------------------------
``rows`` is a list of lists.  ``rows[0]`` is the header (strings); every later
row holds numbers.  A column name ending in ``+`` is a goal to be maximized,
one ending in ``-`` is a goal to be minimized, and anything else is a lab
reading.

Two more questions are answered here:

* "Before the panel sits, estimate what score this new batch
   will get from its lab sheet alone."                        -> guess()
* "Group our batches into a few kinds, and tell me what
   score to expect from each kind."                           -> kinds()

Both only ever look at the ``train`` rows, so a batch being guessed at never
informs its own guess.  The lab sheet is the input: the readings are the
features, and the panel's ``Quality+`` column is what we learn to predict.
The other goal columns are things the cellar is *also* trying to steer, not
measurements of the new batch's quality, so they are kept out of the inputs.

Cost
----
``guess`` is called once per test batch with the same ``train`` rows, so the
model is fitted once and memoized; later calls are a single prediction each.
Fitting compares a handful of small scikit-learn models by cross-validation
and takes well under a second on a few hundred rows.
"""

from __future__ import annotations

import math

import step1

try:  # scikit-learn is expected, but never let a missing wheel be fatal
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - fallbacks below stay pure Python
    _HAVE_SKLEARN = False

TARGET_HINT = "quality"  # the goal the tasting panel actually awards
MIN_KINDS = 2
MAX_KINDS = 6  # "a few kinds": keep the answer readable at the dinner table
MIN_TRAIN_FOR_MODEL = 8  # below this, the cellar median is the honest answer
_CACHE_SIZE = 4


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _cell(row, index):
    """One numeric cell, or None if it is missing/unusable."""
    if row is None or index is None or index < 0 or index >= len(row):
        return None
    return step1._as_float(row[index])


def _target_index(header):
    """Index of the column the panel scores (``Quality+`` if present)."""
    goals, _readings = step1._split_header(header)
    if not goals:
        return None
    for index, name, _direction in goals:
        if name.strip().rstrip("+-").strip().lower().startswith(TARGET_HINT):
            return index
    return goals[-1][0]  # no column called quality: use the last goal


def _feature_indices(header, y_index):
    """Lab-sheet columns used as inputs.

    Readings are the lab sheet.  Only if a file has no readings at all do we
    fall back to the remaining goal columns, so that something can still be
    learned instead of returning a flat constant.
    """
    _goals, readings = step1._split_header(header)
    xs = [i for i, _n, _d in readings if i != y_index]
    if xs:
        return xs
    return [i for i, _n, _d in _goals if i != y_index]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _fingerprint(train):
    """Cheap, content-based cache key for a train set."""
    try:
        return ("rows", tuple(tuple(row) for row in train))
    except TypeError:
        return ("id", id(train), len(train))


# --------------------------------------------------------------------------
# the learned oracle
# --------------------------------------------------------------------------
def _candidates(n_rows):
    """Model factories worth trying on a few hundred wine batches."""
    k = max(1, min(7, n_rows // 10 or 1))
    return [
        lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=1, random_state=0
        ),
        lambda: ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=1, random_state=0
        ),
        lambda: make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=k, weights="distance"),
        ),
        lambda: make_pipeline(
            StandardScaler(), RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
        ),
    ]


class _Oracle:
    """Everything learned from one train set: how to score a lab sheet."""

    def __init__(self, train):
        header = [str(h).strip() for h in train[0]]
        data = [row for row in train[1:] if row is not None]

        self.header = header
        self.data = data
        self.y_index = _target_index(header)
        self.x_index = _feature_indices(header, self.y_index)

        # a missing reading becomes the cellar's typical reading
        self.fill = {i: step1._median(step1._column(data, i)) for i in self.x_index}

        vectors, targets = [], []
        for row in data:
            y = _cell(row, self.y_index)
            if y is None:
                continue
            vectors.append(self.vector(row))
            targets.append(y)

        self.typical = step1._median(targets)
        self.y_lo = min(targets) if targets else self.typical
        self.y_hi = max(targets) if targets else self.typical
        self.model = None

        if (
            _HAVE_SKLEARN
            and self.x_index
            and len(targets) >= MIN_TRAIN_FOR_MODEL
            and self.y_hi > self.y_lo
        ):
            self.model = self._fit(np.asarray(vectors, dtype=float),
                                   np.asarray(targets, dtype=float))

    # -- inputs ----------------------------------------------------------
    def vector(self, row):
        """One lab sheet as a list of numbers, gaps filled in."""
        out = []
        for i in self.x_index:
            v = _cell(row, i)
            out.append(self.fill[i] if v is None else v)
        return out

    # -- fitting ---------------------------------------------------------
    def _fit(self, X, y):
        """Pick the model that cross-validates best, then refit on all of it."""
        folds = max(2, min(5, len(y) // 2))
        cv = KFold(n_splits=folds, shuffle=True, random_state=0)

        best, best_score = None, None
        for make in _candidates(len(y)):
            try:
                model = make()
                score = cross_val_score(
                    model, X, y, cv=cv, scoring="neg_mean_absolute_error"
                ).mean()
            except Exception:
                continue
            if score is None or (isinstance(score, float) and math.isnan(score)):
                continue
            if best_score is None or score > best_score:
                best, best_score = model, score

        if best is None:
            return None
        try:
            best.fit(X, y)
        except Exception:
            return None
        return best

    # -- predicting ------------------------------------------------------
    def _clamp(self, value):
        """Never promise a score the panel has never awarded."""
        if value != value:  # NaN
            return self.typical
        return max(self.y_lo, min(self.y_hi, float(value)))

    def predict(self, row):
        if self.model is None:
            return float(self.typical)
        try:
            raw = self.model.predict(np.asarray([self.vector(row)], dtype=float))[0]
        except Exception:
            return float(self.typical)
        return self._clamp(raw)

    def predict_many(self, rows):
        if not rows:
            return []
        if self.model is None:
            return [float(self.typical)] * len(rows)
        try:
            X = np.asarray([self.vector(r) for r in rows], dtype=float)
            raw = self.model.predict(X)
        except Exception:
            return [float(self.typical)] * len(rows)
        return [self._clamp(v) for v in raw]

    def features(self, rows):
        return np.asarray([self.vector(r) for r in rows], dtype=float)


# --------------------------------------------------------------------------
# memoization: one fit per train set, however many guesses follow
# --------------------------------------------------------------------------
_cache = {}
_cache_order = []


def _oracle(train):
    """The fitted oracle for this train set (built once, then reused)."""
    if not train or len(train) < 2:
        return None

    key = _fingerprint(train)
    if key in _cache:
        return _cache[key]

    oracle = _Oracle(train)
    _cache[key] = oracle
    _cache_order.append(key)
    while len(_cache_order) > _CACHE_SIZE:
        _cache.pop(_cache_order.pop(0), None)
    return oracle


# --------------------------------------------------------------------------
# "estimate what score this new batch will get"
# --------------------------------------------------------------------------
def guess(train, row):
    """Predicted ``Quality+`` for one batch, from its lab readings alone.

    Only `train` teaches the model; `row`'s own goal columns are ignored, so
    this is a fair stand-in for the panel that has not sat yet.
    """
    oracle = _oracle(train)
    if oracle is None:
        return 0.0
    if row is None:
        return float(oracle.typical)
    return oracle.predict(row)


# --------------------------------------------------------------------------
# "group our batches into a few kinds"
# --------------------------------------------------------------------------
def _kinds_fallback(oracle, data):
    """No clustering available: split the cellar by score into three bands."""
    scored = [(oracle.predict(r) if _cell(r, oracle.y_index) is None
               else _cell(r, oracle.y_index), i, r) for i, r in enumerate(data)]
    scored.sort(key=lambda t: (-t[0], t[1]))

    groups, size = [], max(1, len(scored) // 3)
    for start in range(0, len(scored), size):
        chunk = scored[start:start + size]
        if chunk:
            groups.append(([t[2] for t in chunk], _mean([t[0] for t in chunk])))
    return groups


def _expected_quality(oracle, group):
    """What the panel should award this kind of batch, on average.

    Real panel scores are used where we have them; batches whose score is
    missing fall back to the model's estimate.
    """
    seen = [v for v in (_cell(r, oracle.y_index) for r in group) if v is not None]
    if seen:
        return float(_mean(seen))
    return float(_mean(oracle.predict_many(group)))


def kinds(train):
    """Return ``[(group_rows, expected_quality), ...]``, best kind first.

    Batches are grouped by their lab readings only -- that is the part of a
    wine we can measure before tasting -- using k-means on standardized
    readings, with the number of kinds (2..6) chosen by silhouette score.
    Each group is reported with the score to expect from batches like it.
    """
    oracle = _oracle(train)
    if oracle is None:
        return []

    data = oracle.data
    if not data:
        return []
    if len(data) == 1:
        return [(list(data), _expected_quality(oracle, data))]

    if not (_HAVE_SKLEARN and oracle.x_index):
        return _kinds_fallback(oracle, data)

    try:
        X = StandardScaler().fit_transform(oracle.features(data))
    except Exception:
        return _kinds_fallback(oracle, data)

    top = min(MAX_KINDS, len(data) - 1)
    best_labels, best_score = None, None
    for k in range(MIN_KINDS, max(MIN_KINDS, top) + 1):
        try:
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
        except Exception:
            continue
        if best_score is None or score > best_score:
            best_labels, best_score = labels, score

    if best_labels is None:
        return _kinds_fallback(oracle, data)

    buckets = {}
    for row, label in zip(data, best_labels):
        buckets.setdefault(int(label), []).append(row)

    out = [(group, _expected_quality(oracle, group))
           for _label, group in sorted(buckets.items()) if group]
    out.sort(key=lambda pair: -pair[1])  # the kinds worth making, first
    return out
