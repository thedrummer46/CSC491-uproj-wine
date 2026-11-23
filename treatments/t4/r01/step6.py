"""step6.py -- why the poor batches are poor, and what to watch.

Answers the winemaker's two diagnostic requests:

  "Which two or three readings matter most for whether a batch scores well?"
        ->  blame(train, k=3) -> [column name, ...], most important first

  "Our low-scoring batches: which lab readings are dragging them down?"
        ->  dragging(train)   -> [(column name, 'high' | 'low'), ...],
                                 worst first

Data convention is step1's: `rows` is a list of lists, row 0 is the header
(strings), every later row holds numbers; '+' marks a goal to maximize, '-'
a goal to minimize, anything else is a lab reading.  Which column the panel
awards comes from `step2._target_and_features`, cell coercion from
`step1._number`, blank-filling from `step1._median`, and column
classification from `step1._column_kinds`.  Nothing from the earlier steps
is re-derived.

Only lab readings are ever named
--------------------------------
Both answers are restricted to `step1._READING` columns, so neither
'Quality+' nor 'Alcohol-' can be returned.  This is a change from step2 and
step4, which legitimately feed 'Alcohol-' into the model as a number that is
on the lab sheet before the panel sits.  Here the question is different: the
winemaker is asking what to *adjust in the cellar*, and "your alcohol is
dragging your score down" is not advice, it is one goal restated against the
other.  So the forest behind `blame` is grown on the readings alone.

Two different questions, two different measurements
--------------------------------------------------
`blame` asks which readings *matter* -- a question about predictive power,
including the interactions and plateaus a correlation cannot see.  It is
answered by permutation importance on a random forest: shuffle one reading
in data the forest never trained on and see how much the score estimate
degrades.  Out-of-fold rather than in-sample, because a forest reproduces
its own training rows almost perfectly and in-sample shuffling would flatter
every column equally.  Impurity importance was not used: it inflates
high-cardinality continuous columns, and every column here is continuous.

`dragging` asks which readings sit *wrong* in the poor batches -- a question
about direction and size, which importance cannot answer (a reading can
matter enormously and still be fine in the batches that failed).  It is
answered by contrasting the poor batches (Quality+ below 6) against the rest
with Cliff's delta: the chance that a poor batch reads higher than a good one
on that column, rescaled to [-1, 1].  Rank-based on purpose -- residual
sugar, chlorides and the sulfur dioxides have long right tails, and a
difference of means there describes outliers rather than the normal poor
batch.  A column is reported only if the gap clears a small-effect
threshold and a rank-sum test at roughly the 5% level, so the list is the
readings actually out of place, not a ranking of all eleven.

Cost
----
`blame` fits FOLDS forests of N_TREES on ~850 rows each and permutes ten
columns N_REPEATS times per fold: a few seconds.  `dragging` is one sort per
column.  The forest work is cached under a fingerprint of the train table,
so calling both on the same table pays for it once.  A whole split is well
inside the 90s budget.
"""

from __future__ import annotations

import threading

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

import step1
import step2

# "Poor batch", as the winemaker defined it when asking what to send the
# panel: anything the panel scored below 6.
POOR_THRESHOLD = 6.0

DEFAULT_K = 3

# Forest used for the importance measurement.  Smaller than step2's: a
# ranking of ten columns settles down long before a point estimate does, and
# this one is fitted FOLDS times.
N_TREES = 300
FOLDS = 3
N_REPEATS = 5
RANDOM_STATE = step2.RANDOM_STATE

# Cliff's delta below this is a difference too small to act on (the
# conventional 'small effect' line), and a rank-sum z below MIN_Z is a
# difference this many batches cannot establish at all.
MIN_EFFECT = 0.147
MIN_Z = 2.0

_CACHE = {}
_CACHE_LIMIT = 4
_LOCK = threading.Lock()


# ----------------------------------------------------------------- columns

def _reading_indexes(header):
    """Positions of the lab readings -- never a goal column."""
    kinds = step1._column_kinds(header)
    return [i for i, kind in enumerate(kinds) if kind == step1._READING]


def _name(header, index):
    """The column's name exactly as the header spelled it."""
    return str(header[index])


def _design(data, readings, target):
    """(X, y) over the scored batches: readings only, blanks at the median.

    Rows the panel has not scored are dropped -- they carry no answer for a
    supervised importance measurement to degrade.
    """
    medians = {i: step1._median(step1._column(data, i)) for i in readings}

    X = []
    y = []
    for row in data:
        quality = (step1._number(row[target])
                   if target is not None and target < len(row) else None)
        if quality is None:
            continue
        vector = []
        for index in readings:
            value = step1._number(row[index]) if index < len(row) else None
            vector.append(medians[index] if value is None else value)
        X.append(vector)
        y.append(quality)

    if not X:
        return np.zeros((0, len(readings)), dtype=float), np.zeros(0, dtype=float)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


# -------------------------------------------------------------- statistics

def _ranks(values):
    """Average ranks (1-based), ties sharing their mean rank."""
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    out = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1] == ordered[i]:
            j += 1
        out[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return out


def _cliffs_delta(poor, good):
    """(delta, z) for one reading: poor batches against the rest.

    delta is the chance a poor batch reads higher than a good one minus the
    chance it reads lower, so positive means the poor batches run high on
    this column.  z is the normal approximation to the rank-sum statistic,
    without a tie correction -- which overstates the variance slightly and
    therefore errs toward calling a difference unremarkable.
    """
    n1 = len(poor)
    n2 = len(good)
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0

    both = np.concatenate([poor, good])
    ranked = _ranks(both)
    u = float(ranked[:n1].sum()) - n1 * (n1 + 1) / 2.0

    delta = 2.0 * u / (n1 * n2) - 1.0

    spread = np.sqrt(n1 * n2 * (n1 + n2 + 1.0) / 12.0)
    z = 0.0 if spread <= 0 else (u - n1 * n2 / 2.0) / spread
    return float(delta), float(z)


# ------------------------------------------------------------ importances

class _Diagnosis:
    """The reading-only forest's view of a train table, fitted once."""

    def __init__(self, train):
        self.header, self.data = step1._split(train)
        self.target, _features = step2._target_and_features(self.header)
        self.readings = _reading_indexes(self.header)
        self.X, self.y = _design(self.data, self.readings, self.target)
        self.importance = self._importances()

    def _importances(self):
        """Out-of-fold permutation importance per reading, or a fallback.

        Averaged over FOLDS held-out folds so the ranking does not hang on
        one lucky split.  With too few scored batches to hold anything out,
        the ranking falls back to the absolute rank correlation between the
        reading and the panel score -- monotone-only, but honest at that
        size.
        """
        count = len(self.readings)
        if count == 0 or len(self.y) == 0:
            return np.zeros(count, dtype=float)

        if len(self.y) >= 2 * FOLDS and np.std(self.y) > 0:
            totals = np.zeros(count, dtype=float)
            folds = 0
            splitter = KFold(n_splits=FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
            try:
                for fit_rows, held_rows in splitter.split(self.X):
                    forest = RandomForestRegressor(
                        n_estimators=N_TREES,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    )
                    forest.fit(self.X[fit_rows], self.y[fit_rows])
                    result = permutation_importance(
                        forest,
                        self.X[held_rows],
                        self.y[held_rows],
                        n_repeats=N_REPEATS,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    )
                    totals += result.importances_mean
                    folds += 1
            except Exception:
                folds = 0
            if folds:
                return totals / folds

        return self._rank_correlations()

    def _rank_correlations(self):
        """|Spearman| between each reading and the score, as a last resort."""
        out = np.zeros(len(self.readings), dtype=float)
        if len(self.y) < 3:
            return out
        ry = _ranks(self.y)
        ry = ry - ry.mean()
        denom_y = float(np.sqrt((ry ** 2).sum()))
        if denom_y <= 0:
            return out
        for column in range(self.X.shape[1]):
            rx = _ranks(self.X[:, column])
            rx = rx - rx.mean()
            denom_x = float(np.sqrt((rx ** 2).sum()))
            if denom_x <= 0:
                continue
            out[column] = abs(float(rx @ ry) / (denom_x * denom_y))
        return out

    # -------------------------------------------------------------- views

    def ranked_readings(self):
        """Reading names, most important for Quality+ first."""
        order = sorted(
            range(len(self.readings)),
            key=lambda c: (-self.importance[c], c),
        )
        return [_name(self.header, self.readings[c]) for c in order]

    def importance_of(self, column):
        return float(self.importance[column]) if column < len(self.importance) else 0.0

    def _split_by_quality(self):
        """(poor rows, good rows) -- below 6 and the rest.

        If the panel has never awarded 6 or more, the contrast is made
        against the better half of what it did award instead: the winemaker
        still needs to know what separates their worst batches from their
        best, even in a bad season.
        """
        if self.target is None:
            return [], []

        scored = []
        for row in self.data:
            quality = (step1._number(row[self.target])
                       if self.target < len(row) else None)
            if quality is not None:
                scored.append((quality, row))
        if not scored:
            return [], []

        poor = [row for quality, row in scored if quality < POOR_THRESHOLD]
        good = [row for quality, row in scored if quality >= POOR_THRESHOLD]

        if poor and not good:
            cut = step1._median([quality for quality, _row in scored])
            poor = [row for quality, row in scored if quality < cut]
            good = [row for quality, row in scored if quality >= cut]
        return poor, good

    def contrasts(self):
        """Per reading: (name, direction, |delta|, z, importance), unfiltered."""
        poor, good = self._split_by_quality()
        out = []
        if not poor or not good:
            return out

        for column, index in enumerate(self.readings):
            left = np.asarray(step1._column(poor, index), dtype=float)
            right = np.asarray(step1._column(good, index), dtype=float)
            if len(left) == 0 or len(right) == 0:
                continue
            delta, z = _cliffs_delta(left, right)
            out.append((
                _name(self.header, index),
                "high" if delta > 0 else "low",
                abs(delta),
                abs(z),
                self.importance_of(column),
            ))

        # Worst first: the biggest gap between a poor batch and a good one,
        # with predictive importance breaking ties between equal gaps.
        out.sort(key=lambda item: (-item[2], -item[4], item[0]))
        return out


def _diagnosis_for(train):
    key = step2._fingerprint(train)
    with _LOCK:
        found = _CACHE.get(key)
        if found is not None:
            return found

    found = _Diagnosis(train)

    with _LOCK:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.clear()          # only a handful of splits are ever live
        _CACHE[key] = found
    return found


# ------------------------------------------------------------------ public

def blame(train, k=DEFAULT_K):
    """The `k` lab readings that matter most for Quality+, most first.

    Measured by out-of-fold permutation importance on a forest grown from
    the readings alone, so the answer is always a list of lab readings --
    never 'Quality+' and never 'Alcohol-'.  Returns fewer than `k` names
    only when the table has fewer readings than that.
    """
    if not train:
        return []

    try:
        k = int(k)
    except (TypeError, ValueError):
        k = DEFAULT_K
    if k <= 0:
        return []

    return _diagnosis_for(train).ranked_readings()[:k]


def dragging(train):
    """Readings sitting too high or too low in the poor batches, worst first.

    A poor batch is one the panel scored below 6.  Each reading is contrasted
    against the batches that scored 6 or more; 'high' means the poor batches
    run above the good ones on that reading and should be brought down,
    'low' the reverse.  Only gaps big enough to act on and large enough to
    be real are listed, so an empty list is a meaningful answer: no single
    reading separates the poor batches from the good ones.
    """
    if not train:
        return []

    return [
        (name, direction)
        for name, direction, delta, z, _importance in _diagnosis_for(train).contrasts()
        if delta >= MIN_EFFECT and z >= MIN_Z
    ]


# ------------------------------------------------------------------ extras

def blame_report(train):
    """Every reading with its importance, most important first.

    For when the winemaker asks how far ahead the top two or three really
    are -- `blame` truncates the list, this one shows the whole ladder.
    """
    if not train:
        return []
    found = _diagnosis_for(train)
    pairs = [
        (_name(found.header, index), found.importance_of(column))
        for column, index in enumerate(found.readings)
    ]
    pairs.sort(key=lambda pair: (-pair[1], pair[0]))
    return pairs


def dragging_report(train):
    """`dragging` with the evidence shown, and nothing filtered out.

    [{'column', 'direction', 'effect', 'z', 'importance', 'reported'}, ...] --
    the gap as Cliff's delta, the rank-sum z behind it, the reading's
    importance from `blame`, and whether `dragging` called it out.
    """
    if not train:
        return []
    out = []
    for name, direction, delta, z, importance in _diagnosis_for(train).contrasts():
        out.append({
            "column": name,
            "direction": direction,
            "effect": delta,
            "z": z,
            "importance": importance,
            "reported": bool(delta >= MIN_EFFECT and z >= MIN_Z),
        })
    return out
