"""step6.py -- explaining the panel's verdict for the winemaker's AI assistant.

Data convention (same as :mod:`step1` .. :mod:`step5`)
------------------------------------------------------
``rows`` is a list of lists.  ``rows[0]`` is the header (strings); every later
row holds numbers.  A column name ending in ``+`` is a goal to be maximized,
one ending in ``-`` is a goal to be minimized, and anything else is a lab
reading.  The panel's ``Quality+`` column is the score being explained.

Two more questions are answered here -- both about *why*, not *what*:

* "Which two or three readings matter most for whether a batch
   scores well?"                                              -> blame()
* "Our low-scoring batches: which lab readings are dragging
   them down?"                                                -> dragging()

Only lab readings are ever named.  The other goal columns (``Alcohol-``) are
things the cellar is also steering, not explanations of the score, so they are
kept out of both answers -- as is ``Quality+`` itself, which would otherwise
"explain" itself perfectly and say nothing useful at the dinner table.

How "matters most" is measured
------------------------------
Not by how big a number the reading is, and not by a single straight-line
correlation -- panel scores do not move in straight lines.  A random forest is
fitted on the readings, and then each reading in turn is *shuffled* among the
batches and the forest is asked again.  Scrambling a reading that the panel
responds to wrecks the predictions; scrambling one that does not costs
nothing.  The damage done is the reading's importance.  The shuffling is
always done on held-out folds, so a reading cannot earn importance merely by
helping the forest memorize the batches it was fitted on.

Two honest caveats:

* This measures *association as the forest uses it*, not causation.  Two
  readings that move together (a chemist would expect acidity and pH to) split
  the credit between them, so each can look weaker than the pair really is.
* Importance says a reading matters; it does not say which way to turn the
  dial.  That is what :func:`dragging` is for.

How "dragging them down" is decided
-----------------------------------
The cellar is split in two at the score it cares about -- poor batches score
below 6, the rest are fine (the same cutoff :mod:`step4` uses to decide what
is worth sending to the panel).  For each reading, the typical value among the
poor batches is compared with the typical value among the good ones, measured
in standard deviations of that reading so that sulphates and total SO2 can be
put in the same list.  Higher in the poor batches is reported ``'high'``,
lower is reported ``'low'``.

That gap is then weighted by the reading's importance from :func:`blame`.  A
reading can differ between poor and good batches and still be beside the
point -- a coincidence of which barrels happened to be filled when -- and the
winemaker should not be sent to adjust it.  Weighting keeps the top of the
list to readings that both differ *and* matter.  Readings whose gap is too
small to act on are left off the list entirely.

Cost
----
One forest fit per fold plus one prediction sweep per reading per shuffle:
5 folds x 11 readings x 3 shuffles on 1279 rows is a couple of seconds, and
the whole analysis is memoized per train set so that ``blame`` and
``dragging`` share it.  Well inside the 90 second budget.
"""

from __future__ import annotations

import math

import step1
import step2

try:  # scikit-learn carries the forest; without it, rank correlation answers
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - the pure-Python path below still works
    _HAVE_SKLEARN = False

POOR_CUTOFF = 6.0  # "a poor batch" = a panel score below 6 (cf. step4)
DEFAULT_K = 3  # "two or three readings"
TREES = 150  # enough for a stable importance on a few hundred rows
PERM_REPEATS = 3  # shuffles per reading per fold
MIN_GROUP = 5  # poor/good batches needed before the two can be compared
MIN_EFFECT = 0.10  # smaller than a tenth of a spread: not worth acting on
WEIGHT_FLOOR = 0.25  # an unimportant reading still counts for something
MIN_ROWS_FOR_FOREST = 20
_CACHE_SIZE = 4
_EPS = 1e-12


# --------------------------------------------------------------------------
# header helpers
# --------------------------------------------------------------------------
def _reading_indices(header, y_index):
    """Indices of the lab readings only -- never a goal column.

    Unlike :func:`step2._feature_indices` this never falls back to the goal
    columns: here, naming a goal column as an explanation would be wrong, so a
    file with no readings at all honestly has nothing to say.
    """
    _goals, readings = step1._split_header(header)
    return [i for i, _n, _d in readings if i != y_index]


# --------------------------------------------------------------------------
# small statistics, written out so the pure-Python path needs nothing
# --------------------------------------------------------------------------
def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _sd(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = _mean(values)
    spread = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(spread, 0.0))


def _rank(values):
    """Average ranks (ties share their mean rank), for rank correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while (
            stop + 1 < len(order)
            and values[order[stop + 1]] == values[order[start]]
        ):
            stop += 1
        shared = (start + stop) / 2.0 + 1.0
        for position in range(start, stop + 1):
            ranks[order[position]] = shared
        start = stop + 1
    return ranks


def _pearson(a, b):
    n = len(a)
    if n < 3:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    sa = sum((v - ma) ** 2 for v in a)
    sb = sum((v - mb) ** 2 for v in b)
    if sa <= _EPS or sb <= _EPS:
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(sa * sb)


def _spearman(a, b):
    """Monotone association, immune to the odd outlying barrel."""
    return _pearson(_rank(a), _rank(b))


# --------------------------------------------------------------------------
# everything learned about "why", for one train set
# --------------------------------------------------------------------------
class _Explainer:
    """Reading importances, plus how the poor batches differ from the good."""

    def __init__(self, train):
        header = [str(h).strip() for h in train[0]]
        data = [row for row in train[1:] if row is not None]

        self.header = header
        self.y_index = step2._target_index(header)
        self.x_index = _reading_indices(header, self.y_index)
        self.names = [header[i] for i in self.x_index]

        # a blank cell becomes the cellar's typical reading, as elsewhere
        self.fill = {i: step1._median(step1._column(data, i)) for i in self.x_index}

        table, targets = [], []
        for row in data:
            y = step2._cell(row, self.y_index)
            if y is None:
                continue  # unscored batch: nothing to explain with
            table.append(self._vector(row))
            targets.append(y)

        self.table = table
        self.targets = targets
        self.columns = [
            [row[position] for row in table] for position in range(len(self.x_index))
        ]
        self.association = [
            abs(_spearman(column, targets)) if targets else 0.0
            for column in self.columns
        ]
        self.importance = self._importance()

    # -- inputs ----------------------------------------------------------
    def _vector(self, row):
        out = []
        for i in self.x_index:
            value = step2._cell(row, i)
            out.append(self.fill[i] if value is None else value)
        return out

    # -- "which readings matter?" ---------------------------------------
    def _importance(self):
        """One non-negative weight per reading, summing to 1.

        Shuffle-the-reading damage where a forest can be fitted; otherwise the
        strength of each reading's rank association with the score.  Either
        way the scale is arbitrary -- only the ordering and the relative sizes
        are used.
        """
        width = len(self.x_index)
        if width == 0 or not self.targets:
            return []

        gains = None
        if (
            _HAVE_SKLEARN
            and len(self.targets) >= MIN_ROWS_FOR_FOREST
            and max(self.targets) > min(self.targets)
        ):
            gains = self._shuffle_damage()

        if gains is None:
            gains = list(self.association)

        gains = [g if g > 0.0 else 0.0 for g in gains]
        total = sum(gains)
        if total <= _EPS:
            # nothing distinguished itself: say so by refusing to rank, and
            # let the association ordering below break the tie
            return [1.0 / width] * width
        return [g / total for g in gains]

    def _shuffle_damage(self):
        """Mean out-of-fold error increase caused by scrambling each reading."""
        try:
            X = np.asarray(self.table, dtype=float)
            y = np.asarray(self.targets, dtype=float)
        except Exception:
            return None

        folds = max(2, min(5, len(y) // 2))
        rng = np.random.default_rng(0)
        damage = np.zeros(X.shape[1], dtype=float)
        rounds = 0

        try:
            cv = KFold(n_splits=folds, shuffle=True, random_state=0)
            splits = list(cv.split(X))
        except Exception:
            return None

        for train_idx, test_idx in splits:
            try:
                forest = RandomForestRegressor(
                    n_estimators=TREES, random_state=0, n_jobs=-1
                )
                forest.fit(X[train_idx], y[train_idx])
                held = X[test_idx].copy()
                truth = y[test_idx]
                base = float(np.mean(np.abs(forest.predict(held) - truth)))
            except Exception:
                continue

            scale = max(base, 1e-9)  # report damage relative to our own error
            for column in range(X.shape[1]):
                keep = held[:, column].copy()
                worse = 0.0
                try:
                    for _ in range(PERM_REPEATS):
                        held[:, column] = rng.permutation(keep)
                        worse += float(
                            np.mean(np.abs(forest.predict(held) - truth))
                        )
                finally:
                    held[:, column] = keep
                damage[column] += (worse / PERM_REPEATS - base) / scale
            rounds += 1

        if rounds == 0:
            return None
        damage /= rounds
        if not np.all(np.isfinite(damage)):
            return None
        return [float(v) for v in damage]

    # -- the ranking -----------------------------------------------------
    def ranking(self):
        """Reading names, most important to the panel's score first.

        Ties are broken by plain rank association and then by column order, so
        the answer is the same every time it is asked for.
        """
        if not self.names:
            return []
        order = sorted(
            range(len(self.names)),
            key=lambda p: (-self.importance[p], -self.association[p], p),
        )
        return [self.names[p] for p in order]

    # -- "what is dragging the poor batches down?" ----------------------
    def _groups(self):
        poor = [p for p, y in enumerate(self.targets) if y < POOR_CUTOFF]
        good = [p for p, y in enumerate(self.targets) if y >= POOR_CUTOFF]
        return poor, good

    def _weights(self):
        """Importance rescaled to 0..1 against the most important reading."""
        if not self.importance:
            return []
        top = max(self.importance)
        if top <= _EPS:
            return [1.0] * len(self.importance)
        return [w / top for w in self.importance]

    def faults(self):
        """``[(name, 'high'|'low', severity), ...]``, worst first.

        ``severity`` is the poor-vs-good gap in standard deviations of that
        reading, weighted by how much the reading matters to the score.
        """
        if not self.names or not self.targets:
            return []

        poor, good = self._groups()
        weights = self._weights()
        compare = len(poor) >= MIN_GROUP and len(good) >= MIN_GROUP

        out = []
        for position, name in enumerate(self.names):
            column = self.columns[position]
            spread = _sd(column)
            if spread <= _EPS:
                continue  # a reading that never varies drags nothing anywhere

            if compare:
                poor_typical = step1._median([column[p] for p in poor])
                good_typical = step1._median([column[p] for p in good])
                effect = (poor_typical - good_typical) / spread
            else:
                # Not enough of one kind of batch to contrast.  Fall back to
                # the direction of the association: a reading that rises as
                # the score falls is one the poor batches have too much of.
                effect = -_spearman(column, self.targets)

            if abs(effect) <= _EPS:
                continue
            severity = abs(effect) * (
                WEIGHT_FLOOR + (1.0 - WEIGHT_FLOOR) * weights[position]
            )
            direction = "high" if effect > 0 else "low"
            out.append((name, direction, severity, abs(effect), position))

        out.sort(key=lambda t: (-t[2], -t[3], t[4]))

        actionable = [t for t in out if t[3] >= MIN_EFFECT]
        if not actionable and out:
            # Nothing clears the bar for "worth adjusting"; name the single
            # worst reading rather than pretend the poor batches are a mystery.
            actionable = out[:1]
        return [(name, direction, severity) for name, direction, severity, _e, _p in actionable]


# --------------------------------------------------------------------------
# memoization: one analysis per train set, shared by both answers
# --------------------------------------------------------------------------
_cache = {}
_cache_order = []


def _explainer(train):
    """The fitted explainer for this train set (built once, then reused)."""
    if not train or len(train) < 2:
        return None

    key = step2._fingerprint(train)
    if key in _cache:
        return _cache[key]

    explainer = _Explainer(train)
    _cache[key] = explainer
    _cache_order.append(key)
    while len(_cache_order) > _CACHE_SIZE:
        _cache.pop(_cache_order.pop(0), None)
    return explainer


# --------------------------------------------------------------------------
# "which two or three readings matter most?"
# --------------------------------------------------------------------------
def blame(train, k=DEFAULT_K):
    """Return the `k` lab readings that matter most for ``Quality+``.

    Most important first.  Only lab readings are named -- never ``Quality+``
    and never another goal column such as ``Alcohol-`` -- so the answer is a
    list of dials the cellar can actually turn.  Fewer than `k` names come
    back only when the file has fewer readings than that.
    """
    explainer = _explainer(train)
    if explainer is None:
        return []

    try:
        count = DEFAULT_K if k is None else int(k)
    except (TypeError, ValueError):
        count = DEFAULT_K
    if count <= 0:
        return []

    return explainer.ranking()[:count]


# --------------------------------------------------------------------------
# "what is dragging our low-scoring batches down?"
# --------------------------------------------------------------------------
def dragging(train):
    """Return ``[(reading name, 'high'|'low'), ...]`` for the poor batches.

    A batch is poor when the panel gave it less than 6.  Each pair says which
    way that reading sits in the poor batches compared with the good ones:
    ``'high'`` means the poor batches have too much of it, ``'low'`` means too
    little.  Worst offender first, where "worst" combines how far the reading
    strays with how much it matters to the score; readings whose gap is too
    small to act on are left out.
    """
    explainer = _explainer(train)
    if explainer is None:
        return []
    return [(name, direction) for name, direction, _severity in explainer.faults()]
