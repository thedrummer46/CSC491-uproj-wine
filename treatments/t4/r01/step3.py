"""step3.py -- is the difference real, and how often is the guess right?

Answers the last two requests from the winemaker:

  "When you say batch A will beat batch B, tell me whether that's a real
   difference or just tasting noise."
        ->  certify(train, a, b) -> 'A' | 'B' | 'same'

  "How often are your score estimates right on batches you've never seen
   before?"
        ->  accuracy(train, test) -> float

Data convention is step1's and the quality model is step2's: nothing here is
re-derived.  `step2._model_for` supplies the fitted forest, the feature
columns (target column skipped by position, so a labelled row can be handed
in safely) and the train medians used to impute blanks.

How 'same' is decided
---------------------
Two things have to hold before a winner is named.

* Effect size.  The predicted gap must clear a threshold built from the
  model's own out-of-sample error: `max(MIN_GAP, NOISE_FRACTION * sigma)`,
  where sigma is a robust spread of cross-validated residuals on the train
  table.  A gap the model cannot reliably resolve on batches it has never
  seen is not a gap the panel will confirm.

* Agreement.  The forest is an ensemble of 400 trees grown on different
  bootstrap samples and different feature subsets, so the *paired* gap
  (tree by tree, same tree judging both batches) is a free picture of how
  much the answer depends on which data the model happened to see.  At
  least AGREEMENT of the trees must put the same batch in front.

Either test failing means 'same' -- the honest answer to "real difference or
just tasting noise?" is that the evidence does not separate the two batches.

Cost
----
certify is called for 500 pairs against one train table, so the expensive
work happens once and is cached: the step2 fit, the noise estimate, and a
flat (tree, node) -> value table.  Per-tree predictions then come from one
`forest.apply` call plus numpy indexing instead of 400 python-level
`predict` calls, and each row's tree vector is memoized by feature vector
(500 pairs are drawn from only a few hundred distinct test batches).  A
whole split -- fit, noise estimate, 500 certifications and one 320-row
accuracy pass -- runs in a few seconds, well inside the 90s budget.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

import step1
import step2

RANDOM_STATE = step2.RANDOM_STATE

# A gap smaller than this is below the resolution of a panel score no matter
# what the residuals say: a quarter of a point on a coarse integer scale.
MIN_GAP = 0.25

# How much of the model's own out-of-sample error a gap has to clear.
NOISE_FRACTION = 0.5

# Share of trees that must agree on the winner.
AGREEMENT = 0.75

# Folds and forest size used only for the noise estimate; a smaller forest is
# plenty for a spread and keeps the one-off cost down.
NOISE_FOLDS = 5
NOISE_TREES = 150

# Used when there is nothing to estimate a spread from at all (half a point).
DEFAULT_SIGMA = 0.5

_CACHE = {}
_CACHE_LIMIT = 4
_ROW_CACHE_LIMIT = 8192
_LOCK = threading.Lock()


# --------------------------------------------------------------- rounding

def _round(value):
    """Round half away from zero -- what a person means by 'rounds to'.

    numpy's rint rounds halves to even, which would quietly send 4.5 and 5.5
    to the same score.
    """
    if value is None:
        return None
    if value >= 0:
        return math.floor(value + 0.5)
    return -math.floor(-value + 0.5)


# ------------------------------------------------------------ noise scale

def _robust_sigma(residuals):
    """A spread that a handful of wild batches cannot inflate."""
    if len(residuals) == 0:
        return None
    centre = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - centre)))
    sigma = 1.4826 * mad
    if sigma > 0:
        return sigma
    sigma = float(np.std(residuals))
    return sigma if sigma > 0 else None


def _noise_scale(model):
    """Robust spread of the quality model's out-of-sample residuals.

    Cross-validated rather than in-sample: a random forest fits its own
    training rows almost perfectly, so in-sample residuals would say the
    model is far more certain than it is on a new batch.
    """
    if model.forest is None or len(model.labelled) < 2 * NOISE_FOLDS:
        targets = [item[1] for item in model.labelled]
        sigma = _robust_sigma(np.asarray(targets, dtype=float))
        return sigma if sigma is not None else DEFAULT_SIGMA

    X = np.asarray([item[0] for item in model.labelled], dtype=float)
    y = np.asarray([item[1] for item in model.labelled], dtype=float)

    folds = min(NOISE_FOLDS, len(y))
    try:
        predicted = cross_val_predict(
            RandomForestRegressor(
                n_estimators=NOISE_TREES,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            X,
            y,
            cv=KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE),
        )
    except Exception:               # degenerate table: fall back to the labels
        sigma = _robust_sigma(y)
        return sigma if sigma is not None else DEFAULT_SIGMA

    sigma = _robust_sigma(y - predicted)
    return sigma if sigma is not None else DEFAULT_SIGMA


# ------------------------------------------------------------- judge cache

class _Judge:
    """A step2 model plus what it takes to say how sure it is."""

    def __init__(self, train):
        self.model = step2._model_for(train)
        self.sigma = _noise_scale(self.model)
        self.threshold = max(MIN_GAP, NOISE_FRACTION * self.sigma)

        forest = self.model.forest
        self.n_trees = 0 if forest is None else len(forest.estimators_)
        self.leaf_values = None
        self.tree_index = None
        self._rows = {}

        if forest is None or self.n_trees == 0:
            return

        # Flat (tree, node) -> node mean table, so per-tree predictions are
        # one apply() plus a fancy index rather than a loop over 400 trees.
        max_nodes = max(e.tree_.node_count for e in forest.estimators_)
        table = np.zeros((self.n_trees, max_nodes), dtype=float)
        for t, estimator in enumerate(forest.estimators_):
            tree = estimator.tree_
            table[t, : tree.node_count] = tree.value[:, 0, 0]
        self.leaf_values = table
        self.tree_index = np.arange(self.n_trees)

        if not self._table_agrees():
            # Some sklearn version stores node values differently than
            # assumed; drop to the slow-but-certain path.
            self.leaf_values = None
            self.tree_index = None

    def _table_agrees(self):
        """Check the leaf table reproduces forest.predict on a few train rows."""
        sample = np.asarray([item[0] for item in self.model.labelled[:8]],
                            dtype=float)
        if len(sample) == 0:
            return True
        try:
            mine = self._tree_predictions(sample).mean(axis=1)
            theirs = self.model.forest.predict(sample)
        except Exception:
            return False
        return bool(np.allclose(mine, theirs, atol=1e-6))

    # ------------------------------------------------------------ scoring

    def _tree_predictions(self, X):
        """(n_rows, n_trees) array of one prediction per tree per row."""
        if self.leaf_values is not None:
            leaves = self.model.forest.apply(X)
            return self.leaf_values[self.tree_index[None, :], leaves]
        return np.column_stack(
            [e.predict(X) for e in self.model.forest.estimators_]
        )

    def votes(self, row):
        """Every tree's opinion of one batch, memoized by feature vector."""
        vector = step2._vector(list(row), self.model.features,
                               self.model.medians)
        key = tuple(vector)
        with _LOCK:
            cached = self._rows.get(key)
        if cached is not None:
            return cached

        votes = self._tree_predictions(
            np.asarray([vector], dtype=float)
        )[0]

        with _LOCK:
            if len(self._rows) >= _ROW_CACHE_LIMIT:
                self._rows.clear()
            self._rows[key] = votes
        return votes

    def decide(self, a, b):
        """'A', 'B' or 'same' for two batches, with the reasoning available."""
        if self.model.target is None or self.model.forest is None:
            # Nothing was ever learned: every pair is indistinguishable.
            return "same", 0.0, 0.5

        votes_a = self.votes(a)
        votes_b = self.votes(b)
        gap = float(votes_a.mean() - votes_b.mean())

        paired = votes_a - votes_b
        ties = float(np.count_nonzero(paired == 0.0))
        wins_a = (float(np.count_nonzero(paired > 0.0)) + 0.5 * ties) / len(paired)

        if abs(gap) < self.threshold:
            return "same", gap, wins_a
        if gap > 0 and wins_a >= AGREEMENT:
            return "A", gap, wins_a
        if gap < 0 and (1.0 - wins_a) >= AGREEMENT:
            return "B", gap, wins_a
        return "same", gap, wins_a


def _judge_for(train):
    key = step2._fingerprint(train)
    with _LOCK:
        judge = _CACHE.get(key)
        if judge is not None:
            return judge

    judge = _Judge(train)

    with _LOCK:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.clear()          # only a handful of splits are ever live
        _CACHE[key] = judge
    return judge


# ------------------------------------------------------------------ public

def certify(train, a, b):
    """Which of two batches will score better, or 'same' if it is noise.

    Returns 'A' when batch `a` is predicted to beat batch `b` by more than
    the model's own out-of-sample error can explain *and* at least
    AGREEMENT of the forest's trees agree, 'B' in the mirror case, and
    'same' whenever either test fails.  Both batches may be bare lab sheets
    or fully labelled rows; their Quality+ cell is never read.
    """
    if not train or a is None or b is None:
        return "same"
    verdict, _gap, _agreement = _judge_for(train).decide(list(a), list(b))
    return verdict


def accuracy(train, test):
    """Share of unseen batches whose guessed score rounds to the true one.

    Only scored test batches count toward the share -- an unscored batch has
    no answer to be right about.  Returns 0.0 if there is nothing to score.
    """
    if not train or not test:
        return 0.0

    judge = _judge_for(train)
    model = judge.model
    if model.target is None:
        return 0.0

    _header, data = step1._split(test)
    if not data:
        return 0.0

    truths = []
    scored = []
    for row in data:
        truth = (step1._number(row[model.target])
                 if model.target < len(row) else None)
        if truth is None:
            continue
        truths.append(truth)
        scored.append(row)

    if not scored:
        return 0.0

    predictions = model.predict_many(scored)   # one batched forest.predict
    hits = sum(
        1
        for predicted, truth in zip(predictions, truths)
        if _round(float(predicted)) == _round(float(truth))
    )
    return hits / float(len(truths))


# ------------------------------------------------------------------ extras

def noise_scale(train):
    """The spread of the model's out-of-sample error, in panel-score points.

    This is the number `certify` measures gaps against; handy when the
    winemaker asks how big a difference has to be before it means anything.
    """
    if not train:
        return DEFAULT_SIGMA
    return float(_judge_for(train).sigma)


def certify_report(train, a, b):
    """`certify` with its reasoning shown, for explaining a verdict.

    {'verdict', 'gap', 'threshold', 'trees_for_a'} -- the predicted gap in
    score points, the gap it had to clear, and the share of trees that put
    batch A in front.
    """
    if not train or a is None or b is None:
        return {
            "verdict": "same",
            "gap": 0.0,
            "threshold": MIN_GAP,
            "trees_for_a": 0.5,
        }
    judge = _judge_for(train)
    verdict, gap, agreement = judge.decide(list(a), list(b))
    return {
        "verdict": verdict,
        "gap": gap,
        "threshold": float(judge.threshold),
        "trees_for_a": agreement,
    }
