"""step3.py -- confidence and self-assessment for the winemaker's AI assistant.

Data convention (same as :mod:`step1` and :mod:`step2`)
-------------------------------------------------------
``rows`` is a list of lists.  ``rows[0]`` is the header (strings); every later
row holds numbers.  A column name ending in ``+`` is a goal to be maximized,
one ending in ``-`` is a goal to be minimized, and anything else is a lab
reading.  The panel's ``Quality+`` column is the score being predicted.

Two more questions are answered here:

* "When you say batch A will beat batch B, tell me whether that's a
   real difference or just tasting noise."                    -> certify()
* "How often are your score estimates right on batches you've
   never seen before?"                                        -> accuracy()

How "just noise" is decided
---------------------------
A prediction is never exact, so a small gap between two predicted scores means
nothing.  Rather than guess how small is too small, the size of our own error
is measured on the train rows themselves: the step 2 model is refitted on
cross-validation folds so that every train batch gets a prediction made
*without* it, and those honest out-of-fold predictions are paired up against
the scores the panel actually awarded.  Among those pairs we know the truth --
A really was better, B really was better, or the two really tied -- so we can
pick the one number we need, the smallest predicted gap worth calling a
winner, by choosing the gap that would have answered the most train pairs
correctly.  Gaps below it are reported as ``'same'``.

Cost
----
Both answers come from one fitted model plus one cross-validation pass over
the train rows, memoized per train set: the 500 ``certify`` calls and the
single ``accuracy`` call then cost one prediction per distinct batch (further
memoized), which is far inside the 90 second budget.
"""

from __future__ import annotations

import math

import step1
import step2

try:  # scikit-learn is expected, but a missing wheel must not be fatal
    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - the pure-Python path below still works
    _HAVE_SKLEARN = False

MAX_PAIRS = 20000  # train pairs used to calibrate "how big is a real gap"
MAX_THRESHOLDS = 256  # candidate gaps tried during that calibration
DEFAULT_GAP = 0.5  # half a panel point, used when nothing can be calibrated
_CACHE_SIZE = 4


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _round_half_up(value):
    """Round the way a person rounds: 5.5 -> 6, -5.5 -> -6."""
    if value != value:  # NaN
        return 0.0
    if value >= 0:
        return float(math.floor(value + 0.5))
    return -float(math.floor(-value + 0.5))


def _same_score(guessed, truth):
    """Did the guess round to the score the panel gave?"""
    rounded = _round_half_up(guessed)
    if abs(rounded - truth) < 1e-9:
        return True
    # tolerate a target that is itself not written as a whole number
    return abs(rounded - _round_half_up(truth)) < 1e-9


# --------------------------------------------------------------------------
# honest (out-of-fold) predictions for the train rows
# --------------------------------------------------------------------------
def _out_of_fold(oracle):
    """Predict every labelled train batch from a model that never saw it.

    Returns ``(predictions, truths)`` as numpy arrays, or ``None`` when no
    model could be fitted (in which case there is nothing to calibrate).
    """
    if not (_HAVE_SKLEARN and oracle is not None and oracle.model is not None):
        return None

    rows, truths = [], []
    for row in oracle.data:
        y = step2._cell(row, oracle.y_index)
        if y is None:
            continue
        rows.append(row)
        truths.append(y)

    if len(truths) < max(4, step2.MIN_TRAIN_FOR_MODEL):
        return None

    X = oracle.features(rows)
    y = np.asarray(truths, dtype=float)
    folds = max(2, min(5, len(y) // 2))

    predictions = np.empty(len(y), dtype=float)
    try:
        cv = KFold(n_splits=folds, shuffle=True, random_state=0)
        for train_idx, test_idx in cv.split(X):
            model = clone(oracle.model)
            model.fit(X[train_idx], y[train_idx])
            predictions[test_idx] = model.predict(X[test_idx])
    except Exception:
        return None

    # the same promise made by step2.guess: never leave the observed range
    np.clip(predictions, oracle.y_lo, oracle.y_hi, out=predictions)
    if not np.all(np.isfinite(predictions)):
        return None
    return predictions, y


# --------------------------------------------------------------------------
# "how big a gap is a real difference?"
# --------------------------------------------------------------------------
def _calibrate_gap(predictions, truths):
    """Smallest predicted gap worth calling a winner, learned from train pairs.

    Train batches are paired up at random.  For each pair we know both what we
    would have said (the gap between two out-of-fold predictions) and what was
    true (A better, B better, or a genuine tie).  The gap returned is the one
    that would have answered the most of those pairs correctly -- it is small
    when the model is sharp and the panel splits hairs, and large when our
    error swamps the differences we are being asked about.
    """
    n = len(truths)
    if n < 4:
        return DEFAULT_GAP

    rng = np.random.default_rng(0)
    if n * (n - 1) // 2 <= MAX_PAIRS:
        left, right = np.triu_indices(n, k=1)
    else:
        left = rng.integers(0, n, size=MAX_PAIRS)
        right = rng.integers(0, n, size=MAX_PAIRS)
        keep = left != right
        left, right = left[keep], right[keep]
    if len(left) == 0:
        return DEFAULT_GAP

    gap = predictions[left] - predictions[right]
    true_diff = truths[left] - truths[right]
    truth_sign = np.sign(np.where(np.abs(true_diff) < 1e-9, 0.0, true_diff))

    magnitude = np.abs(gap)
    gap_sign = np.sign(gap)

    # candidate thresholds: 0 (always name a winner) plus observed gap sizes
    candidates = np.unique(magnitude)
    if len(candidates) > MAX_THRESHOLDS:
        quantiles = np.linspace(0.0, 1.0, MAX_THRESHOLDS)
        candidates = np.unique(np.quantile(magnitude, quantiles))
    candidates = np.concatenate(([0.0], candidates, [magnitude.max() + 1.0]))

    best_gap, best_hits = DEFAULT_GAP, -1
    for threshold in candidates:
        said = np.where(magnitude >= threshold, gap_sign, 0.0)
        hits = int(np.count_nonzero(said == truth_sign))
        if hits > best_hits:
            best_gap, best_hits = float(threshold), hits
    return best_gap


# --------------------------------------------------------------------------
# everything learned about our own reliability, for one train set
# --------------------------------------------------------------------------
class _Judge:
    """Point predictions plus a sense of when a gap is meaningful."""

    def __init__(self, train):
        self.oracle = step2._oracle(train)
        self.gap = DEFAULT_GAP
        self.typical_error = None
        self._memo = {}

        folded = _out_of_fold(self.oracle)
        if folded is None:
            # No usable model: predictions are the cellar median, every gap is
            # zero, and every comparison is honestly "too close to call".
            if self.oracle is None or self.oracle.model is None:
                self.gap = DEFAULT_GAP
            return

        predictions, truths = folded
        self.typical_error = float(np.mean(np.abs(predictions - truths)))
        self.gap = _calibrate_gap(predictions, truths)

    # -- predicting ------------------------------------------------------
    def predict(self, row):
        """Predicted panel score for one batch, memoized by its lab sheet."""
        if self.oracle is None:
            return 0.0
        if row is None:
            return float(self.oracle.typical)
        try:
            key = tuple(row)
        except TypeError:
            return self.oracle.predict(row)
        if key not in self._memo:
            self._memo[key] = self.oracle.predict(row)
        return self._memo[key]

    def predict_many(self, rows):
        if self.oracle is None:
            return [0.0] * len(rows)
        return self.oracle.predict_many(rows)

    # -- comparing -------------------------------------------------------
    def compare(self, a, b):
        """``'A'``, ``'B'`` or ``'same'`` for two batches."""
        if self.oracle is None:
            return "same"
        if a is None and b is None:
            return "same"
        if a is None:
            return "B"
        if b is None:
            return "A"

        difference = self.predict(a) - self.predict(b)
        if abs(difference) < self.gap or abs(difference) < 1e-12:
            return "same"
        return "A" if difference > 0 else "B"


# --------------------------------------------------------------------------
# memoization: one calibration per train set, however many questions follow
# --------------------------------------------------------------------------
_cache = {}
_cache_order = []


def _judge(train):
    """The calibrated judge for this train set (built once, then reused)."""
    if not train or len(train) < 2:
        return None

    key = step2._fingerprint(train)
    if key in _cache:
        return _cache[key]

    judge = _Judge(train)
    _cache[key] = judge
    _cache_order.append(key)
    while len(_cache_order) > _CACHE_SIZE:
        _cache.pop(_cache_order.pop(0), None)
    return judge


# --------------------------------------------------------------------------
# "is that a real difference, or just tasting noise?"
# --------------------------------------------------------------------------
def certify(train, a, b):
    """Which of two batches will score better: ``'A'``, ``'B'`` or ``'same'``.

    Both batches are scored from their lab readings using only `train`, and the
    gap between the two scores is compared against the gap that train rows
    showed to be meaningful.  A smaller gap is reported as ``'same'``: we would
    be reading our own error, not a difference in the wine.
    """
    judge = _judge(train)
    if judge is None:
        return "same"
    return judge.compare(a, b)


# --------------------------------------------------------------------------
# "how often are your estimates right on batches you've never seen?"
# --------------------------------------------------------------------------
def accuracy(train, test):
    """Share of `test` batches whose guessed score rounds to their true score.

    The model is built from `train` alone; each test batch is scored from its
    lab readings, rounded to a whole panel point, and checked against the score
    the panel actually gave.  Test batches with no recorded score cannot be
    checked and are left out of the share.
    """
    judge = _judge(train)
    if judge is None or not test or len(test) < 2:
        return 0.0

    oracle = judge.oracle
    if oracle is None:
        return 0.0

    # the test file carries its own header; find its score column there
    header = [str(h).strip() for h in test[0]]
    y_index = step2._target_index(header)
    if y_index is None:
        y_index = oracle.y_index

    rows, truths = [], []
    for row in test[1:]:
        if row is None:
            continue
        y = step2._cell(row, y_index)
        if y is None:
            continue
        rows.append(row)
        truths.append(y)

    if not rows:
        return 0.0

    guesses = judge.predict_many(rows)
    hits = sum(1 for guess, truth in zip(guesses, truths)
               if _same_score(guess, truth))
    return hits / len(rows)
