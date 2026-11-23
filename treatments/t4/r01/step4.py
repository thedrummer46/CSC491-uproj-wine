"""step4.py -- the batch that just came off the line, judged on the spot.

Answers the winemaker's last two requests:

  "As each new batch comes off the line, tell me right away whether it's
   worth sending to the panel."
        ->  predict(model, row) -> float      (estimated Quality+)
            worth(model, row)   -> bool       (>= 6 points, so send it)

  "Keep sharpening your sense of what makes a good batch as this season's
   scores come in, without starting over."
        ->  start(names)        -> model      (knows only the header)
            learn(model, row)   -> model      (one more scored batch)

Data convention is step1's, and the column bookkeeping is step2's: which
column the panel awards and which columns are inputs both come from
`step2._target_and_features`, so 'Quality+' is the target here exactly as it
is there and 'Alcohol-' is an input here exactly as it is there (it is on the
lab sheet before the panel ever sits).  `step1._number` does the cell
coercion.  Nothing from the earlier steps is re-derived.

Why not step2's forest
----------------------
step2 fits a random forest to a finished table.  A forest cannot absorb one
new row -- it can only be grown again -- and the winemaker explicitly asked
not to start over.  So this step keeps a model whose whole state is a set of
running sums:

* a ridge regression held as sufficient statistics (n, sum x, sum x x', sum
  x y, sum y).  One new batch is a rank-one update of a 11x11 matrix; the
  coefficients are then the exact least-squares answer for everything seen
  so far, re-solved lazily from those sums the next time anyone asks for a
  prediction.  No past row is revisited, and there is no learning rate or
  pass count to get wrong -- an online gradient method would give a worse
  answer for more fuss.

* an append-only memory of the standardized batches themselves, read as a
  distance-weighted k nearest neighbours.  The ridge line carries the broad
  trend; the neighbours carry the curvature the line cannot (quality flattens
  off at both ends of the alcohol range).  Blending the two beats either one
  alone at the thing being measured, which is whether the estimate rounds to
  the score the panel actually gave.

Cost
----
learn is O(d^2) with d = 11, so ~1300 calls are a few milliseconds in total.
predict is one dot product plus one scan of the stored batches: ~14k floats
per call, so ~4000 calls land well under a second.  The whole split is
comfortably inside the 90s budget, and the expensive part of the budget
(solving for the coefficients) happens once per batch of predictions rather
than once per prediction.
"""

from __future__ import annotations

import numpy as np

import step1
import step2

# "Worth sending to the panel" as the winemaker defined it.
WORTH_THRESHOLD = 6.0

# Ridge penalty, applied in standardized units.  Small: it is here to keep
# the normal equations well behaved while two correlated readings (density
# and fixed acidity, say) are still nearly collinear in a young season.
RIDGE = 1e-3

# Below this many scored batches, a fitted line is noise; answer with the
# running average score instead.
MIN_FOR_LINEAR = 12

# Below this many, the neighbourhood of a new batch is too sparse to say
# anything local about it.
MIN_FOR_NEIGHBOURS = 40

NEIGHBOURS = 10
KNN_WEIGHT = 0.5

# Softening term in the inverse-distance weights, so one near-duplicate
# batch cannot drown out the other nine neighbours.
DISTANCE_FLOOR = 0.1

INITIAL_CAPACITY = 256
_TINY = 1e-12


# ------------------------------------------------------------------- model

class Model:
    """Everything the assistant knows, in a form one batch can update.

    State is deliberately all running sums plus an append-only row store:
    `learn` touches no past row, and the coefficients are recomputed from
    the sums only when a prediction is next asked for.
    """

    def __init__(self, header):
        self.header = [str(name) for name in header]
        self.target, self.features = step2._target_and_features(self.header)
        self.d = len(self.features)

        # Sufficient statistics for the ridge fit.
        self.n = 0
        self.sum_x = np.zeros(self.d, dtype=float)
        self.sum_xx = np.zeros((self.d, self.d), dtype=float)
        self.sum_xy = np.zeros(self.d, dtype=float)
        self.sum_y = 0.0

        # Observed range of panel scores, used to keep estimates sane.
        self.y_lo = None
        self.y_hi = None

        # Append-only memory of scored batches, for the neighbour term.
        self._X = np.zeros((INITIAL_CAPACITY, self.d), dtype=float)
        self._Y = np.zeros(INITIAL_CAPACITY, dtype=float)

        # Lazily derived from the sums above; invalidated by every learn.
        self._fresh = False
        self._mean = np.zeros(self.d, dtype=float)
        self._scale = np.zeros(self.d, dtype=float)
        self._w = None
        self._b = 0.0
        self._ybar = 0.0

    # ------------------------------------------------------------- reading

    def _mean_vector(self):
        if self.n == 0:
            return np.zeros(self.d, dtype=float)
        return self.sum_x / self.n

    def _vector(self, row):
        """Feature vector for one batch, by column position.

        The target column is skipped by position, so a fully labelled row
        can be handed to `predict` or `worth` without the panel's own score
        leaking into the answer.  Blanks are imputed with the running mean
        of everything seen so far -- the only average available online.
        """
        mean = self._mean_vector()
        out = np.empty(self.d, dtype=float)
        for j, index in enumerate(self.features):
            value = step1._number(row[index]) if index < len(row) else None
            out[j] = mean[j] if value is None else value
        return out

    # ------------------------------------------------------------ learning

    def _store(self, x, y):
        if self.n >= len(self._Y):
            grown = max(INITIAL_CAPACITY, 2 * len(self._Y))
            X = np.zeros((grown, self.d), dtype=float)
            Y = np.zeros(grown, dtype=float)
            X[: self.n] = self._X[: self.n]
            Y[: self.n] = self._Y[: self.n]
            self._X, self._Y = X, Y
        self._X[self.n] = x
        self._Y[self.n] = y

    def learn_row(self, row):
        """Absorb one scored batch.  Constant work; nothing is refitted."""
        if self.target is None:
            return self

        y = (step1._number(row[self.target])
             if self.target < len(row) else None)
        if y is None:
            # An unscored batch teaches a supervised model nothing, and
            # folding it in would corrupt the sums it is built from.
            return self

        x = self._vector(row)
        self._store(x, y)

        self.n += 1
        self.sum_x += x
        self.sum_xx += np.outer(x, x)
        self.sum_xy += x * y
        self.sum_y += y

        self.y_lo = y if self.y_lo is None else min(self.y_lo, y)
        self.y_hi = y if self.y_hi is None else max(self.y_hi, y)

        self._fresh = False
        return self

    # ------------------------------------------------------------ fitting

    def _refresh(self):
        """Solve for the coefficients from the running sums, if stale.

        Called from `predict`, so the solve happens once per burst of
        predictions rather than once per learned batch.
        """
        if self._fresh:
            return
        self._fresh = True

        n = self.n
        if n == 0:
            self._mean = np.zeros(self.d, dtype=float)
            self._scale = np.zeros(self.d, dtype=float)
            self._w = None
            self._b = 0.0
            self._ybar = 0.0
            return

        mean = self.sum_x / n
        ybar = self.sum_y / n
        self._mean = mean
        self._ybar = ybar
        self._w = None
        self._b = ybar

        variance = np.maximum(np.diag(self.sum_xx) / n - mean ** 2, 0.0)
        self._scale = np.where(
            variance > _TINY, 1.0 / np.sqrt(np.maximum(variance, _TINY)), 0.0
        )

        if not self.d or n < MIN_FOR_LINEAR:
            return

        # Centred covariance of the readings against the panel score.  The
        # system is solved in standardized units: chlorides and total SO2
        # differ by three orders of magnitude, and a single ridge penalty
        # has to mean the same thing to both.
        cov = self.sum_xx / n - np.outer(mean, mean)
        cross = self.sum_xy / n - mean * ybar
        scale = self._scale

        A = cov * scale[:, None] * scale[None, :] + RIDGE * np.eye(self.d)
        rhs = cross * scale
        try:
            u = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            u = np.linalg.lstsq(A, rhs, rcond=None)[0]

        if np.all(np.isfinite(u)):
            w = u * scale
            self._w = w
            self._b = ybar - float(w @ mean)

    # ------------------------------------------------------------ scoring

    def _neighbour_value(self, x):
        """Distance-weighted score of the nearest scored batches, or None."""
        if not self.d or self.n == 0:
            return None
        diff = (self._X[: self.n] - x) * self._scale
        d2 = np.einsum("ij,ij->i", diff, diff)

        k = min(NEIGHBOURS, self.n)
        if k < self.n:
            picked = np.argpartition(d2, k - 1)[:k]
        else:
            picked = np.arange(self.n)

        weights = 1.0 / (d2[picked] + DISTANCE_FLOOR)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return None
        return float(np.dot(weights, self._Y[picked]) / total)

    def predict_one(self, row):
        if self.target is None:
            return 0.0
        self._refresh()
        if self.n == 0:
            # Nothing has been seen yet; there is no honest estimate.
            return 0.0

        x = self._vector(row)

        if self._w is None:
            value = self._ybar
        else:
            value = self._b + float(self._w @ x)
            if self.n >= MIN_FOR_NEIGHBOURS:
                local = self._neighbour_value(x)
                if local is not None:
                    value = (1.0 - KNN_WEIGHT) * value + KNN_WEIGHT * local

        if not np.isfinite(value):
            value = self._ybar
        # The panel has never awarded a score outside what we have seen, so
        # neither should we.
        return float(min(max(value, self.y_lo), self.y_hi))


# ------------------------------------------------------------------ public

def start(names):
    """An empty model that knows only the header row.

    Accepts the header itself, or a rows-style table whose first row is the
    header, so it can be called either way.
    """
    if names is None:
        header = []
    elif len(names) and isinstance(names[0], (list, tuple)):
        header = list(names[0])
    else:
        header = list(names)
    return Model(header)


def learn(model, row):
    """The model after seeing one more scored batch.

    Updates in place and returns the same model, so a caller can either
    rebind the result or ignore it.  The update is a rank-one addition to
    the stored sums plus an append to the row memory: no past batch is read
    again and nothing is refitted.
    """
    if model is None:
        return model
    if row is None:
        return model
    return model.learn_row(list(row))


def predict(model, row):
    """Predicted Quality+ for one batch, from what the model has seen.

    `row` may be a bare lab sheet or a fully labelled row; its Quality+ cell
    is never read.  Returns the running average score before there is enough
    data to fit a line, and 0.0 before there is any data at all.
    """
    if model is None or row is None:
        return 0.0
    return model.predict_one(list(row))


def worth(model, row):
    """True if this batch looks worth sending to the panel.

    That is, if its predicted Quality+ is 6 or more.  The tolerance is there
    so a batch that lands on 6 to within floating-point dust is sent rather
    than dropped.
    """
    return bool(predict(model, row) >= WORTH_THRESHOLD - 1e-9)


# ------------------------------------------------------------------ extras

def seen(model):
    """How many scored batches the model has learned from so far.

    Handy when the winemaker asks how much of this season is behind an
    answer -- early in the season `predict` is leaning on the running
    average, later on the fit and the neighbours.
    """
    return 0 if model is None else int(model.n)
