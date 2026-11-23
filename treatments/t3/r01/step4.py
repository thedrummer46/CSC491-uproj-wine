"""step4.py -- the batch-by-batch learner for the winemaker's AI assistant.

Data convention (same as :mod:`step1`, :mod:`step2` and :mod:`step3`)
---------------------------------------------------------------------
A column name ending in ``+`` is a goal to be maximized, one ending in ``-``
is a goal to be minimized, and anything else is a lab reading.  The panel's
``Quality+`` column is the score being learned.  Here, though, the cellar does
not hand us a table: it hands us one batch at a time, as it comes off the line.

Two more questions are answered here:

* "As each new batch comes off the line, tell me right away whether
   it's worth sending to the panel."                          -> worth()
* "Keep sharpening your sense of what makes a good batch as this
   season's scores come in, without starting over."           -> learn()

How the learning stays incremental
----------------------------------
Nothing is ever refitted from scratch.  Each scored batch is folded into a
handful of running quantities, all of which cost the same whether it is the
first batch of the season or the thousandth:

* the cellar's running average score, and the running mean/spread of every
  reading (Welford's method, used to fill a blank cell on the lab sheet);
* the normal equations of a least-squares fit -- a 12x12 sum of outer products
  -- which are *exactly* the batch fit over everything seen so far, updated
  one rank-one term at a time and solved only when someone asks for a
  prediction;
* a growing table of past lab sheets, for a nearest-neighbour estimate that
  reads the score off the most similar batches;
* a scikit-learn ``SGDRegressor``, nudged one batch at a time by
  ``partial_fit``.

Which of those four voices to believe is also learned on the fly.  Before a
new batch is folded in, each one is asked for its prediction and graded
against the score that actually arrived -- a fair test, because that batch has
not been seen yet.  Each voice carries a decayed average of its own recent
error, and the final answer is their error-weighted blend, so a voice that is
doing well this season is listened to and one that is not quietly fades out.
That blend is then held inside the range of scores the panel has actually
awarded: we never promise a score nobody has ever given.

Cost
----
One ``learn`` is a few hundred arithmetic operations plus one ``partial_fit``;
one ``predict`` is a dot product, one distance sweep over the stored batches,
and (at most once per burst of questions) one 12x12 solve.  The stated
workload -- 1279 learns interleaved with about 4000 predictions -- runs in a
couple of seconds, far inside the 90 second budget.
"""

from __future__ import annotations

import math

import step1
import step2

try:  # numpy carries the linear algebra; without it the mean still answers
    import numpy as np

    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False

try:  # scikit-learn's incremental regressor joins the panel of voices
    from sklearn.linear_model import SGDRegressor

    _HAVE_SKLEARN = _HAVE_NUMPY
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False

WORTH_CUTOFF = 6.0  # "worth sending to the panel" = a predicted score of 6+
ERROR_DECAY = 0.98  # ~50 batch memory for a voice's track record
ETA = 6.0  # how sharply the better-scoring voices are preferred
MIN_SCORED = 3  # batches graded before a voice's record is trusted
_EPS = 1e-12


# --------------------------------------------------------------------------
# running statistics (constant cost per batch)
# --------------------------------------------------------------------------
class _Running:
    """Count, mean and spread of one column, updated one value at a time."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def sd(self):
        if self.n < 2:
            return 0.0
        return math.sqrt(max(self.m2 / (self.n - 1), 0.0))


# --------------------------------------------------------------------------
# the voices
# --------------------------------------------------------------------------
class _Expert:
    """One way of guessing a score, plus its own recent track record."""

    name = "expert"

    def __init__(self):
        self.err = 0.0
        self.seen = 0

    def grade(self, error):
        """Fold in how far off this voice was on a batch it had not seen."""
        if self.seen == 0:
            self.err = error
        else:
            self.err = ERROR_DECAY * self.err + (1.0 - ERROR_DECAY) * error
        self.seen += 1

    # subclasses override
    def predict(self, x, model):
        return None

    def learn(self, x, y, model):
        pass


class _Typical(_Expert):
    """The cellar's running average score: the answer when nothing else fits."""

    name = "typical"

    def predict(self, x, model):
        if model.y_stat.n == 0:
            return None
        return float(model.y_stat.mean)


class _Linear(_Expert):
    """Least squares on the lab readings, kept as running normal equations.

    ``A`` accumulates the sum of outer products of each lab sheet (with a
    trailing 1 for the intercept) and ``b`` the sum of sheet-times-score, so
    after any number of batches ``A w = b`` is exactly the fit over everything
    seen so far.  The solve is deferred until a prediction is wanted and then
    cached, which is why a burst of 320 questions costs one solve.
    """

    name = "linear"

    def __init__(self, width):
        _Expert.__init__(self)
        self.ok = _HAVE_NUMPY and width > 0
        self.width = width + 1
        self.rows = 0
        self.w = None
        self.dirty = True
        if self.ok:
            self.A = np.zeros((self.width, self.width), dtype=float)
            self.b = np.zeros(self.width, dtype=float)

    def learn(self, x, y, model):
        if not self.ok:
            return
        v = np.empty(self.width, dtype=float)
        v[:-1] = x
        v[-1] = 1.0
        self.A += np.outer(v, v)
        self.b += v * y
        self.rows += 1
        self.dirty = True

    def _solve(self):
        self.dirty = False
        self.w = None
        # Jacobi preconditioning: readings differ in scale by orders of
        # magnitude (sulphates vs total SO2), so normalize before solving and
        # undo it afterwards.  The tiny ridge only keeps the solve well posed.
        diag = np.sqrt(np.maximum(np.diag(self.A), _EPS))
        scale = 1.0 / diag
        A = self.A * np.outer(scale, scale)
        b = self.b * scale
        A = A + 1e-4 * np.eye(self.width)
        try:
            w = np.linalg.solve(A, b)
        except Exception:
            try:
                w = np.linalg.lstsq(A, b, rcond=None)[0]
            except Exception:
                return
        w = w * scale
        if np.all(np.isfinite(w)):
            self.w = w

    def predict(self, x, model):
        if not self.ok or self.rows < self.width + 2:
            return None
        if self.dirty:
            self._solve()
        if self.w is None:
            return None
        return float(np.dot(self.w[:-1], x) + self.w[-1])


class _Neighbours(_Expert):
    """Read the score off the most similar past batches.

    Nothing is fitted, so there is nothing to refit: a new batch is appended
    to the table (doubling it when full) and that is the whole update.  The
    comparison is made on standardized readings, using the running spreads, so
    every reading counts for about as much.
    """

    name = "neighbours"

    MIN_ROWS = 8
    MIN_K = 3
    MAX_K = 12

    def __init__(self, width):
        _Expert.__init__(self)
        self.ok = _HAVE_NUMPY and width > 0
        self.width = width
        self.rows = 0
        if self.ok:
            self.X = np.zeros((64, width), dtype=float)
            self.y = np.zeros(64, dtype=float)

    def learn(self, x, y, model):
        if not self.ok:
            return
        if self.rows >= len(self.y):
            self.X = np.resize(self.X, (2 * len(self.y), self.width))
            self.y = np.resize(self.y, 2 * len(self.y))
        self.X[self.rows, :] = x
        self.y[self.rows] = y
        self.rows += 1

    def predict(self, x, model):
        if not self.ok or self.rows < self.MIN_ROWS:
            return None
        scale = model.inv_sd()
        if scale is None:
            return None
        diff = (self.X[: self.rows] - np.asarray(x, dtype=float)) * scale
        d2 = np.einsum("ij,ij->i", diff, diff)
        k = min(self.rows, max(self.MIN_K, min(self.MAX_K, int(round(math.sqrt(self.rows))))))
        if k >= self.rows:
            idx = np.arange(self.rows)
        else:
            idx = np.argpartition(d2, k - 1)[:k]
        weights = 1.0 / (np.sqrt(np.maximum(d2[idx], 0.0)) + 1e-6)
        total = float(weights.sum())
        if total <= 0.0:
            return float(self.y[idx].mean())
        return float(np.dot(weights, self.y[idx]) / total)


class _Sgd(_Expert):
    """scikit-learn's ``SGDRegressor``, nudged one batch at a time.

    Its weights are learned against a standardization frozen once enough
    batches have been seen to measure one; freezing matters, because weights
    learned under one scaling would be meaningless under another.  Scores are
    learned as departures from the frozen average, so the fit has no intercept
    to chase.
    """

    name = "sgd"

    START = 25
    MIN_ROWS = 10

    def __init__(self, width):
        _Expert.__init__(self)
        self.ok = _HAVE_SKLEARN and width > 0
        self.model = None
        self.center = None
        self.scale = None
        self.offset = 0.0
        self.rows = 0

    @staticmethod
    def _make():
        kwargs = dict(
            penalty="l2",
            alpha=1e-4,
            learning_rate="invscaling",
            eta0=0.01,
            power_t=0.25,
            random_state=0,
            max_iter=1,
            tol=None,
            warm_start=True,
        )
        try:
            return SGDRegressor(loss="squared_error", **kwargs)
        except Exception:  # pragma: no cover - older scikit-learn
            return SGDRegressor(loss="squared_loss", **kwargs)

    def _z(self, x):
        return ((np.asarray(x, dtype=float) - self.center) * self.scale).reshape(1, -1)

    def learn(self, x, y, model):
        if not self.ok:
            return
        if self.model is None:
            if model.y_stat.n < self.START:
                return
            center = model.means()
            scale = model.inv_sd()
            if center is None or scale is None:
                return
            self.center, self.scale = center, scale
            self.offset = float(model.y_stat.mean)
            try:
                self.model = self._make()
            except Exception:
                self.ok = False
                return
        try:
            self.model.partial_fit(self._z(x), np.asarray([y - self.offset], dtype=float))
        except Exception:
            self.ok = False
            self.model = None
            return
        self.rows += 1

    def predict(self, x, model):
        if not self.ok or self.model is None or self.rows < self.MIN_ROWS:
            return None
        try:
            value = float(self.model.predict(self._z(x))[0]) + self.offset
        except Exception:
            return None
        return value if math.isfinite(value) else None


# --------------------------------------------------------------------------
# the model itself
# --------------------------------------------------------------------------
class _Model:
    """Everything the assistant knows so far, and nothing it has to refit."""

    def __init__(self, names):
        header = [str(h).strip() for h in (names or [])]
        self.header = header
        self.y_index = step2._target_index(header)
        self.x_index = step2._feature_indices(header, self.y_index)

        width = len(self.x_index)
        self.stats = [_Running() for _ in range(width)]
        self.y_stat = _Running()
        self.y_lo = None
        self.y_hi = None

        self._version = 0  # bumped whenever a reading's statistics change
        self._cached_at = -1
        self._means = None
        self._inv_sd = None

        self.experts = [_Typical(), _Linear(width), _Neighbours(width), _Sgd(width)]

    # -- the lab sheet as numbers ---------------------------------------
    def vector(self, row):
        """One lab sheet, with any blank cell filled by the typical reading."""
        out = []
        for position, index in enumerate(self.x_index):
            value = step2._cell(row, index)
            out.append(self.stats[position].mean if value is None else value)
        return out

    def _refresh(self):
        if not _HAVE_NUMPY or not self.x_index:
            self._means = self._inv_sd = None
            self._cached_at = self._version
            return
        means, inv = [], []
        for stat in self.stats:
            means.append(stat.mean)
            sd = stat.sd
            inv.append(1.0 / sd if sd > 1e-9 else 1.0)
        self._means = np.asarray(means, dtype=float)
        self._inv_sd = np.asarray(inv, dtype=float)
        self._cached_at = self._version

    def means(self):
        if self._cached_at != self._version:
            self._refresh()
        return self._means

    def inv_sd(self):
        if self._cached_at != self._version:
            self._refresh()
        return self._inv_sd

    # -- learning --------------------------------------------------------
    def observe(self, row):
        """Fold one scored batch in: grade the voices, then teach them."""
        if row is None:
            return self

        y = step2._cell(row, self.y_index)
        x = self.vector(row)

        if y is None:
            # No panel score (or a stray header row): there is nothing to
            # learn about quality, but the readings still shape what "normal"
            # looks like for filling blanks and comparing batches.
            self._note_readings(row)
            return self

        # graded before being taught, so no voice is tested on its own batch
        for expert in self.experts:
            guess = expert.predict(x, self)
            if guess is not None and math.isfinite(guess):
                expert.grade(abs(guess - y))

        self.y_stat.add(y)
        self.y_lo = y if self.y_lo is None else min(self.y_lo, y)
        self.y_hi = y if self.y_hi is None else max(self.y_hi, y)

        for expert in self.experts:
            expert.learn(x, y, self)

        self._note_readings(row)
        return self

    def _note_readings(self, row):
        for position, index in enumerate(self.x_index):
            value = step2._cell(row, index)
            if value is not None:
                self.stats[position].add(value)
        self._version += 1

    # -- predicting ------------------------------------------------------
    def _clamp(self, value):
        if value != value:  # NaN
            return float(self.y_stat.mean)
        if self.y_lo is None:
            return float(value)
        return float(max(self.y_lo, min(self.y_hi, value)))

    def _blend(self, x):
        """Error-weighted average of whichever voices can speak."""
        ready = []
        for expert in self.experts:
            guess = expert.predict(x, self)
            if guess is not None and math.isfinite(guess):
                ready.append((expert, guess))
        if not ready:
            return float(self.y_stat.mean)

        scored = [(e, g) for e, g in ready if e.seen >= MIN_SCORED]
        if not scored:
            return sum(g for _e, g in ready) / len(ready)

        best = min(e.err for e, _g in scored)
        span = max(best, 1e-3)
        total, acc = 0.0, 0.0
        for expert, guess in scored:
            weight = math.exp(-ETA * (expert.err - best) / span)
            total += weight
            acc += weight * guess
        if total <= 0.0:
            return sum(g for _e, g in scored) / len(scored)
        return acc / total

    def predict(self, row):
        if self.y_stat.n == 0:
            return 0.0  # nothing has been tasted yet; no score to offer
        if row is None:
            return float(self.y_stat.mean)
        return self._clamp(self._blend(self.vector(row)))


# --------------------------------------------------------------------------
# "start with an empty model that knows only the column names"
# --------------------------------------------------------------------------
def start(names):
    """An assistant that knows the shape of a lab sheet and nothing else."""
    return _Model(names)


# --------------------------------------------------------------------------
# "keep sharpening as this season's scores come in"
# --------------------------------------------------------------------------
def learn(model, row):
    """Fold one more scored batch into `model` and return it.

    The update is incremental: running statistics, one rank-one term added to
    the least-squares normal equations, one row appended to the neighbour
    table and one ``partial_fit``.  Nothing is refitted from scratch, so the
    thousandth batch costs what the first one did.
    """
    if model is None:
        return model
    try:
        return model.observe(row)
    except Exception:
        # a malformed batch must never cost the cellar everything learned
        return model


# --------------------------------------------------------------------------
# "estimate the score from the lab sheet"
# --------------------------------------------------------------------------
def predict(model, row):
    """Predicted ``Quality+`` for one batch, from what `model` has seen.

    Before any scored batch has arrived there is nothing to go on and 0.0 is
    returned; after that the answer stays within the range of scores the panel
    has actually awarded.
    """
    if model is None:
        return 0.0
    try:
        return float(model.predict(row))
    except Exception:
        return 0.0


# --------------------------------------------------------------------------
# "is this one worth sending to the panel?"
# --------------------------------------------------------------------------
def worth(model, row):
    """True when this batch's predicted score reaches ``WORTH_CUTOFF`` (6).

    One caveat worth knowing at the dinner table: a model fitted to panel
    scores is deliberately cautious at the extremes -- it pulls its guesses
    toward the middle of the cellar -- so a genuinely fine batch can be
    predicted at 5.8 and be held back.  If the cellar would rather send a few
    too many than miss a good one, lower ``WORTH_CUTOFF``; 5.5 sends every
    batch whose prediction rounds to a 6.
    """
    return bool(predict(model, row) >= WORTH_CUTOFF - 1e-9)
