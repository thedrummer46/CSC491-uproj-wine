"""step2.py -- predicting the panel's score, and grouping batches into kinds.

Answers two requests from the winemaker:

  "Before the panel sits, estimate what score this new batch will get
   from its lab sheet alone."
        ->  guess(train, row) -> float

  "Group our batches into a few kinds, and tell me what score to
   expect from each kind."
        ->  kinds(train) -> [(group_rows, expected_quality), ...]

Data convention is step1's: `rows` is a list of lists, row 0 is the header
(strings), every later row holds numbers; '+' suffix means a goal to
maximize, '-' a goal to minimize, anything else is a lab reading.  Parsing,
column classification and the median are imported from step1 rather than
rewritten.

Two judgement calls worth stating plainly:

* The prediction target is the maximized goal whose name starts with
  "Quality" (falling back to the last '+' column).  Everything else is used
  as an input -- including 'Alcohol-'.  Alcohol is a goal to minimize when
  the cellar is choosing what to make, but it is also a number that is on
  the lab sheet before the panel ever sits, so it is legitimate input for
  this question.  Only the panel's own score is unknown at prediction time.

* `guess` never looks at `row`'s target cell, even when the caller hands in
  a full labelled row.  The feature vector is built by column position with
  the target position skipped.

Cost: `guess` is called once per test batch with the same `train` table, so
the model is fitted once and cached under a fingerprint of that table; later
calls are a single `predict`.  A train/test split of a few hundred rows runs
in a couple of seconds.
"""

from __future__ import annotations

import threading

import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import step1

RANDOM_STATE = 0
N_TREES = 400
MIN_KINDS = 2
MAX_KINDS = 6

_CACHE = {}
_CACHE_LIMIT = 4
_LOCK = threading.Lock()


# ----------------------------------------------------------------- columns

def _target_and_features(header):
    """(target column index, feature column indexes) for the header row.

    The target is the maximized goal the panel awards; every other column is
    assumed to be measurable before tasting and therefore usable as input.
    """
    kinds_ = step1._column_kinds(header)
    maximized = [i for i, kind in enumerate(kinds_) if kind == step1._MAXIMIZE]

    target = None
    for i in maximized:
        if str(header[i]).strip().lower().startswith("quality"):
            target = i
            break
    if target is None and maximized:
        target = maximized[-1]

    features = [i for i in range(len(header)) if i != target]
    return target, features


def _cell(row, index, fallback):
    """One numeric cell, with a fallback for missing / non-numeric / absent."""
    if index < len(row):
        value = step1._number(row[index])
        if value is not None:
            return value
    return fallback


def _vector(row, features, medians):
    """Feature vector for one batch; blanks imputed with the train median."""
    return [_cell(row, index, medians[index]) for index in features]


# -------------------------------------------------------------- model cache

class _Model:
    """A fitted quality model plus everything needed to score a fresh row."""

    def __init__(self, header, data):
        self.header = header
        self.data = data
        self.target, self.features = _target_and_features(header)

        self.medians = {
            index: step1._median(step1._column(data, index))
            for index in range(len(header))
        }

        self.forest = None
        self.fallback = 0.0
        self.labelled = []          # (feature vector, quality) for train rows

        if self.target is None:
            return

        targets = step1._column(data, self.target)
        self.fallback = step1._median(targets)

        for row in data:
            quality = (step1._number(row[self.target])
                       if self.target < len(row) else None)
            if quality is None:
                continue            # unscored batch teaches nothing here
            self.labelled.append((_vector(row, self.features, self.medians),
                                  quality, row))

        if len(self.labelled) < 2 or not self.features:
            return

        X = np.asarray([item[0] for item in self.labelled], dtype=float)
        y = np.asarray([item[1] for item in self.labelled], dtype=float)

        # A random forest suits this table: a few hundred rows, a dozen
        # correlated chemical readings, a strongly non-linear and plateauing
        # relationship with the panel score, and no scaling to get wrong.
        self.forest = RandomForestRegressor(
            n_estimators=N_TREES,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=1,               # single row per predict: threads only cost
        )
        self.forest.fit(X, y)

    # ------------------------------------------------------------- scoring

    def predict_one(self, row):
        if self.target is None:
            return 0.0
        if self.forest is None:
            return float(self.fallback)
        vector = np.asarray([_vector(row, self.features, self.medians)],
                            dtype=float)
        return float(self.forest.predict(vector)[0])

    def predict_many(self, rows_of_numbers):
        if self.forest is None:
            return np.full(len(rows_of_numbers), float(self.fallback))
        X = np.asarray([_vector(row, self.features, self.medians)
                        for row in rows_of_numbers], dtype=float)
        if len(X) == 0:
            return np.asarray([])
        return self.forest.predict(X)


def _fingerprint(rows):
    """A hashable summary of a train table, so the fit is done only once."""
    header, data = step1._split(rows)
    return (
        tuple(str(name) for name in header),
        tuple(tuple(step1._number(value) for value in row) for row in data),
    )


def _model_for(train):
    key = _fingerprint(train)
    with _LOCK:
        model = _CACHE.get(key)
        if model is not None:
            return model

    header, data = step1._split(train)
    model = _Model(header, data)

    with _LOCK:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.clear()          # only ever a handful of splits in flight
        _CACHE[key] = model
    return model


# ------------------------------------------------------------------- public

def guess(train, row):
    """Predicted Quality+ for one batch, learned from `train` alone.

    `row` may be a bare lab sheet or a full labelled row; its target cell is
    ignored either way.  Returns the train median quality when the table is
    too thin to fit anything, and 0.0 when there is nothing to predict.
    """
    if not train or row is None:
        return 0.0
    return _model_for(train).predict_one(list(row))


def kinds(train):
    """Group the batches into a few kinds and say what each kind scores.

    Returns a list of (group_rows, expected_quality) pairs, best-scoring kind
    first.  `group_rows` is a list of the data rows in that kind (copies, no
    header); `expected_quality` is the median observed Quality+ of its
    members -- median for the same reason step1.summary uses it, since the
    quality scale is coarse and a few outlier batches should not move the
    number the winemaker plans around.

    Kinds are found by k-means on the standardized lab-sheet columns, so a
    kind is a recognizable style of wine rather than a quality bracket.  k is
    chosen between 2 and 6 by silhouette score, which keeps the answer to the
    "a few" the winemaker asked for.
    """
    header, data = step1._split(train)
    if not data:
        return []

    model = _model_for(train)

    if not model.features or len(data) < MIN_KINDS * 2:
        return [(
            [list(row) for row in data],
            _expected(data, model),
        )]

    X = np.asarray([_vector(row, model.features, model.medians)
                    for row in data], dtype=float)
    X = StandardScaler().fit_transform(X)

    labels = _best_labels(X)
    if labels is None:
        return [([list(row) for row in data], _expected(data, model))]

    groups = []
    for label in sorted(set(labels)):
        members = [data[i] for i in range(len(data)) if labels[i] == label]
        if not members:
            continue
        groups.append(([list(row) for row in members], _expected(members, model)))

    groups.sort(key=lambda pair: pair[1], reverse=True)
    return groups


# ------------------------------------------------------------------ helpers

def _best_labels(X):
    """k-means labels for the k in [2, 6] with the best silhouette score."""
    upper = min(MAX_KINDS, len(X) - 1)
    best = None
    best_score = -2.0
    for k in range(MIN_KINDS, upper + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
        except ValueError:
            continue
        if score > best_score:
            best_score = score
            best = labels
    return best


def _expected(members, model):
    """The score to expect from a kind: median observed quality of its rows.

    Falls back to the model's own prediction for the kind when none of its
    batches has been scored yet, and to the train median after that.
    """
    if model.target is None:
        return 0.0

    observed = []
    for row in members:
        if model.target < len(row):
            value = step1._number(row[model.target])
            if value is not None:
                observed.append(value)
    if observed:
        return float(step1._median(observed))

    predicted = model.predict_many(members)
    if len(predicted):
        return float(np.median(predicted))
    return float(model.fallback)
