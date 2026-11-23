"""step1.py -- memory and summary primitives for the winemaker's assistant.

Data convention
---------------
`rows` is a list of lists.  Row 0 is the header (strings); every later row
holds numbers.  A column whose name ends in '+' is a goal to maximize, one
ending in '-' is a goal to minimize, and everything else is a lab reading
(an independent observation, not something to optimize).

Public API
----------
remember(rows) -> the handful of truly exceptional past batches
                  (never more than 5% of the data rows)
summary(rows)  -> {column name: typical value} for a normal batch
"""

from __future__ import annotations

import math

MAX_REMEMBERED_FRACTION = 0.05

_MAXIMIZE = "max"
_MINIMIZE = "min"
_READING = "x"


# ----------------------------------------------------------------- parsing

def _column_kinds(header):
    """Classify each header cell as a goal to maximize/minimize, or a reading."""
    kinds = []
    for name in header:
        text = str(name).strip()
        if text.endswith("+"):
            kinds.append(_MAXIMIZE)
        elif text.endswith("-"):
            kinds.append(_MINIMIZE)
        else:
            kinds.append(_READING)
    return kinds


def _split(rows):
    """Return (header, data_rows).  Tolerates an empty or header-only table."""
    if not rows:
        return [], []
    return list(rows[0]), [r for r in rows[1:] if r is not None]


def _number(value):
    """Coerce a cell to float, or None if it is missing / not a number."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _column(data, index):
    """The finite numeric values of one column."""
    out = []
    for row in data:
        if index < len(row):
            value = _number(row[index])
            if value is not None:
                out.append(value)
    return out


# -------------------------------------------------------------- statistics

def _median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


# ------------------------------------------------------- goal-space scoring

def _goal_matrix(data, kinds):
    """Each row's goal values normalized so that 1.0 is always 'best'.

    Lab readings are ignored here: a batch is exceptional because of its
    outcomes, not because of where its chemistry happens to sit.  Missing
    goal values fall back to the column median so one blank cell cannot
    make a batch look heroic.
    """
    goal_indexes = [i for i, kind in enumerate(kinds) if kind != _READING]
    if not goal_indexes:
        return [], []

    columns = []
    for index in goal_indexes:
        values = _column(data, index)
        lo, hi = (min(values), max(values)) if values else (0.0, 0.0)
        columns.append((index, lo, hi, _median(values)))

    matrix = []
    for row in data:
        scaled = []
        for index, lo, hi, med in columns:
            raw = _number(row[index]) if index < len(row) else None
            if raw is None:
                raw = med
            span = hi - lo
            unit = 0.5 if span <= 0 else (raw - lo) / span
            if kinds[index] == _MINIMIZE:
                unit = 1.0 - unit
            scaled.append(unit)
        matrix.append(scaled)
    return matrix, goal_indexes


def _distance_to_heaven(scaled):
    """0.0 == best possible on every goal at once; larger is worse."""
    if not scaled:
        return 0.0
    gap = sum((1.0 - value) ** 2 for value in scaled)
    return math.sqrt(gap / len(scaled))


def _dominates(a, b):
    """True when `a` is at least as good everywhere and better somewhere."""
    better_anywhere = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            better_anywhere = True
    return better_anywhere


def _pareto_ranks(matrix):
    """Rank 0 is the non-dominated front, rank 1 the next shell, and so on.

    O(n^2) in the number of batches, which is nothing at the 320 rows this
    code is called on.
    """
    size = len(matrix)
    ranks = [0] * size
    remaining = set(range(size))
    rank = 0
    while remaining:
        front = [
            i for i in remaining
            if not any(_dominates(matrix[j], matrix[i])
                       for j in remaining if j != i)
        ]
        if not front:                      # cyclic ties: stop the recursion
            front = list(remaining)
        for i in front:
            ranks[i] = rank
        remaining.difference_update(front)
        rank += 1
    return ranks


# ------------------------------------------------------------------- public

def remember(rows):
    """The handful of past batches that were truly exceptional.

    Batches are shelled into Pareto fronts over the goal columns (so a batch
    is not punished for trading one goal off against another), then ordered
    within a shell by distance to the ideal corner of goal space.  The best
    few are returned, capped at 5% of the data rows.
    """
    header, data = _split(rows)
    if not data:
        return []

    budget = max(1, int(MAX_REMEMBERED_FRACTION * len(data)))
    budget = min(budget, len(data))

    kinds = _column_kinds(header)
    matrix, goal_indexes = _goal_matrix(data, kinds)
    if not goal_indexes:
        # Nothing to optimize: no batch can be called exceptional.
        return []

    ranks = _pareto_ranks(matrix)
    order = sorted(
        range(len(data)),
        key=lambda i: (ranks[i], _distance_to_heaven(matrix[i]), i),
    )
    return [list(data[i]) for i in order[:budget]]


def summary(rows):
    """A quick picture of a normal batch: the typical value of every column.

    The median is used throughout.  Wine chemistry columns such as residual
    sugar, chlorides and the sulfur dioxides are strongly right-skewed with
    a long tail of outlier batches, and a mean would describe a batch the
    cellar has never actually made.
    """
    header, data = _split(rows)
    out = {}
    for index, name in enumerate(header):
        out[str(name)] = _median(_column(data, index))
    return out


# ------------------------------------------------------------------ helpers

def remembered_indexes(rows):
    """Row positions (0-based within the data rows) of the exceptional batches.

    Convenience for callers that need to point back at the original table
    rather than carry copies of the rows around.
    """
    header, data = _split(rows)
    chosen = remember(rows)
    taken = set()
    indexes = []
    for pick in chosen:
        for i, row in enumerate(data):
            if i not in taken and list(row) == list(pick):
                taken.add(i)
                indexes.append(i)
                break
    return indexes
