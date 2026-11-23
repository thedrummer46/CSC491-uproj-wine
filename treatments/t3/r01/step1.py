"""step1.py -- helpers for the winemaker's AI assistant.

Data convention
---------------
`rows` is a list of lists.  `rows[0]` is the header (strings); every later row
holds numbers.  A column name ending in ``+`` is a goal to be maximized, one
ending in ``-`` is a goal to be minimized, and anything else is a lab reading.

Two questions are answered here:

* "Show me the handful of batches that were truly exceptional."   -> remember()
* "What does a normal batch look like, reading by reading?"        -> summary()
"""

from __future__ import annotations

import math

MAX_FRACTION = 0.05  # "a handful": never more than 5% of the data rows


# --------------------------------------------------------------------------
# header / parsing helpers
# --------------------------------------------------------------------------
def _split_header(header):
    """Return (goals, readings) as lists of (index, name, direction).

    direction is +1 for a column we want large, -1 for one we want small.
    Readings get direction 0.
    """
    goals, readings = [], []
    for i, raw in enumerate(header):
        name = str(raw).strip()
        if name.endswith("+"):
            goals.append((i, name, 1))
        elif name.endswith("-"):
            goals.append((i, name, -1))
        else:
            readings.append((i, name, 0))
    return goals, readings


def _as_float(value):
    """Best-effort numeric conversion; returns None for unusable cells."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    try:
        f = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _column(data, index):
    """Numeric values of one column, skipping unusable cells."""
    out = []
    for row in data:
        if index < len(row):
            v = _as_float(row[index])
            if v is not None:
                out.append(v)
    return out


def _median(values):
    if not values:
        return 0.0
    vs = sorted(values)
    n = len(vs)
    mid = n // 2
    if n % 2:
        return float(vs[mid])
    return (vs[mid - 1] + vs[mid]) / 2.0


# --------------------------------------------------------------------------
# "show me the exceptional batches"
# --------------------------------------------------------------------------
def _goal_score(data, goals):
    """One score per row: mean of direction-adjusted, min-max normalized goals.

    Normalizing per goal puts every objective on a 0..1 scale (1 = best seen),
    so goals measured in different units contribute comparably.  A row missing
    a goal value is treated as worst-case on that goal.
    """
    n = len(data)
    totals = [0.0] * n
    used = 0

    for index, _name, direction in goals:
        values = _column(data, index)
        if not values:
            continue
        lo, hi = min(values), max(values)
        span = hi - lo
        used += 1
        for r, row in enumerate(data):
            v = _as_float(row[index]) if index < len(row) else None
            if v is None:
                continue  # worst case: contributes 0
            frac = 0.5 if span <= 0 else (v - lo) / span
            totals[r] += frac if direction > 0 else 1.0 - frac

    if used == 0:
        return [0.0] * n
    return [t / used for t in totals]


def remember(rows):
    """Return the exceptional past batches, best first.

    At most 5% of the data rows are returned (e.g. 16 of 320).  Batches are
    ranked by how well they satisfy every goal column at once.
    """
    if not rows or len(rows) < 2:
        return []

    header = rows[0]
    data = [row for row in rows[1:] if row is not None]
    if not data:
        return []

    goals, _readings = _split_header(header)

    budget = int(len(data) * MAX_FRACTION)  # floor -> never exceeds 5%
    budget = max(1, min(budget, len(data)))

    if not goals:
        return list(data[:budget])

    scores = _goal_score(data, goals)
    order = sorted(range(len(data)), key=lambda r: (-scores[r], r))
    return [data[r] for r in order[:budget]]


# --------------------------------------------------------------------------
# "a quick picture of a normal batch"
# --------------------------------------------------------------------------
def summary(rows):
    """Map every column name to its typical value for this cellar.

    The median is used: it is the value a normal batch sits at and it is not
    dragged around by the occasional odd barrel.
    """
    if not rows:
        return {}

    header = [str(h).strip() for h in rows[0]]
    data = [row for row in rows[1:] if row is not None]

    out = {}
    for i, name in enumerate(header):
        out[name] = _median(_column(data, i))
    return out
