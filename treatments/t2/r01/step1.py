"""Step 1 of the winemaker's assistant.

Answers two questions about past batches:

  remember(rows) -- "show me the handful of batches that were truly exceptional"
  summary(rows)  -- "what does a normal batch look like, reading by reading"

Column meaning is taken from the header row: a name ending in ``+`` is a
goal to maximize, a name ending in ``-`` is a goal to minimize, and anything
else is a lab reading that is described but not optimized.

Plain Python 3, standard library only.
"""

from math import floor

# At most this fraction of the data rows may be called exceptional.
EXCEPTIONAL_FRACTION = 0.05


# ---------------------------------------------------------------- header info

def _classify(header):
    """Return (goal_columns, reading_columns).

    goal_columns is a list of (index, direction) where direction is +1 for a
    column we want large and -1 for a column we want small.
    """
    goals, readings = [], []
    for i, name in enumerate(header):
        text = str(name).strip()
        if text.endswith("+"):
            goals.append((i, 1))
        elif text.endswith("-"):
            goals.append((i, -1))
        else:
            readings.append(i)
    return goals, readings


def _number(value):
    """Best-effort conversion to float; None when the cell is unusable."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "" or text == "?":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _column(data, index):
    """Every usable number in one column, in row order."""
    out = []
    for row in data:
        if index < len(row):
            x = _number(row[index])
            if x is not None:
                out.append(x)
    return out


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _split(rows):
    """Separate the header row from the data rows."""
    if not rows:
        return [], []
    return list(rows[0]), [r for r in rows[1:] if r]


# ------------------------------------------------------------------- scoring

def _oriented(data, goals):
    """Per-row goal values rewritten so that bigger is always better.

    Missing cells fall back to the column median, so a row with a gap is
    treated as merely average on that goal rather than being discarded.
    """
    fallbacks = {}
    for index, _ in goals:
        fallbacks[index] = _median(_column(data, index))

    table = []
    for row in data:
        values = []
        for index, direction in goals:
            x = _number(row[index]) if index < len(row) else None
            if x is None:
                x = fallbacks[index]
            if x is None:
                x = 0.0
            values.append(direction * x)
        table.append(values)
    return table


def _scores(table, count):
    """Min-max normalize each goal, then average: one score per row."""
    if not table:
        return []
    lo = [min(col) for col in zip(*table)]
    hi = [max(col) for col in zip(*table)]
    scores = []
    for values in table:
        total = 0.0
        for k, x in enumerate(values):
            span = hi[k] - lo[k]
            total += 0.5 if span <= 0 else (x - lo[k]) / span
        scores.append(total / count)
    return scores


def _nondominated(table):
    """Flags marking rows on the Pareto front of the goal columns."""
    n = len(table)
    flags = [True] * n
    for i in range(n):
        a = table[i]
        for j in range(n):
            if i == j or not flags[i]:
                continue
            b = table[j]
            # does b dominate a?
            better_anywhere = False
            worse_anywhere = False
            for k in range(len(a)):
                if b[k] > a[k]:
                    better_anywhere = True
                elif b[k] < a[k]:
                    worse_anywhere = True
                    break
            if better_anywhere and not worse_anywhere:
                flags[i] = False
    return flags


# -------------------------------------------------------------- the two asks

def remember(rows):
    """The handful of truly exceptional past batches.

    Returns at most 5% of the data rows (always at least one when there is
    any data), best first.  Batches on the Pareto front of the goal columns
    come first; within each tier the combined goal score breaks the tie.
    """
    header, data = _split(rows)
    if not data:
        return []

    goals, _ = _classify(header)
    budget = max(1, int(floor(len(data) * EXCEPTIONAL_FRACTION)))
    budget = min(budget, len(data))

    if not goals:
        # No goal columns declared: nothing distinguishes the batches.
        return [list(row) for row in data[:budget]]

    table = _oriented(data, goals)
    scores = _scores(table, len(goals))
    front = _nondominated(table)

    order = sorted(
        range(len(data)),
        key=lambda i: (0 if front[i] else 1, -scores[i], i),
    )
    return [list(data[i]) for i in order[:budget]]


def summary(rows):
    """A quick picture of a normal batch: typical value per column.

    The median is used throughout -- it is the value a winemaker would
    recognize as "usual" and it is not dragged around by the odd outlier
    batch.  Keys are the column names exactly as they appear in the header,
    so the ``+``/``-`` goal markers are preserved.
    """
    header, data = _split(rows)
    out = {}
    for i, name in enumerate(header):
        out[name] = _median(_column(data, i))
    return out
