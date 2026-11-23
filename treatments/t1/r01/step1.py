"""Step 1 of the winemaker's assistant.

The cellar's records arrive as a list of lists: row 0 is the header, every
later row is one batch of numbers.  Header names carry their own meaning:

    "Quality+"   -> a goal we want to be high
    "Alcohol-"   -> a goal we want to be low
    anything else -> a lab reading, neither good nor bad on its own

Two questions are answered here:

    remember(rows) -> the handful of truly exceptional past batches
    summary(rows)  -> what a normal batch looks like, column by column
"""

import csv
import math
import sys

# "At most 5% of the data rows" is a hard cap; "a handful" is the winemaker's
# own word for how many they actually want to look at.
MAX_EXCEPTIONAL_FRACTION = 0.05
HANDFUL = 5


def _split(rows):
    """Return (header names, data rows).  Tolerates empty / header-only input."""
    if not rows:
        return [], []
    header = [str(name).strip() for name in rows[0]]
    return header, list(rows[1:])


def _goal_columns(header):
    """Columns we judge batches by, as (index, direction).  +1 = higher is better."""
    goals = []
    for index, name in enumerate(header):
        if name.endswith("+"):
            goals.append((index, 1.0))
        elif name.endswith("-"):
            goals.append((index, -1.0))
    return goals


def _number(row, index):
    """The numeric value at `index`, or None when it is missing or unusable."""
    try:
        value = float(row[index])
    except (IndexError, TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _column(data, index):
    return [v for v in (_number(row, index) for row in data) if v is not None]


def _median(values):
    """Middle value; the typical batch should not be dragged around by outliers."""
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return None
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def summary(rows):
    """A quick picture of a normal batch: each column name -> its typical value.

    The typical value is the median of that column across all past batches.
    A column with no usable numbers maps to None.
    """
    header, data = _split(rows)
    return {name: _median(_column(data, index)) for index, name in enumerate(header)}


def _batch_score(row, goals, spans):
    """How exceptional one batch is: 0.0 (worst seen) to 1.0 (best seen).

    Each goal is rescaled onto 0..1 using the range actually seen in the
    cellar, flipped when lower is better, then averaged so that no single
    goal's units dominate.  None when the batch has no usable goal readings.
    """
    parts = []
    for index, direction in goals:
        low, high = spans.get(index, (None, None))
        if low is None or high <= low:
            continue  # constant column: tells us nothing about exceptionality
        value = _number(row, index)
        if value is None:
            continue
        fraction = (value - low) / (high - low)
        parts.append(fraction if direction > 0 else 1.0 - fraction)
    if not parts:
        return None
    return sum(parts) / len(parts)


def remember(rows):
    """The exceptional batches, best first, at most 5% of the data rows.

    Batches are ranked on every goal column at once (maximizing the "+"
    columns, minimizing the "-" ones).  Lab readings are deliberately
    ignored here: they describe a batch, they do not make it good.

    The 5% cap is honoured strictly, so a cellar with fewer than 20 recorded
    batches has no batch exceptional enough to single out and gets [].
    """
    header, data = _split(rows)
    goals = _goal_columns(header)
    if not data or not goals:
        return []

    spans = {}
    for index, _direction in goals:
        values = _column(data, index)
        spans[index] = (min(values), max(values)) if values else (None, None)

    scored = []
    for position, row in enumerate(data):
        score = _batch_score(row, goals, spans)
        if score is not None:
            # position keeps ties in original cellar order
            scored.append((-score, position, row))
    if not scored:
        return []

    keep = min(HANDFUL, int(len(data) * MAX_EXCEPTIONAL_FRACTION), len(scored))
    if keep <= 0:
        return []

    scored.sort(key=lambda item: (item[0], item[1]))
    return [row for _negated_score, _position, row in scored[:keep]]


def load_csv(path):
    """Read a cellar CSV into the list-of-lists shape these functions expect."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        raw = [line for line in csv.reader(handle) if line]
    if not raw:
        return []
    rows = [raw[0]]
    for line in raw[1:]:
        rows.append([_number(line, i) for i in range(len(line))])
    return rows


def _report(rows):
    header, data = _split(rows)
    print("%d past batches, %d columns" % (len(data), len(header)))

    print("\nA normal batch from the cellar:")
    typical = summary(rows)
    width = max((len(name) for name in header), default=0)
    for name in header:
        value = typical[name]
        shown = "n/a" if value is None else "%.4g" % value
        print("  %-*s  %s" % (width, name, shown))

    exceptional = remember(rows)
    print("\nTruly exceptional batches (%d):" % len(exceptional))
    if not exceptional:
        print("  none singled out")
        return
    print("  " + ", ".join(header))
    for row in exceptional:
        print("  " + ", ".join("n/a" if v is None else "%.4g" % v for v in row))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python step1.py <cellar.csv>", file=sys.stderr)
        raise SystemExit(2)
    _report(load_csv(sys.argv[1]))
