"""Winemaker's AI assistant -- step 1.

Reads batch tables shaped like::

    [["Fixedacidity", ..., "Alcohol-", "Quality+"],
     [7.4, ..., 9.4, 5],
     ...]

The header row names the columns.  A name ending in ``+`` is a goal to be
maximized, a name ending in ``-`` is a goal to be minimized, and every other
column is an ordinary lab reading.

Two questions are answered here:

``remember(rows)``
    "Out of all our past batches, show me the handful that were truly
    exceptional."

``summary(rows)``
    "Give me a quick picture of a normal batch from our cellar: what each lab
    reading usually looks like."
"""

import csv

MAXIMIZE = "+"
MINIMIZE = "-"

# A "handful": never more than this fraction of the recorded batches.
EXCEPTIONAL_FRACTION = 0.05


# ---------------------------------------------------------------- header


def classify(header):
    """Return (goals, readings).

    ``goals`` is a list of ``(index, name, direction)`` where direction is
    ``+1`` for columns to maximize and ``-1`` for columns to minimize.
    ``readings`` is a list of ``(index, name)`` for the plain lab columns.
    """
    goals = []
    readings = []
    for i, name in enumerate(header):
        name = str(name).strip()
        if name.endswith(MAXIMIZE):
            goals.append((i, name, 1))
        elif name.endswith(MINIMIZE):
            goals.append((i, name, -1))
        else:
            readings.append((i, name))
    return goals, readings


def split(rows):
    """Split a table into its header row and its data rows."""
    if not rows:
        return [], []
    return list(rows[0]), [row for row in rows[1:]]


def number(value):
    """Best-effort conversion of a cell to a float, or None if impossible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text == "?":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def column(data, index):
    """The usable (numeric) values of one column."""
    out = []
    for row in data:
        if index < len(row):
            value = number(row[index])
            if value is not None:
                out.append(value)
    return out


# ---------------------------------------------------------------- statistics


def median(values):
    """Middle value -- the typical reading, unmoved by a few odd batches."""
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    middle = n // 2
    if n % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def mean(values):
    return sum(values) / float(len(values)) if values else None


def spread(values):
    """Standard deviation (population); 0.0 when there is nothing to vary."""
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return (sum((v - mu) ** 2 for v in values) / float(len(values))) ** 0.5


# ---------------------------------------------------------------- scoring


def _oriented(data, goals):
    """Rescale every goal column to 0..1 where 1 is always better.

    Rows missing a goal value are given the worst score for that goal, so a
    batch is never called exceptional on the strength of a missing reading.
    """
    scales = []
    for index, _name, direction in goals:
        values = column(data, index)
        low = min(values) if values else 0.0
        high = max(values) if values else 0.0
        scales.append((index, direction, low, high))

    oriented = []
    for row in data:
        point = []
        for index, direction, low, high in scales:
            value = number(row[index]) if index < len(row) else None
            if value is None or high <= low:
                point.append(0.0 if value is None else 0.5)
                continue
            fraction = (value - low) / (high - low)
            point.append(fraction if direction > 0 else 1.0 - fraction)
        oriented.append(tuple(point))
    return oriented


def _dominates(a, b):
    """True when point ``a`` is at least as good everywhere and better somewhere."""
    better_somewhere = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            better_somewhere = True
    return better_somewhere


def _pareto(points):
    """Indices of the points no other point beats outright."""
    front = []
    for i, point in enumerate(points):
        if not any(_dominates(other, point) for j, other in enumerate(points) if j != i):
            front.append(i)
    return front


# ---------------------------------------------------------------- answers


def remember(rows, fraction=EXCEPTIONAL_FRACTION):
    """The exceptional past batches, at most ``fraction`` of the data rows.

    Batches are judged only on the goal columns (``+`` / ``-``).  Those that
    nothing else beats on every goal at once come first -- these are the
    genuinely unbeaten batches -- ranked by their average goal score, and the
    handful is filled out with the next best scorers if needed.

    At least one batch is returned whenever there is data to look at, even for
    a cellar with fewer than twenty recorded batches.
    """
    header, data = split(rows)
    if not data:
        return []

    goals, _readings = classify(header)
    if not goals:
        return []

    keep = int(len(data) * fraction)
    if keep < 1:
        keep = 1

    points = _oriented(data, goals)
    scores = [mean(point) or 0.0 for point in points]

    front = set(_pareto(points))
    order = sorted(
        range(len(data)),
        key=lambda i: (0 if i in front else 1, -scores[i], i),
    )
    return [data[i] for i in order[:keep]]


def summary(rows):
    """Map every column name to its typical value for a normal batch.

    The median is used: it describes the batch in the middle of the cellar
    rather than being dragged around by a handful of unusual ones.  Columns
    with no usable numbers map to ``None``.
    """
    header, data = split(rows)
    picture = {}
    for index, name in enumerate(header):
        picture[str(name).strip()] = median(column(data, index))
    return picture


def describe(rows):
    """A longer picture of a normal batch: typical value and usual swing."""
    header, data = split(rows)
    goals, _readings = classify(header)
    directions = {index: direction for index, _name, direction in goals}

    out = {}
    for index, name in enumerate(header):
        values = column(data, index)
        out[str(name).strip()] = {
            "typical": median(values),
            "spread": spread(values),
            "low": min(values) if values else None,
            "high": max(values) if values else None,
            "goal": {1: "maximize", -1: "minimize"}.get(directions.get(index), "reading"),
        }
    return out


# ---------------------------------------------------------------- csv helper


def load_rows(path):
    """Read a CSV into the ``[header, *numeric rows]`` shape used above."""
    with open(path, "r", newline="") as handle:
        raw = [row for row in csv.reader(handle) if row]
    if not raw:
        return []
    header = [cell.strip() for cell in raw[0]]
    data = []
    for row in raw[1:]:
        data.append([number(cell) for cell in row])
    return [header] + data


def _show(path):
    rows = load_rows(path)
    header, data = split(rows)
    print("batches recorded: %d" % len(data))

    print("\na normal batch from the cellar:")
    picture = describe(rows)
    for name in header:
        name = name.strip()
        item = picture[name]
        typical = "n/a" if item["typical"] is None else "%g" % item["typical"]
        print("  %-16s %10s  (+/- %g)  [%s]" % (name, typical, item["spread"], item["goal"]))

    print("\ntruly exceptional batches:")
    best = remember(rows)
    print("  " + ", ".join(name.strip() for name in header))
    for row in best:
        print("  " + ", ".join("%g" % value if value is not None else "?" for value in row))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python step1.py <batches.csv>")
        raise SystemExit(2)
    _show(sys.argv[1])
