"""Step 1 of the winemaker's assistant.

Two questions from the cellar:

  "Out of all our past batches, show me the handful that were truly
   exceptional."                                    -> remember(rows)

  "Give me a quick picture of a normal batch from our cellar: what each
   lab reading usually looks like."                  -> summary(rows)

Input convention
----------------
`rows` is a list of lists.  rows[0] is the header (strings); every later
row holds numbers.  A column name ending in '+' is a goal to maximize,
one ending in '-' is a goal to minimize, and anything else is a lab
reading (an input the winemaker controls or measures, not a goal).

Only the Python standard library is used.  Both functions are O(n log n)
in the number of rows, so a 320-row split finishes in milliseconds.
"""

import csv
import sys
from collections import Counter

# Fraction of past batches that may be called "exceptional".
KEEP_FRACTION = 0.05


# ---------------------------------------------------------------- header --

def _split_columns(header):
    """Return (goals, readings).

    goals    -- list of (column index, weight) where weight is +1 for a
                column to maximize and -1 for one to minimize.
    readings -- list of indexes of the plain lab-reading columns.
    """
    goals = []
    readings = []
    for i, raw in enumerate(header):
        name = str(raw).strip()
        if name.endswith("+"):
            goals.append((i, 1.0))
        elif name.endswith("-"):
            goals.append((i, -1.0))
        else:
            readings.append(i)
    return goals, readings


def _column(data, i):
    """The finite numeric values of column `i`, skipping blanks/junk."""
    out = []
    for row in data:
        if i >= len(row):
            continue
        v = row[i]
        if v is None or v == "" or v == "?":
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f and abs(f) != float("inf"):  # drop NaN / inf
            out.append(f)
    return out


def _value(row, i):
    """One numeric cell, or None if it is missing or not a number."""
    if i >= len(row):
        return None
    v = row[i]
    if v is None or v == "" or v == "?":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or abs(f) == float("inf"):
        return None
    return f


# ----------------------------------------------------------- "exceptional" --

def _rank_scale(values):
    """Map each distinct value to its percentile position in 0..1.

    A rank scale is used instead of raw numbers so that goals measured on
    different scales (a 0-10 quality score and a percent of alcohol) carry
    equal weight, and so that one freak batch cannot dominate the ranking.
    Ties share the average position.
    """
    if not values:
        return {}
    order = sorted(values)
    n = len(order)
    if n == 1:
        return {order[0]: 0.5}

    scale = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and order[j + 1] == order[i]:
            j += 1
        # average 0-based position of this run of equal values
        pos = (i + j) / 2.0
        scale[order[i]] = pos / (n - 1)
        i = j + 1
    return scale


def _goal_scores(data, goals):
    """One 0..1 desirability score per data row (higher is better)."""
    scales = {}
    for i, _w in goals:
        scales[i] = _rank_scale(_column(data, i))

    scores = []
    for row in data:
        total = 0.0
        weight = 0.0
        for i, w in goals:
            v = _value(row, i)
            if v is None:
                continue
            p = scales[i].get(v)
            if p is None:
                continue
            # minimizing a column means a low reading is the good end
            total += p if w > 0 else (1.0 - p)
            weight += 1.0
        scores.append(total / weight if weight else 0.0)
    return scores


def remember(rows):
    """The handful of past batches that were truly exceptional.

    Batches are ranked by how well they do on the goal columns (the '+'
    and '-' names), each goal converted to a percentile so the goals are
    comparable, then averaged.  At most 5% of the data rows come back,
    best first.  The header row is not included.
    """
    if not rows or len(rows) < 2:
        return []

    header = rows[0]
    data = [r for r in rows[1:]]
    goals, _readings = _split_columns(header)

    keep = int(len(data) * KEEP_FRACTION)  # floor: never more than 5%
    if keep < 1:
        keep = 1
    if keep > len(data):
        keep = len(data)

    if not goals:
        # No goal columns to judge by: nothing is defensibly exceptional.
        return []

    scores = _goal_scores(data, goals)
    # sort by score, descending; original order breaks ties, so the result
    # is deterministic for a given input
    order = sorted(range(len(data)), key=lambda k: (-scores[k], k))
    return [data[k] for k in order[:keep]]


# ------------------------------------------------------------- "normal batch" --

def _median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _typical(values):
    """A robust single stand-in for a column of readings.

    The median, because lab readings here are skewed and a stuck sensor or
    one odd barrel should not move the picture.  Columns that only ever
    hold whole numbers (a 1-10 quality score, say) report the most common
    whole number instead, so the answer stays a value a batch could
    actually have rather than something like 5.5.
    """
    if not values:
        return None

    med = _median(values)
    if all(float(v).is_integer() for v in values):
        counts = Counter(int(v) for v in values)
        best = max(counts.values())
        # most common; if several are equally common, the one nearest the median
        winners = [v for v, c in counts.items() if c == best]
        return min(winners, key=lambda v: (abs(v - med), v))
    return med


def summary(rows):
    """A quick picture of a normal batch: each column's typical value.

    Keys are the column names exactly as they appear in the header (goal
    markers '+' / '-' kept), so the result lines up with the data.  A
    column with no usable numbers maps to None.
    """
    out = {}
    if not rows:
        return out

    header = rows[0]
    data = rows[1:]
    for i, raw in enumerate(header):
        out[str(raw).strip()] = _typical(_column(data, i))
    return out


# -------------------------------------------------------------------- CLI --

def load_csv(path):
    """Read a data file into the rows format used above."""
    rows = []
    with open(path, newline="") as f:
        for n, fields in enumerate(csv.reader(f)):
            if not fields:
                continue
            if n == 0:
                rows.append([s.strip() for s in fields])
            else:
                rows.append([float(s) for s in fields])
    return rows


def main(argv):
    if len(argv) != 2:
        print("usage: python step1.py data.csv")
        return 1

    rows = load_csv(argv[1])
    header = rows[0]

    print("A normal batch from the cellar:")
    for name, value in summary(rows).items():
        print("  %-18s %s" % (name, value))

    best = remember(rows)
    print("\n%d exceptional batches out of %d:" % (len(best), len(rows) - 1))
    print("  " + ", ".join("%-10s" % h for h in header))
    for row in best:
        print("  " + ", ".join("%-10s" % v for v in row))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
