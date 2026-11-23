"""Winemaker's AI assistant -- step 2.

Builds on :mod:`step1`, which already knows how to read a batch table, tell a
goal column (``+`` / ``-``) from an ordinary lab reading, and describe a normal
batch.  Everything here reuses those pieces.

Two more questions are answered:

``guess(train, row)``
    "Before the panel sits, estimate what score this new batch will get from
    its lab sheet alone."

``kinds(train)``
    "Group our batches into a few kinds, and tell me what score to expect from
    each kind."

The score being guessed is the panel's ``Quality+`` column.  Everything else on
the sheet -- including the other goal, ``Alcohol-`` -- is fair game as a clue,
because those numbers come off the lab bench before the panel ever tastes
anything.
"""

import math

import step1
from step1 import classify, column, mean, median, number, split, spread

TARGET = "Quality+"

# A guess is an average over this many look-alike batches at most.
NEIGHBOURS_CAP = 10

# "A few kinds" means somewhere in this range.
FEWEST_KINDS = 2
MOST_KINDS = 5

# How many times the grouping is allowed to settle.
ROUNDS = 20


# ---------------------------------------------------------------- header


def target_column(header):
    """Return ``(index, name, direction)`` for the score we are predicting.

    ``Quality+`` by name when it is there; otherwise the first column to be
    maximized; otherwise the first goal of any kind.  ``None`` when the table
    has no goal column at all.
    """
    goals, _readings = classify(header)
    if not goals:
        return None
    for goal in goals:
        if goal[1] == TARGET:
            return goal
    for goal in goals:
        if goal[2] > 0:
            return goal
    return goals[0]


# ---------------------------------------------------------------- geometry


def _scales(data, indexes):
    """Per-column ``(index, low, high, typical)`` taken from the train rows."""
    out = []
    for index in indexes:
        values = column(data, index)
        low = min(values) if values else 0.0
        high = max(values) if values else 0.0
        out.append((index, low, high, median(values)))
    return out


def _point(row, scales):
    """A lab sheet as a 0..1 point, so no column shouts louder than the rest.

    A missing reading is stood in for by the typical value of that column, and
    a new batch beyond anything seen before is pinned to the edge of the known
    range rather than being allowed to run off to infinity.
    """
    point = []
    for index, low, high, typical in scales:
        value = number(row[index]) if index < len(row) else None
        if value is None:
            value = typical
        if value is None or high <= low:
            point.append(0.5)
        else:
            fraction = (value - low) / (high - low)
            point.append(min(1.0, max(0.0, fraction)))
    return tuple(point)


def _distance(a, b):
    """How unlike two lab sheets are."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _centre(points):
    """The average lab sheet of a bunch of batches."""
    width = len(points[0])
    return tuple(mean([p[d] for p in points]) for d in range(width))


# ---------------------------------------------------------------- the cellar


class Cellar(object):
    """What step 2 remembers: every past lab sheet, scaled, beside its score.

    Nothing is fitted or tuned here -- the cellar simply keeps the batches in a
    shape where "which past batches does this new one resemble?" is a question
    with an answer.
    """

    def __init__(self, train):
        self.header, self.rows = split(train)
        self.target = target_column(self.header)
        self.target_index = self.target[0] if self.target else None

        clues = [i for i in range(len(self.header)) if i != self.target_index]
        self.scales = _scales(self.rows, clues)

        self.points = [_point(row, self.scales) for row in self.rows]
        self.scores = []
        for row in self.rows:
            index = self.target_index
            if index is None or index >= len(row):
                self.scores.append(None)
            else:
                self.scores.append(number(row[index]))

        known = [s for s in self.scores if s is not None]
        self.typical = median(known)
        self.swing = spread(known)
        self.scored = [i for i, s in enumerate(self.scores) if s is not None]

    # -------------------------------------------------- one new batch

    def neighbours(self, row, count):
        """The ``count`` scored past batches whose sheets look most like ``row``."""
        point = _point(row, self.scales)
        ranked = sorted(
            ((_distance(point, self.points[i]), i) for i in self.scored),
            key=lambda pair: pair,
        )
        return ranked[:count]

    def guess(self, row, count=None):
        """Expected score for one new batch, from its lab sheet alone.

        The batch is compared with every scored batch in the cellar and the
        closest handful have their scores averaged, the nearest counting most.
        With nothing to compare against, the typical cellar score is the honest
        answer.
        """
        if not self.scored:
            return float(self.typical) if self.typical is not None else 0.0

        if count is None:
            count = int(round(math.sqrt(len(self.scored))))
        count = max(1, min(NEIGHBOURS_CAP, count, len(self.scored)))

        close = self.neighbours(row, count)

        exact = [self.scores[i] for gap, i in close if gap <= 0.0]
        if exact:
            return float(mean(exact))

        weight_total = 0.0
        score_total = 0.0
        for gap, i in close:
            weight = 1.0 / gap
            weight_total += weight
            score_total += weight * self.scores[i]
        if weight_total <= 0.0:
            return float(mean([self.scores[i] for _gap, i in close]))
        return float(score_total / weight_total)

    # -------------------------------------------------- kinds of batch

    def how_many_kinds(self):
        """A few: enough to be useful, few enough to keep in your head."""
        n = len(self.points)
        if n < FEWEST_KINDS:
            return 1
        wanted = int(round(math.sqrt(n / 2.0)))
        return max(FEWEST_KINDS, min(MOST_KINDS, wanted, n))

    def groups(self, count=None):
        """Split the cellar into kinds; returns a list of lists of row indices."""
        if not self.points:
            return []
        if count is None:
            count = self.how_many_kinds()
        count = max(1, min(count, len(self.points)))
        if count == 1:
            return [list(range(len(self.points)))]

        labels = _settle(self.points, _seeds(self.points, count))
        buckets = [[] for _ in range(count)]
        for i, label in enumerate(labels):
            buckets[label].append(i)
        return [bucket for bucket in buckets if bucket]

    def expected(self, indexes):
        """The score to expect from a kind of batch: its middle score."""
        known = [self.scores[i] for i in indexes if self.scores[i] is not None]
        if known:
            return float(median(known))
        return float(self.typical) if self.typical is not None else 0.0


# ---------------------------------------------------------------- grouping


def _seeds(points, count):
    """Starting points, picked the same way every time.

    Begin at the batch nearest the middle of the cellar, then keep adding the
    batch least like anything picked so far.  No dice are rolled, so the same
    cellar always yields the same kinds.
    """
    middle = _centre(points)
    first = min(range(len(points)), key=lambda i: (_distance(points[i], middle), i))
    chosen = [points[first]]
    while len(chosen) < count:
        far = max(
            range(len(points)),
            key=lambda i: (min(_distance(points[i], c) for c in chosen), -i),
        )
        chosen.append(points[far])
    return chosen


def _settle(points, centres, rounds=ROUNDS):
    """Hand each batch to its nearest centre, re-centre, repeat until calm."""
    centres = list(centres)
    labels = [None] * len(points)
    for _round in range(rounds):
        changed = False
        for i, point in enumerate(points):
            best = min(
                range(len(centres)),
                key=lambda c: (_distance(point, centres[c]), c),
            )
            if best != labels[i]:
                labels[i] = best
                changed = True
        for c in range(len(centres)):
            members = [points[i] for i in range(len(points)) if labels[i] == c]
            if members:
                centres[c] = _centre(members)
        if not changed:
            break
    return labels


# ---------------------------------------------------------------- answers


def guess(train, row, neighbours=None):
    """Predicted ``Quality+`` for one new batch, learned only from ``train``.

    ``train`` is a table in the step 1 shape (header row first); ``row`` is a
    single batch's lab sheet laid out like the train rows.  Any ``Quality+``
    value already sitting in ``row`` is ignored -- the point is to answer
    before the panel sits.
    """
    return Cellar(train).guess(row, neighbours)


def kinds(train, groups=None):
    """The few kinds of batch in the cellar, best-scoring kind first.

    Returns a list of ``(group_rows, expected_quality)`` pairs, where
    ``group_rows`` are the train data rows (no header) belonging to that kind
    and ``expected_quality`` is the score to expect from it.  Batches are
    grouped on their lab sheets alone; the scores are only read off afterwards,
    so a kind is a kind of wine, not a kind of verdict.
    """
    cellar = Cellar(train)
    out = []
    for bucket in cellar.groups(groups):
        out.append(([cellar.rows[i] for i in bucket], cellar.expected(bucket)))
    out.sort(key=lambda pair: (-pair[1], -len(pair[0])))
    return out


# ---------------------------------------------------------------- checking


def check(rows, every=5):
    """How wrong the guesses are, holding back every ``every``-th batch.

    Reports the average miss alongside the miss you would get by always saying
    "a normal score", so the guessing can be seen to be worth something.
    """
    header, data = split(rows)
    if not data:
        return {"tested": 0, "error": None, "baseline": None}

    train = [header] + [row for i, row in enumerate(data) if i % every]
    tests = [row for i, row in enumerate(data) if not i % every]

    cellar = Cellar(train)
    if cellar.target_index is None:
        return {"tested": 0, "error": None, "baseline": None}

    misses = []
    flat = []
    for row in tests:
        truth = number(row[cellar.target_index]) if cellar.target_index < len(row) else None
        if truth is None:
            continue
        misses.append(abs(cellar.guess(row) - truth))
        if cellar.typical is not None:
            flat.append(abs(cellar.typical - truth))

    return {
        "tested": len(misses),
        "error": mean(misses),
        "baseline": mean(flat),
        "swing": cellar.swing,
    }


# ---------------------------------------------------------------- demo


def _show(path):
    rows = step1.load_rows(path)
    header, data = split(rows)
    print("batches recorded: %d" % len(data))

    target = target_column(header)
    print("guessing: %s" % (target[1] if target else "nothing -- no goal column"))

    report = check(rows)
    if report["tested"]:
        print(
            "\nheld-back batches: %d\n  average miss: %.3f  (always-normal would miss %.3f)"
            % (report["tested"], report["error"], report["baseline"])
        )

    print("\nkinds of batch in the cellar:")
    goals, readings = classify(header)
    for number_, (group, expected) in enumerate(kinds(rows), 1):
        print("  kind %d: %d batches, expect %s %.2f" % (
            number_, len(group), target[1] if target else "score", expected))
        for index, name in readings[:4]:
            print("    %-16s typically %s" % (
                name, "%g" % median(column(group, index)) if column(group, index) else "n/a"))

    if data:
        print("\nbefore the panel sits, for the first batch on file:")
        print("  guess: %.2f" % guess(rows, data[0]))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python step2.py <batches.csv>")
        raise SystemExit(2)
    _show(sys.argv[1])
