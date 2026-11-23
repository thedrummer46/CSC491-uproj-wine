"""Step 2 of the winemaker's assistant.

Rows arrive in the same shape as step 1: row 0 is the header, every later row
is one batch of numbers, and the header names still carry their own meaning
("Quality+" is the score the tasting panel awards, "Alcohol-" a goal to keep
low, the rest plain lab readings).  The small readers and helpers from step 1
are reused here rather than rewritten.

Two more questions are answered:

    guess(train, row) -> the score the panel will probably give one batch
    kinds(train)      -> a few kinds of batch, each with the score to expect

Both work from the lab sheet, meaning every column except the panel's score:
those numbers are all on the bench before the panel ever sits, so they are
fair game as inputs.  Alcohol is one of them -- it is listed as "Alcohol-"
because the winemaker wants it low, not because it is unknown at this point.
Only "Quality+" is held back, since that is the thing being predicted.
"""

import math
import random
import sys

import step1

# A prediction leans on the closest past batches; never more than this many,
# so one odd neighbour in a huge cellar cannot sway the answer.
NEIGHBOURS_CAP = 10

# "A few kinds" in the winemaker's sense: more than one, few enough to talk
# about over a glass.
MIN_KINDS = 2
MAX_KINDS = 5

KMEANS_RESTARTS = 5
KMEANS_ITERATIONS = 50
SEED = 20240101  # fixed, so the same cellar always yields the same kinds


def _quality_column(header):
    """Index of the panel's score, i.e. "Quality+"."""
    for index, name in enumerate(header):
        if name.endswith("+") and name[:-1].strip().lower() == "quality":
            return index
    for index, name in enumerate(header):  # no column literally named Quality
        if name.endswith("+"):             # then any goal to maximise will do
            return index
    return None


def _sheet_columns(header, target):
    """The lab sheet: every column except the score we are trying to predict."""
    return [index for index in range(len(header)) if index != target]


def _spread_fallback(values, centre):
    """Plain standard deviation, for columns where the robust spread is zero."""
    if len(values) < 2:
        return 0.0
    variance = sum((value - centre) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _scales(data, columns):
    """Per reading: (index, centre, spread), so every reading counts the same.

    Readings come in wildly different units -- grams per litre, pH, degrees of
    alcohol.  Centring each on its median and dividing by a robust spread
    (the median absolute deviation) stops whichever column happens to carry
    the biggest numbers from deciding on its own which batches look alike.

    Columns that never vary are dropped: they cannot tell two batches apart.
    """
    scales = []
    for index in columns:
        values = step1._column(data, index)
        if not values:
            continue
        centre = step1._median(values)
        spread = 1.4826 * step1._median([abs(value - centre) for value in values])
        if spread <= 0:
            spread = _spread_fallback(values, centre)
        if spread <= 0:
            continue
        scales.append((index, centre, spread))
    return scales


def _vector(row, scales):
    """One batch as a list of comparable numbers; None where a reading is missing."""
    vector = []
    for index, centre, spread in scales:
        value = step1._number(row, index)
        vector.append(None if value is None else (value - centre) / spread)
    return vector


def _filled(vector):
    """The same batch with gaps closed up at the typical value (0.0 once scaled)."""
    return [0.0 if value is None else value for value in vector]


def _distance(left, right):
    """Typical distance per shared reading; None when the two share none.

    Dividing by the number of readings actually compared keeps a batch with a
    half-finished lab sheet on the same footing as a complete one.
    """
    total = 0.0
    shared = 0
    for a, b in zip(left, right):
        if a is None or b is None:
            continue
        total += (a - b) ** 2
        shared += 1
    if shared == 0:
        return None
    return math.sqrt(total / shared)


def guess(train, row):
    """The score the panel will probably give `row`, from its lab sheet alone.

    The cellar's own history is the model.  Find the past batches whose
    readings look most like this one, then average the scores they were
    awarded, leaning hardest on the closest matches.  Nothing is assumed
    about the shape of the relationship between chemistry and taste, which
    suits a few hundred batches better than a fitted curve would.

    The batch's own Quality+ is never read, so an unjudged batch (score
    blank, or the column absent altogether) is perfectly fine.  When the lab
    sheet says nothing usable the answer falls back to the typical past
    score, and the result is never outside the range of scores ever awarded.
    """
    header, data = step1._split(train)
    target = _quality_column(header)
    if target is None:
        return 0.0

    judged = []
    scores = []
    for past in data:
        score = step1._number(past, target)
        if score is not None:
            judged.append(past)
            scores.append(score)
    if not scores:
        return 0.0

    typical = float(step1._median(scores))
    lowest, highest = min(scores), max(scores)

    scales = _scales(judged, _sheet_columns(header, target))
    if not scales:
        return typical
    wanted = _vector(row, scales)
    if all(value is None for value in wanted):
        return typical

    measured = []
    for past, score in zip(judged, scores):
        distance = _distance(wanted, _vector(past, scales))
        if distance is not None:
            measured.append((distance, score))
    if not measured:
        return typical

    measured.sort(key=lambda pair: pair[0])
    count = max(1, min(NEIGHBOURS_CAP, int(round(math.sqrt(len(measured))))))
    near = measured[:count]

    # The furthest neighbour kept sets the scale of "close".  Softening the
    # weights by a tenth of it stops an all-but-identical past batch from
    # drowning out the rest, and keeps an exact match finite.
    softener = 0.1 * near[-1][0] if near[-1][0] > 0 else 0.1
    predicted = 0.0
    weight_total = 0.0
    for distance, score in near:
        weight = 1.0 / (distance + softener)
        predicted += weight * score
        weight_total += weight
    if weight_total <= 0:
        return typical

    return float(min(highest, max(lowest, predicted / weight_total)))


def _squared(left, right):
    return sum((a - b) ** 2 for a, b in zip(left, right))


def _nearest(point, centres):
    """(index, squared distance) of the closest centre."""
    best_index = 0
    best = _squared(point, centres[0])
    for index in range(1, len(centres)):
        distance = _squared(point, centres[index])
        if distance < best:
            best_index, best = index, distance
    return best_index, best


def _seed_centres(points, count, rng):
    """k-means++ seeding: spread the first guesses out across the cellar."""
    centres = [list(rng.choice(points))]
    while len(centres) < count:
        weights = [min(_squared(point, centre) for centre in centres) for point in points]
        total = sum(weights)
        if total <= 0:  # every batch already sits on a centre
            centres.append(list(rng.choice(points)))
            continue
        target = rng.random() * total
        running = 0.0
        chosen = points[-1]
        for point, weight in zip(points, weights):
            running += weight
            if running >= target:
                chosen = point
                break
        centres.append(list(chosen))
    return centres


def _recentre(points, labels, centres, count):
    """Move each centre to the middle of the batches that chose it."""
    width = len(points[0])
    sums = [[0.0] * width for _ in range(count)]
    sizes = [0] * count
    for point, label in zip(points, labels):
        sizes[label] += 1
        running = sums[label]
        for dimension, value in enumerate(point):
            running[dimension] += value

    moved = []
    for index in range(count):
        if sizes[index]:
            moved.append([value / sizes[index] for value in sums[index]])
        else:
            moved.append(list(centres[index]))  # nobody chose it; leave it put
    return moved


def _kmeans(points, count):
    """Sort points into `count` groups.  Returns (label per point, centres)."""
    rng = random.Random(SEED)
    best = None
    for _restart in range(KMEANS_RESTARTS):
        centres = _seed_centres(points, count, rng)
        labels = [-1] * len(points)
        for _round in range(KMEANS_ITERATIONS):
            settled = True
            for index, point in enumerate(points):
                label = _nearest(point, centres)[0]
                if label != labels[index]:
                    labels[index] = label
                    settled = False
            centres = _recentre(points, labels, centres, count)
            if settled:
                break
        tightness = sum(_squared(point, centres[label]) for point, label in zip(points, labels))
        if best is None or tightness < best[0]:
            best = (tightness, list(labels), centres)
    return best[1], best[2]


def _separation(points, labels, centres):
    """How cleanly a set of kinds separates: near 1.0 is tidy, near 0.0 is mush.

    For every batch, compare the distance to its own kind's centre with the
    distance to the next nearest kind's centre.  Averaged over the cellar,
    this lets two different numbers of kinds be judged against each other.
    """
    if len(centres) < 2 or not points:
        return -1.0
    total = 0.0
    for point, label in zip(points, labels):
        own = math.sqrt(_squared(point, centres[label]))
        other = min(
            math.sqrt(_squared(point, centre))
            for index, centre in enumerate(centres)
            if index != label
        )
        widest = max(own, other)
        total += 0.0 if widest <= 0 else (other - own) / widest
    return total / len(points)


def _expected(rows, target):
    """The score to expect from a group: the median of the scores it has earned.

    A median rather than an average, so one freakish bottle does not reset
    the winemaker's expectations for a whole kind of wine.  None when no
    batch of this kind has faced the panel yet.
    """
    if target is None:
        return None
    scores = [value for value in (step1._number(row, target) for row in rows) if value is not None]
    return step1._median(scores)


def kinds(train):
    """A few kinds of batch, each paired with the score to expect from it.

    Batches are grouped by their lab sheets -- the shape of the wine, not the
    verdict on it.  The panel's score is deliberately left out of the
    grouping: if it helped decide the kinds, then reporting a score per kind
    would only be repeating the question back.

    How many kinds there are is not fixed in advance.  Every count from 2 to
    5 is tried and the one that separates the cellar most cleanly wins, with
    ties going to the simpler answer.

    Returns [(rows of that kind, expected score), ...], most promising kind
    first; kinds whose batches have never been judged come last.
    """
    header, data = step1._split(train)
    if not data:
        return []

    target = _quality_column(header)
    scales = _scales(data, _sheet_columns(header, target))
    if not scales or len(data) < MIN_KINDS:
        return [(list(data), _expected(data, target))]

    points = [_filled(_vector(row, scales)) for row in data]
    distinct = len({tuple(point) for point in points})
    if distinct < MIN_KINDS:
        return [(list(data), _expected(data, target))]

    best = None
    for count in range(MIN_KINDS, min(MAX_KINDS, distinct) + 1):
        labels, centres = _kmeans(points, count)
        cleanliness = _separation(points, labels, centres)
        if best is None or cleanliness > best[0]:
            best = (cleanliness, labels, count)
    _cleanliness, labels, count = best

    grouped = [[] for _ in range(count)]
    for row, label in zip(data, labels):
        grouped[label].append(row)

    found = [(group, _expected(group, target)) for group in grouped if group]
    found.sort(key=lambda pair: (pair[1] is None, -(pair[1] or 0.0)))
    return found


def _check(rows):
    """An honest look at guess(): hold half the cellar back and score ourselves."""
    header, data = step1._split(rows)
    target = _quality_column(header)
    if target is None or len(data) < 4:
        return None

    learn = [rows[0]] + data[0::2]
    tested = [row for row in data[1::2] if step1._number(row, target) is not None]
    if not tested or len(learn) < 2:
        return None

    typical = step1._median([v for v in (step1._number(r, target) for r in data[0::2]) if v is not None])
    ours = sum(abs(guess(learn, row) - step1._number(row, target)) for row in tested) / len(tested)
    plain = sum(abs(typical - step1._number(row, target)) for row in tested) / len(tested)
    return ours, plain, len(tested)


def _report(rows):
    header, data = step1._split(rows)
    print("%d past batches, %d columns" % (len(data), len(header)))

    scored = _check(rows)
    if scored:
        ours, plain, tested = scored
        print("\nGuessing the panel's score (%d batches held back):" % tested)
        print("  average miss: %.3f points" % ours)
        print("  always saying 'typical': %.3f points" % plain)

    found = kinds(rows)
    print("\n%d kinds of batch in the cellar:" % len(found))
    target = _quality_column(header)
    for number, (group, expected) in enumerate(found, start=1):
        shown = "not yet judged" if expected is None else "expect %.2f" % expected
        print("  kind %d: %d batches, %s" % (number, len(group), shown))
        typical = step1.summary([header] + group)
        for index, name in enumerate(header):
            if index == target:
                continue
            value = typical[name]
            print("      %-18s %s" % (name, "n/a" if value is None else "%.4g" % value))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python step2.py <cellar.csv>", file=sys.stderr)
        raise SystemExit(2)
    _report(step1.load_csv(sys.argv[1]))
