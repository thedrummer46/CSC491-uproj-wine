"""Step 6 of the winemaker's assistant.

The records arrive in the familiar shape -- row 0 is the header, every later
row is one batch of numbers, "Quality+" is the score the tasting panel
awards, "Alcohol-" a goal to keep low, and everything else a plain lab
reading.  The readers from step 1 and the scaling from step 2 are reused, so
the numbers here are the same numbers the earlier steps work from.

Two requests are answered, both about explaining rather than predicting:

    blame(train, k=3) -> the readings that matter most for Quality+
    dragging(train)   -> the readings that are out of line in poor batches

Only lab readings are ever named.  The goal columns are deliberately kept out
of the reckoning on both counts: "Quality+ explains Quality+" is no use to
anyone, and naming "Alcohol-" as the reason a batch scored badly would hand
back a goal the winemaker already has rather than a dial on the bench.  What
is wanted here is the readings a cellarman can actually act on.

The two questions are genuinely different, and it is worth saying how.
blame() asks which readings carry information about the score across the
whole cellar -- including the readings that mark out the *good* batches.
dragging() asks only about the poor batches, and names a reading only when it
sits out of line *and* that drift points the wrong way.  A reading can top
one list and be absent from the other.
"""

import math
import random
import sys

import step1
import step2
import step4

# A poor batch, in the winemaker's own terms.  The same bar step 4 uses to
# decide who gets a place at the tasting, kept in one place on purpose.
POOR_SCORE = step4.WORTH_THRESHOLD

# How the importance of a reading is measured: fit on most of the cellar,
# then see how much shuffling that one reading costs on the part held back.
FOLDS = 5
SHUFFLES = 3  # shuffles per reading per fold, averaged
SEED = 20240606  # fixed, so the same cellar always blames the same readings

# Below this many judged batches there is nothing to hold back, and the
# ranking falls back to the size of the fitted pull per reading.
MIN_FOR_FOLDS = 10

# A pinch of ridge, so two near-identical readings (density and sugar, say)
# cannot blow the fit up between them by cancelling out.
RIDGE = 0.01

# How far the poor batches have to drift on a reading before it is worth
# naming, measured in robust spreads of that reading.
DRAG_THRESHOLD = 0.2

# ...and how much that reading has to track the score at all.  A reading that
# drifts but has no bearing on taste is a coincidence, not a cause.
TRACK_THRESHOLD = 0.05

# Fewer good batches than this and there is no sensible "good" group to
# compare the poor ones against; the whole judged cellar stands in instead.
MIN_GROUP = 3


def _reading_columns(header):
    """The lab sheet: every column that is not a goal.

    This is what guarantees neither answer here can name a goal column.
    """
    goals = {index for index, _direction in step1._goal_columns(header)}
    return [index for index in range(len(header)) if index not in goals]


def _judged(train):
    """(header, the batches the panel has actually scored, index of that score)."""
    header, data = step1._split(train)
    target = step2._quality_column(header)
    if target is None:
        return header, [], None
    judged = [row for row in data if step1._number(row, target) is not None]
    return header, judged, target


def _bench(train):
    """The judged cellar as (header, scales, comparable readings, scores).

    Each reading is centred on its median and divided by a robust spread, as
    in step 2, so a column measured in hundreds cannot out-shout one measured
    in tenths when their importance is compared.  Readings that never vary
    are dropped: they explain nothing.  Gaps are closed up at the typical
    value, which amounts to letting a missing reading say nothing either way.
    """
    header, judged, target = _judged(train)
    if target is None or not judged:
        return header, [], [], []
    scales = step2._scales(judged, _reading_columns(header))
    if not scales:
        return header, [], [], []
    points = [step2._filled(step2._vector(row, scales)) for row in judged]
    scores = [step1._number(row, target) for row in judged]
    return header, scales, points, scores


def _solve(matrix, vector):
    """Gaussian elimination with partial pivoting; None when it will not solve."""
    size = len(vector)
    work = [list(row) + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(work[r][column]))
        if abs(work[pivot][column]) < 1e-12:
            return None
        work[column], work[pivot] = work[pivot], work[column]
        divisor = work[column][column]
        for row in range(column + 1, size):
            factor = work[row][column] / divisor
            if factor == 0.0:
                continue
            for cell in range(column, size + 1):
                work[row][cell] -= factor * work[column][cell]
    answer = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = work[row][size] - sum(
            work[row][cell] * answer[cell] for cell in range(row + 1, size)
        )
        answer[row] = total / work[row][row]
    return answer


def _fit(points, scores):
    """Fit score on the comparable readings: (pull per reading, means, centre).

    A straight line through every reading at once, which is the point: it
    asks what each reading adds *given the others*, so two readings that say
    the same thing cannot both claim full credit for it.  A reading's pull is
    in points of Quality+ per robust spread of that reading, which is what
    makes the pulls comparable with one another.
    """
    count = len(points)
    width = len(points[0]) if count else 0
    centre = sum(scores) / float(count) if count else 0.0
    if not width:
        return [], [], centre

    means = [sum(point[index] for point in points) / float(count) for index in range(width)]
    gram = [[0.0] * width for _ in range(width)]
    moment = [0.0] * width
    for point, score in zip(points, scores):
        shifted = [point[index] - means[index] for index in range(width)]
        offset = score - centre
        for index in range(width):
            value = shifted[index]
            moment[index] += value * offset
            row = gram[index]
            for other in range(index, width):
                row[other] += value * shifted[other]
    for index in range(width):
        for other in range(index):
            gram[index][other] = gram[other][index]
        gram[index][index] += RIDGE * count + 1e-9

    weights = _solve(gram, moment)
    if weights is None:
        return [0.0] * width, means, centre
    return weights, means, centre


def _predicted(weights, means, centre, point):
    return centre + sum(
        weight * (value - mean) for weight, value, mean in zip(weights, point, means)
    )


def _parts(count, folds, fold):
    """(held back, kept) row positions for one fold, taken at a steady stride."""
    held = [index for index in range(count) if index % folds == fold]
    kept = [index for index in range(count) if index % folds != fold]
    return held, kept


def _importances(points, scores):
    """Per reading: how many points of error shuffling it costs, cross-validated.

    Most of the cellar is fitted, the rest held back; then one reading's
    column is shuffled among the held-back batches and the error is measured
    again.  If the reading mattered, scrambling it hurts.  This is a fairer
    ranking than reading the fitted pulls straight off, because it asks what
    each reading is worth on batches the fit has never seen.

    On a cellar too small to hold anything back, the size of the fitted pull
    stands in -- a cruder measure in different units, but a monotone one.
    """
    count = len(points)
    width = len(points[0]) if count else 0
    if not width:
        return []
    if count < MIN_FOR_FOLDS:
        weights, _means, _centre = _fit(points, scores)
        return [abs(weight) for weight in weights]

    rng = random.Random(SEED)
    folds = min(FOLDS, count)
    totals = [0.0] * width
    used = 0
    for fold in range(folds):
        held, kept = _parts(count, folds, fold)
        if len(held) < 2 or len(kept) <= width:
            continue
        weights, means, centre = _fit(
            [points[index] for index in kept], [scores[index] for index in kept]
        )
        held_points = [points[index] for index in held]
        held_scores = [scores[index] for index in held]
        base = [_predicted(weights, means, centre, point) for point in held_points]
        plain = sum(
            abs(guess - score) for guess, score in zip(base, held_scores)
        ) / float(len(held))

        for column in range(width):
            scrambled = 0.0
            for _repeat in range(SHUFFLES):
                swapped = [point[column] for point in held_points]
                rng.shuffle(swapped)
                total = 0.0
                for position, score in enumerate(held_scores):
                    # only this one reading moves, so the rest of the
                    # prediction can be reused as it stands
                    moved = base[position] + weights[column] * (
                        swapped[position] - held_points[position][column]
                    )
                    total += abs(moved - score)
                scrambled += total / float(len(held))
            totals[column] += scrambled / SHUFFLES - plain
        used += 1

    if not used:
        weights, _means, _centre = _fit(points, scores)
        return [abs(weight) for weight in weights]
    return [total / float(used) for total in totals]


def strengths(train):
    """Every lab reading, paired with how much it matters and which way it pulls.

    Returns [(column name, strength, pull), ...], strongest first.  `pull` is
    the fitted effect on Quality+ per robust spread of that reading, so a
    positive pull means higher readings go with better scores.  Ties are
    broken by the order the columns appear on the sheet, so the answer is
    stable from one run to the next.
    """
    header, scales, points, scores = _bench(train)
    if not scales:
        return []

    marks = _importances(points, scores)
    weights, _means, _centre = _fit(points, scores)

    found = []
    for position, (index, _centre_value, _spread) in enumerate(scales):
        strength = marks[position] if position < len(marks) else 0.0
        pull = weights[position] if position < len(weights) else 0.0
        found.append((header[index], strength, pull, index))
    found.sort(key=lambda item: (-item[1], item[3]))
    return [(name, strength, pull) for name, strength, pull, _index in found]


def blame(train, k=3):
    """The `k` lab readings that matter most for whether a batch scores well.

    Answers "which two or three readings matter most?".  Most important
    first; never a goal column, since Quality+ and Alcohol- are not readings
    anyone can turn a dial on.

    Importance is judged by holding part of the cellar back, scrambling one
    reading at a time, and seeing how much worse the score becomes to
    predict.  Two cautions worth passing on with the answer.  First, the
    readings are judged together, so where two of them say much the same
    thing (density and residual sugar, for instance) the credit is split and
    both can look weaker than either really is.  Second, this is association,
    not proof: a reading at the top of this list travels with good wine, but
    changing it on the bench is a separate experiment.

    Fewer names come back than asked for when the cellar has fewer readings
    that vary; [] when no batch has been judged yet, or the sheet carries no
    score column at all.
    """
    try:
        wanted = int(k)
    except (TypeError, ValueError):
        wanted = 3
    if wanted <= 0:
        return []
    return [name for name, _strength, _pull in strengths(train)[:wanted]]


def _ranks(values):
    """Positions in sorted order, with ties sharing the average position."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = shared
        position = end + 1
    return ranks


def _rank_correlation(left, right):
    """Does one column climb with the other: +1 always, -1 never, 0 no telling.

    Ranks rather than raw numbers, so the answer holds for any steadily
    rising relationship and is not thrown off by a few freak readings.
    """
    count = len(left)
    if count < 3:
        return 0.0
    first, second = _ranks(left), _ranks(right)
    mean_first = sum(first) / float(count)
    mean_second = sum(second) / float(count)
    together = sum((a - mean_first) * (b - mean_second) for a, b in zip(first, second))
    spread_first = sum((a - mean_first) ** 2 for a in first)
    spread_second = sum((b - mean_second) ** 2 for b in second)
    if spread_first <= 0 or spread_second <= 0:
        return 0.0
    return together / math.sqrt(spread_first * spread_second)


def _spread_of(values):
    """A robust spread for one column, falling back to the plain one."""
    centre = step1._median(values)
    spread = 1.4826 * step1._median([abs(value - centre) for value in values])
    if spread <= 0:
        spread = step2._spread_fallback(values, centre)
    return spread


def _drags(train):
    """The full working behind dragging(): (name, way, severity, drift, track)."""
    header, judged, target = _judged(train)
    if target is None or len(judged) < 2:
        return []

    poor = [row for row in judged if step1._number(row, target) < POOR_SCORE]
    good = [row for row in judged if step1._number(row, target) >= POOR_SCORE]
    if not poor:
        return []  # no poor batches on the books: nothing is dragging anything
    reference = good if len(good) >= MIN_GROUP else judged

    found = []
    for index in _reading_columns(header):
        values = step1._column(judged, index)
        if len(values) < 2:
            continue
        spread = _spread_of(values)
        if spread <= 0:
            continue  # a reading that never varies cannot be out of line

        poor_values = step1._column(poor, index)
        reference_values = step1._column(reference, index)
        if not poor_values or not reference_values:
            continue

        # How far the poor batches sit from the good ones on this reading,
        # in spreads of the reading itself.  Medians on both sides, so one
        # ruined barrel does not define what a poor batch looks like.
        drift = (step1._median(poor_values) - step1._median(reference_values)) / spread
        if abs(drift) < DRAG_THRESHOLD:
            continue

        pairs = [
            (step1._number(row, index), step1._number(row, target)) for row in judged
        ]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        track = _rank_correlation([a for a, _b in pairs], [b for _a, b in pairs])
        if abs(track) < TRACK_THRESHOLD:
            continue  # drifts, but has no bearing on taste: a coincidence
        if drift * track >= 0:
            # The poor batches drift the way that ought to *help* on this
            # reading.  Whatever is wrong with them, it is not this.
            continue

        way = "high" if drift > 0 else "low"
        found.append((header[index], way, abs(drift) * abs(track), drift, track))

    order = {name: position for position, name in enumerate(header)}
    found.sort(key=lambda item: (-item[2], order.get(item[0], 0)))
    return found


def dragging(train):
    """The readings that are out of line in the poor batches, worst first.

    Answers "our low-scoring batches: which lab readings are dragging them
    down?".  A poor batch is one the panel scored below %g.  Each returned
    pair is (column name, 'high' or 'low'), where 'high' means that reading
    runs too high in the poor batches and 'low' that it runs too low.

    A reading earns its place only on three counts at once: the poor batches
    sit a real distance from the good ones on it, that reading genuinely
    tracks the score across the cellar, and the drift runs the wrong way.
    The third test is what keeps the list honest -- poor batches differ from
    good ones in all sorts of harmless ways, and only the differences that
    point towards a worse score are worth a cellarman's morning.

    Returns [] when no batch scored below the bar, when nothing has been
    judged, or when no reading is far enough out of line to name.  As with
    blame(), this is the company a poor score keeps, not proof of cause.
    """ % POOR_SCORE
    return [(name, way) for name, way, _severity, _drift, _track in _drags(train)]


def _cv_mae(points, scores):
    """Average miss in points, holding each fold back in turn.  None if too few."""
    count = len(points)
    width = len(points[0]) if count else 0
    if count < MIN_FOR_FOLDS:
        return None
    folds = min(FOLDS, count)
    total = 0.0
    tested = 0
    for fold in range(folds):
        held, kept = _parts(count, folds, fold)
        if not held or len(kept) <= width + 1:
            continue
        weights, means, centre = _fit(
            [points[index] for index in kept], [scores[index] for index in kept]
        )
        for index in held:
            guess = _predicted(weights, means, centre, points[index])
            total += abs(guess - scores[index])
            tested += 1
    return None if not tested else total / float(tested)


def _typical_mae(scores):
    """The lazy answer to beat: always say the typical score."""
    count = len(scores)
    if count < MIN_FOR_FOLDS:
        return None
    folds = min(FOLDS, count)
    total = 0.0
    tested = 0
    for fold in range(folds):
        held, kept = _parts(count, folds, fold)
        if not held or not kept:
            continue
        typical = step1._median([scores[index] for index in kept])
        for index in held:
            total += abs(typical - scores[index])
            tested += 1
    return None if not tested else total / float(tested)


def _check(train, names):
    """Is the short list enough?  Fit on those readings alone and compare."""
    header, scales, points, scores = _bench(train)
    if not scales:
        return None
    wanted = set(names)
    chosen = [
        position
        for position, (index, _centre, _spread) in enumerate(scales)
        if header[index] in wanted
    ]
    if not chosen:
        return None
    few = [[point[position] for position in chosen] for point in points]
    return _cv_mae(points, scores), _cv_mae(few, scores), _typical_mae(scores)


def _report(train, wanted=3):
    header, judged, target = _judged(train)
    _header, data = step1._split(train)
    print("%d past batches, %d judged, %d columns" % (len(data), len(judged), len(header)))
    if target is None:
        print("\nno panel score on record: nothing to explain")
        return
    if not judged:
        print("\nno batch has been judged yet: nothing to explain")
        return

    ranked = strengths(train)
    if not ranked:
        print("\nno lab reading on the sheet varies: nothing to explain")
        return

    width = max(len(name) for name, _s, _p in ranked)
    print("\nWhat the readings are worth for Quality+ (shuffling cost, points):")
    for name, strength, pull in ranked:
        print("  %-*s  %+.4f   pull %+.3f" % (width, name, strength, pull))

    short = blame(train, wanted)
    print("\nThe %d readings that matter most: %s" % (len(short), ", ".join(short)))

    checked = _check(train, short)
    if checked:
        every, few, plain = checked
        if every is not None and few is not None:
            print("  average miss on batches held back:")
            print("    all %d readings:   %.3f points" % (len(ranked), every))
            print("    these %d alone:     %.3f points" % (len(short), few))
            if plain is not None:
                print("    always 'typical':  %.3f points" % plain)

    poor = [row for row in judged if step1._number(row, target) < POOR_SCORE]
    print("\nPoor batches (Quality+ below %g): %d of %d judged"
          % (POOR_SCORE, len(poor), len(judged)))
    drags = _drags(train)
    if not drags:
        print("  nothing far enough out of line to name")
        return
    print("  what is dragging them down, worst first:")
    for name, way, severity, drift, track in drags:
        print("    %-*s  runs too %-4s (%+.2f spreads, tracks the score %+.2f,"
              " severity %.3f)" % (width, name, way, drift, track, severity))


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("usage: python step6.py <cellar.csv> [how many readings]", file=sys.stderr)
        raise SystemExit(2)
    _report(step1.load_csv(sys.argv[1]),
            int(sys.argv[2]) if len(sys.argv) == 3 else 3)
