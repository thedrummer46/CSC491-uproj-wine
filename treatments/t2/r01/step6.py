"""Winemaker's AI assistant -- step 6.

Builds on :mod:`step1` (reading a batch table, telling a goal column from an
ordinary lab reading, and the plain statistics -- median, spread), :mod:`step2`
(finding the column the panel fills in), :mod:`step3` (saying when a difference
is real and when it is only tasting noise) and :mod:`step4` (the score at which
a batch is worth the panel's time).  Nothing written there is written again
here.

Steps 2 to 5 answer *what will this batch score* and *which batch should we
keep*.  This one answers *why*, which is the question a winemaker can actually
act on -- you cannot change a batch's score, but you can change the cellar
practice behind a reading:

``blame(train, k=3)``
    "Which two or three readings matter most for whether a batch scores well?"

``dragging(train)``
    "Our low-scoring batches: which lab readings are dragging them down?"

Only lab readings are ever named.  The goal columns (``Quality+``, ``Alcohol-``)
are left out on purpose: saying "your quality is low because your quality is
low" is no help, and the alcohol is a goal in its own right, not a lever.

How a reading earns its place is deliberately robust, because a cellar's
numbers are lumpy and whole-number panel scores are nothing like a bell curve:

* **separation** -- line the batches up by one reading and ask how often a
  good batch sits above a poor one.  All the way to one end scores 1, no
  pattern at all scores 0.  This is a rank comparison, so one freak batch
  cannot swing it.
* **shape** -- cut the same line-up into quarters and see how far apart the
  quarters' average scores are.  This catches a reading that is only good in
  the middle, which a straight "higher is better" test would miss entirely.

A reading's strength is the stronger of the two.  Then, in the spirit of step
3, that strength has to clear a noise floor earned the honest way: the panel's
scores are shuffled against the same readings a number of times, and whatever
strength the shuffling throws up by luck alone is the bar the real reading has
to beat.
"""

import math
import random

import step1
import step2
import step3
import step4
from step1 import classify, mean, median, number, spread
from step2 import target_column

# A batch at or above this score is a good one; below it, a poor one.
GOOD_SCORE = step4.PANEL_THRESHOLD

# Quarters: enough to see a reading that is only good in the middle, few
# enough that each quarter still holds a believable number of batches.
BINS = 4

# How many times the scores are shuffled to see what luck alone can manage.
SHUFFLES = 25
NOISE_SHARE = 0.95

# Fixed, so the same cellar always gets the same answer.
SEED = step4.SEED

# Below this much separation, "too high" and "too low" are not worth saying.
MIN_SEPARATION = 0.10

# Two readings this alike are the same story told twice; only one is named.
TWINS = 0.90


# ---------------------------------------------------------------- the sheet


def _sheet(train):
    """Return ``(header, data, target, readings)`` for a batch table.

    ``target`` is the goal column the panel fills in (``Quality+``), found by
    step 2; ``readings`` are the ordinary lab columns, found by step 1.  A
    table with no header, no rows or no goal column yields empty parts, which
    is how the answers below come back empty rather than invented.
    """
    header, data = step3._table(train)
    if not header or not data:
        return [], [], None, []
    _goals, readings = classify(header)
    target = target_column(header)
    if target is None or not readings:
        return header, data, None, []
    return header, data, target, readings


def _pairs(data, index, target_index):
    """Every batch as ``(reading, score)``, skipping the ones missing either."""
    out = []
    for row in data:
        value = number(row[index]) if index < len(row) else None
        score = number(row[target_index]) if target_index < len(row) else None
        if value is None or score is None:
            continue
        out.append((value, score))
    return out


def _is_good(score, direction, threshold):
    """True when the panel would call this a good batch."""
    if direction < 0:
        return score <= threshold
    return score >= threshold


# ---------------------------------------------------------------- separation


def _average_ranks(values):
    """Each value's place in the line-up, ties sharing their places evenly."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        share = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = share
        i = j + 1
    return ranks


def _separation(ranks, flags):
    """How cleanly one reading tells good batches from poor ones, in -1..1.

    ``+1`` means every good batch reads higher than every poor one, ``-1`` the
    other way about, ``0`` that the two are shuffled through each other and the
    reading says nothing.  No distribution is assumed -- only the order of the
    batches is used.
    """
    good = sum(1 for flag in flags if flag)
    poor = len(flags) - good
    if not good or not poor:
        return 0.0
    total = sum(rank for rank, flag in zip(ranks, flags) if flag)
    beats = total - good * (good + 1) / 2.0
    return 2.0 * beats / float(good * poor) - 1.0


def _shape(order, scores, swing, bins=BINS):
    """How far apart the quarters of a line-up score, as a share of the swing.

    Zero when every quarter of the cellar scores alike -- the reading makes no
    difference anywhere.  Larger when some stretch of the reading scores
    plainly better or worse, whether or not it does so all in one direction.
    """
    n = len(order)
    if swing <= 0.0 or n < bins * 2:
        return 0.0
    middles = []
    for b in range(bins):
        low = (b * n) // bins
        high = ((b + 1) * n) // bins
        chunk = [scores[i] for i in order[low:high]]
        if chunk:
            middles.append(mean(chunk))
    if len(middles) < 2:
        return 0.0
    return min(1.0, spread(middles) / swing)


def _noise(ranks, order, scores, direction, threshold, swing, dice,
           shuffles=SHUFFLES, share=NOISE_SHARE):
    """The strength luck alone can manage on this many batches.

    The panel's scores are dealt out again at random against the very same
    readings.  Whatever separation and shape that turns up is pattern that was
    never there, and a real reading has to beat it before it is worth naming.
    """
    bag = list(scores)
    draws = []
    for _ in range(shuffles):
        dice.shuffle(bag)
        flags = [_is_good(s, direction, threshold) for s in bag]
        draws.append(max(abs(_separation(ranks, flags)), _shape(order, bag, swing)))
    if not draws:
        return 0.0
    draws.sort()
    spot = int(math.ceil(share * len(draws))) - 1
    return draws[max(0, min(len(draws) - 1, spot))]


# ---------------------------------------------------------------- one reading


def _weigh(name, index, pairs, direction, threshold, dice):
    """The whole story of one lab reading, as a dict.

    Beside its strength and the noise floor that strength had to clear, the
    story carries the plain numbers a winemaker can check by eye: what this
    reading usually is in good batches, what it usually is in poor ones, and
    how far apart those two are in usual swings of the reading.
    """
    values = [value for value, _score in pairs]
    scores = [score for _value, score in pairs]
    good = [value for value, score in pairs if _is_good(score, direction, threshold)]
    poor = [value for value, score in pairs if not _is_good(score, direction, threshold)]

    story = {
        "name": name,
        "index": index,
        "batches": len(pairs),
        "separation": 0.0,
        "shape": 0.0,
        "strength": 0.0,
        "noise": 0.0,
        "clears": False,
        "good": median(good),
        "poor": median(poor),
        "gap": 0.0,
        "swing": spread(values),
        "side": None,
    }
    if len(pairs) < 4 or not good or not poor:
        return story

    swing = spread(scores)
    ranks = _average_ranks(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    flags = [_is_good(score, direction, threshold) for score in scores]

    separation = _separation(ranks, flags)
    shape = _shape(order, scores, swing)
    strength = max(abs(separation), shape)
    noise = _noise(ranks, order, scores, direction, threshold, swing, dice)

    story["separation"] = separation
    story["shape"] = shape
    story["strength"] = strength
    story["noise"] = noise
    story["clears"] = strength > noise

    if story["swing"] > 0.0 and story["good"] is not None and story["poor"] is not None:
        story["gap"] = (story["poor"] - story["good"]) / story["swing"]

    # Which way the poor batches sit: a separation above zero means the good
    # batches read higher, so the poor ones are running low, and the reverse.
    if separation > 0:
        story["side"] = "low"
    elif separation < 0:
        story["side"] = "high"
    return story


# ---------------------------------------------------------------- all readings


_WEIGHED = {}


def influence(train, threshold=GOOD_SCORE):
    """Every lab reading weighed against the panel's score, strongest first.

    A list of the dicts built by :func:`_weigh` -- the long form behind both
    answers below, for when the winemaker asks to see the numbers.  Goal
    columns never appear.  The work is remembered per table, so asking both
    questions about one cellar costs no more than asking either.
    """
    key = (step3._fingerprint(train), float(threshold))
    if key in _WEIGHED:
        return _WEIGHED[key]

    _header, data, target, readings = _sheet(train)
    if target is None:
        _WEIGHED[key] = []
        return []

    target_index, _target_name, direction = target
    dice = random.Random(SEED)

    out = []
    for index, name in readings:
        pairs = _pairs(data, index, target_index)
        if not pairs:
            continue
        out.append(_weigh(name, index, pairs, direction, float(threshold), dice))

    out.sort(key=lambda item: (-item["strength"], item["name"]))
    _WEIGHED[key] = out
    return out


def _alike(data, left, right):
    """How closely two lab readings move together, in -1..1.

    Used to keep the shortlist from naming the same story twice: fixed acidity
    and density rise and fall as one in most cellars, and a winemaker handed
    both has been told one thing, not two.
    """
    xs = []
    ys = []
    for row in data:
        x = number(row[left]) if left < len(row) else None
        y = number(row[right]) if right < len(row) else None
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < 3:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    top = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    down = math.sqrt(sum((x - mx) ** 2 for x in xs)) * math.sqrt(sum((y - my) ** 2 for y in ys))
    if down <= 0.0:
        return 0.0
    return top / down


# ---------------------------------------------------------------- answers


def blame(train, k=3):
    """The ``k`` lab readings that matter most for ``Quality+``, strongest first.

    ``train`` is a table in the step 1 shape (header row first).  Only lab
    readings are returned -- never ``Quality+`` itself and never the other goal
    ``Alcohol-``, because a goal is something to aim at, not a lever to pull.

    Each reading is judged on how cleanly it tells a good batch from a poor one
    and on how far apart the quarters of its line-up score, whichever speaks
    louder; and two readings that rise and fall together count as one story, so
    the shortlist points at ``k`` different things to look at rather than ``k``
    views of the same thing.

    Returns fewer than ``k`` names only when the cellar has fewer readings than
    that, or nothing usable at all.  Which of the names clear the noise floor,
    and by how much, is in :func:`influence` and :func:`report`.
    """
    _header, data, target, _readings = _sheet(train)
    ranked = influence(train)
    if target is None or not ranked:
        return []

    k = max(1, int(k))
    picked = []
    twins = []
    for item in ranked:
        if len(picked) >= k:
            break
        if any(abs(_alike(data, item["index"], kept["index"])) >= TWINS for kept in picked):
            twins.append(item)
            continue
        picked.append(item)

    # If setting twins aside left the shortlist short, fill it back out: the
    # winemaker asked for k readings and gets k.
    for item in twins:
        if len(picked) >= k:
            break
        picked.append(item)

    return [item["name"] for item in picked]


def dragging(train, below=GOOD_SCORE):
    """The readings running too high or too low in the poor batches, worst first.

    ``below`` is the score that divides a poor batch from a good one -- 6 by
    default, the same line step 4 uses to decide a batch is worth the panel's
    time.  Each pair is ``(column name, 'high')`` or ``(column name, 'low')``:
    ``'high'`` when the poor batches read above the good ones on that reading,
    ``'low'`` when they read below it.

    A reading is only named when the difference survives two doubts, as in step
    3: it has to beat what shuffling the scores throws up by luck, and the two
    groups have to be properly separated rather than merely nudged apart.  So
    an empty list is a real answer -- it means nothing on the lab sheet tells
    the poor batches apart from the good ones, and the trouble is elsewhere.

    Readings that matter but not in one direction -- good in the middle, bad at
    both ends -- have no honest ``'high'`` or ``'low'`` to report, so they are
    left out here and show up under :func:`blame` instead.
    """
    ranked = influence(train, threshold=below)

    out = []
    for item in ranked:
        if not item["clears"] or item["side"] is None:
            continue
        if abs(item["separation"]) < MIN_SEPARATION:
            continue
        out.append((abs(item["separation"]), abs(item["gap"]), item["name"], item["side"]))

    out.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [(name, side) for _sep, _gap, name, side in out]


# ---------------------------------------------------------------- the why


def report(train, k=3, below=GOOD_SCORE):
    """The full story behind both answers, for when the winemaker asks why.

    Returns the shortlist, the dragging readings, the line that divides a poor
    batch from a good one, how many batches each group holds, and a sentence
    per named reading giving its usual value in good batches, its usual value
    in poor ones, and whether its strength cleared the noise floor.
    """
    header, data, target, _readings = _sheet(train)
    story = {
        "readings": [],
        "shortlist": [],
        "dragging": [],
        "threshold": float(below),
        "good": 0,
        "poor": 0,
        "why": "no goal column to explain",
    }
    if target is None:
        return story

    target_index, target_name, direction = target
    scores = [
        number(row[target_index])
        for row in data
        if target_index < len(row) and number(row[target_index]) is not None
    ]
    story["good"] = sum(1 for s in scores if _is_good(s, direction, float(below)))
    story["poor"] = len(scores) - story["good"]
    story["judging"] = target_name

    weighed = influence(train, threshold=below)
    story["readings"] = weighed
    story["shortlist"] = blame(train, k)
    story["dragging"] = dragging(train, below)

    lines = []
    for item in weighed:
        if item["name"] not in story["shortlist"] and not item["clears"]:
            continue
        good = "n/a" if item["good"] is None else "%g" % item["good"]
        poor = "n/a" if item["poor"] is None else "%g" % item["poor"]
        lines.append(
            "%s: good batches run around %s, poor ones around %s "
            "(strength %.2f against a noise floor of %.2f -- %s)"
            % (
                item["name"],
                good,
                poor,
                item["strength"],
                item["noise"],
                "worth saying" if item["clears"] else "inside the noise",
            )
        )
    story["notes"] = lines
    story["why"] = (
        "%d good batches at %g or better, %d poor ones below it, weighed on %d lab readings"
        % (story["good"], float(below), story["poor"], len(weighed))
    )
    return story


# ---------------------------------------------------------------- demo


def _show(path):
    rows = step1.load_rows(path)
    header, data = step1.split(rows)
    print("batches on file: %d" % len(data))

    target = target_column(header)
    print("judging: %s" % (target[1] if target else "nothing -- no goal column"))
    if not data or target is None:
        return

    story = report(rows)
    print("a poor batch is one scoring under %g: %d of them, against %d good ones"
          % (story["threshold"], story["poor"], story["good"]))

    print("\nthe readings that matter most for whether a batch scores well:")
    for place, name in enumerate(story["shortlist"], 1):
        item = next(i for i in story["readings"] if i["name"] == name)
        print("  %d. %-16s strength %.2f  (noise floor %.2f)%s"
              % (place, name, item["strength"], item["noise"],
                 "" if item["clears"] else "  -- inside the noise, treat gently"))

    print("\nin the poor batches, these readings are dragging them down:")
    if not story["dragging"]:
        print("  nothing on the lab sheet separates them -- the trouble is elsewhere")
    for name, side in story["dragging"]:
        item = next(i for i in story["readings"] if i["name"] == name)
        print("  %-16s too %-4s  (poor %g vs good %g, %.2f swings apart)"
              % (name, side, item["poor"], item["good"], abs(item["gap"])))

    print("\nthe long form, reading by reading:")
    print("  %-16s %9s %9s %7s %7s" % ("reading", "separate", "shape", "strong", "noise"))
    for item in story["readings"]:
        print("  %-16s %9.2f %9.2f %7.2f %7.2f"
              % (item["name"], item["separation"], item["shape"], item["strength"], item["noise"]))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python step6.py <batches.csv>")
        raise SystemExit(2)
    _show(sys.argv[1])
