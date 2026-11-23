"""Winemaker's AI assistant -- step 3.

Builds on :mod:`step1` (reading a batch table, telling a goal column from a
lab reading, describing a normal batch) and :mod:`step2` (the cellar of past
lab sheets, and the guess it makes for a new one).  Nothing already written
there is written again here.

Two last questions are answered:

``certify(train, a, b)``
    "When you say batch A will beat batch B, tell me whether that's a real
    difference or just tasting noise."

``accuracy(train, test)``
    "How often are your score estimates right on batches you've never seen
    before?"

The honesty in both answers comes from the same place: a guess is an average
over a handful of look-alike past batches, and look-alikes disagree.  Step 3
measures that disagreement instead of hiding it.  Two guesses closer together
than the distance a guess is usually wrong by are reported as ``'same'``, and
accuracy is only ever measured on batches the cellar was not allowed to see.
"""

import math
import random

import step1
import step2
from step1 import mean, median, number, split, spread
from step2 import Cellar, target_column

# How sure we insist on being before naming a winner.
CONFIDENCE = 0.95

# How many times the tasting is replayed from the look-alike batches.
RESAMPLES = 500

# Fixed, so the same two batches always get the same verdict.
SEED = 20260101

# Held-back slices used when reporting reliability over a whole cellar.
FOLDS = 5


# ---------------------------------------------------------------- tables


def _table(rows):
    """Return ``(header, data)`` for a table that may or may not name its columns.

    The winemaker hands over test batches both ways: sometimes with the header
    row copied across, sometimes as bare lab sheets.  A first row carrying any
    text is taken to be a header.
    """
    if not rows:
        return [], []
    first = rows[0]
    if any(isinstance(cell, str) for cell in first):
        return split(rows)
    return [], [row for row in rows]


def _fingerprint(rows):
    """A cheap, hashable stand-in for a table, used to remember work about it."""
    header, data = _table(rows)
    edges = ()
    if data:
        edges = (tuple(data[0]), tuple(data[-1]))
    return (tuple(str(cell) for cell in header), len(data), edges)


def _truth_index(cellar, header):
    """Where the panel's score sits in a test row.

    The test table's own header is believed when it has one; otherwise the
    layout is assumed to match the train table, which is how the batches come
    off the same lab sheet.
    """
    if header:
        target = target_column(header)
        if target is not None:
            return target[0]
    return cellar.target_index


# ---------------------------------------------------------------- rounding


def _grain(values):
    """The step the panel actually scores in: whole points, or ``None`` if finer.

    Wine panels hand back whole numbers, so "right" means "rounds to the score
    they gave".  If a cellar is kept in fractions, no rounding is imposed.
    """
    if not values:
        return 1.0
    for value in values:
        if abs(value - round(value)) > 1e-9:
            return None
    return 1.0


def _round_to(value, grain):
    """Round the way a person rounds -- halves go up -- onto the panel's grain."""
    if grain is None or grain <= 0:
        return float(value)
    return math.floor(float(value) / grain + 0.5) * grain


def _same_score(x, y, grain):
    return abs(_round_to(x, grain) - _round_to(y, grain)) < 1e-9


# ---------------------------------------------------------------- noise


def _neighbourhood(cellar, row):
    """The scores of the past batches this one looks like, nearest first.

    These are the very batches :meth:`step2.Cellar.guess` averages, so the
    spread among them is the spread behind the guess.
    """
    if not cellar.scored:
        return []
    count = int(round(math.sqrt(len(cellar.scored))))
    count = max(1, min(step2.NEIGHBOURS_CAP, count, len(cellar.scored)))
    return [cellar.scores[i] for _gap, i in cellar.neighbours(row, count)]


_FLOOR = {}


def noise_floor(train, cellar=None):
    """How far apart two guesses must be before the difference is worth saying.

    Taken from the cellar itself, not from a rule of thumb: batches are held
    back, guessed at, and the average miss is measured (that is step 2's own
    ``check``).  A gap smaller than the distance a guess is usually wrong by
    cannot be told from tasting noise.  The floor is never allowed below half a
    panel point, because a smaller gap would not change the score they write
    down anyway.
    """
    key = _fingerprint(train)
    if key in _FLOOR:
        return _FLOOR[key]

    if cellar is None:
        cellar = Cellar(train)

    report = step2.check(train)
    miss = report.get("error")
    if not miss:
        miss = cellar.swing or 0.0

    grain = _grain([s for s in cellar.scores if s is not None])
    floor = float(miss)
    if grain:
        floor = max(floor, grain / 2.0)

    _FLOOR[key] = floor
    return floor


def _beats(a_scores, b_scores, resamples=RESAMPLES, seed=SEED):
    """Share of replayed tastings in which A's look-alikes outscore B's.

    The look-alike batches of each wine are drawn again, with replacement, as
    though the panel had happened to taste a slightly different set of them.
    Ties are split.  No distribution is assumed -- a cellar of whole-number
    scores is nothing like a bell curve.
    """
    if not a_scores or not b_scores:
        return 0.5
    dice = random.Random(seed)
    n = len(a_scores)
    m = len(b_scores)
    wins = 0.0
    for _ in range(resamples):
        x = mean([a_scores[dice.randrange(n)] for _ in range(n)])
        y = mean([b_scores[dice.randrange(m)] for _ in range(m)])
        if x > y:
            wins += 1.0
        elif x == y:
            wins += 0.5
    return wins / float(resamples)


# ---------------------------------------------------------------- answers


def verdict(train, a, b, confidence=CONFIDENCE):
    """The full story behind :func:`certify`, for when the winemaker asks why.

    Returns a dict with the two guesses, the gap between them, the noise floor
    that gap had to clear, how often the replayed tastings agreed, and the call
    itself.
    """
    cellar = Cellar(train)
    story = {
        "a": None,
        "b": None,
        "gap": 0.0,
        "floor": 0.0,
        "agreement": 0.5,
        "confidence": confidence,
        "call": "same",
        "why": "no scored batches to compare against",
    }
    if cellar.target_index is None or not cellar.scored:
        return story

    guess_a = cellar.guess(a)
    guess_b = cellar.guess(b)
    gap = guess_a - guess_b
    floor = noise_floor(train, cellar)

    story.update({"a": guess_a, "b": guess_b, "gap": gap, "floor": floor})

    if abs(gap) < floor:
        story["why"] = "gap of %.3f is inside our usual miss of %.3f" % (abs(gap), floor)
        return story

    scores_a = _neighbourhood(cellar, a)
    scores_b = _neighbourhood(cellar, b)
    if not scores_a or not scores_b:
        story["why"] = "nothing in the cellar looks like one of these batches"
        return story

    share = _beats(scores_a, scores_b)
    agreement = share if gap > 0 else 1.0 - share
    story["agreement"] = agreement

    if agreement < confidence:
        story["why"] = (
            "look-alike batches only agreed %.0f%% of the time, short of %.0f%%"
            % (100.0 * agreement, 100.0 * confidence)
        )
        return story

    story["call"] = "A" if gap > 0 else "B"
    story["why"] = "gap of %.3f clears our usual miss of %.3f, and %.0f%% of replayed tastings agree" % (
        abs(gap),
        floor,
        100.0 * agreement,
    )
    return story


def certify(train, a, b, confidence=CONFIDENCE):
    """Which of two batches will score better -- or ``'same'`` if it is noise.

    ``train`` is a table in the step 1 shape (header row first); ``a`` and ``b``
    are single lab sheets laid out like the train rows.  Any ``Quality+`` value
    sitting in them is ignored, as in step 2.

    Returns ``'A'``, ``'B'`` or ``'same'``.  A winner is only named when the
    difference survives two separate doubts: the gap between the guesses is
    wider than the distance our guesses are usually wrong by, and the batches'
    look-alike neighbours keep the same winner in at least ``confidence`` of
    the replayed tastings.  Otherwise the honest answer is ``'same'`` -- not
    "the batches are identical", but "we cannot tell them apart".
    """
    return verdict(train, a, b, confidence)["call"]


def accuracy(train, test):
    """Share of unseen test batches whose guessed score rounds to the true one.

    ``train`` and ``test`` are both tables in the step 1 shape; ``test`` may
    omit the header row, in which case the train layout is assumed.  Test rows
    with no panel score are skipped, because there is nothing there to be right
    or wrong about.  A cellar that can be asked nothing scores ``0.0``.

    Rounding follows the panel: a guess of 5.5 counts as a 6.
    """
    cellar = Cellar(train)
    if cellar.target_index is None:
        return 0.0

    header, data = _table(test)
    index = _truth_index(cellar, header)
    if index is None:
        return 0.0

    grain = _grain([s for s in cellar.scores if s is not None])

    right = 0
    tested = 0
    for row in data:
        truth = number(row[index]) if index < len(row) else None
        if truth is None:
            continue
        tested += 1
        if _same_score(cellar.guess(row), truth, grain):
            right += 1

    return right / float(tested) if tested else 0.0


def reliability(rows, folds=FOLDS):
    """How often the guesses are right across a whole cellar, not one split.

    Every batch takes its turn being unseen: the cellar is cut into ``folds``
    slices, each slice guessed at by a cellar built from the others.  Reports
    the share of scores hit exactly, the average miss, and -- for comparison --
    the share you would hit by always saying "a normal score".
    """
    header, data = _table(rows)
    if not header:
        header = []
    if not data:
        return {"tested": 0, "accuracy": None, "error": None, "baseline": None, "swing": None}

    folds = max(2, min(folds, len(data)))

    shares = []
    misses = []
    flat_hits = []
    tested = 0

    for fold in range(folds):
        train = [header] + [row for i, row in enumerate(data) if i % folds != fold]
        held = [row for i, row in enumerate(data) if i % folds == fold]
        if not held or len(train) < 2:
            continue

        cellar = Cellar(train)
        if cellar.target_index is None:
            continue
        grain = _grain([s for s in cellar.scores if s is not None])

        for row in held:
            index = cellar.target_index
            truth = number(row[index]) if index < len(row) else None
            if truth is None:
                continue
            tested += 1
            estimate = cellar.guess(row)
            shares.append(1.0 if _same_score(estimate, truth, grain) else 0.0)
            misses.append(abs(estimate - truth))
            if cellar.typical is not None:
                flat_hits.append(1.0 if _same_score(cellar.typical, truth, grain) else 0.0)

    known = [number(row[target_column(header)[0]]) for row in data] if target_column(header) else []
    known = [value for value in known if value is not None]

    return {
        "tested": tested,
        "accuracy": mean(shares),
        "error": mean(misses),
        "baseline": mean(flat_hits),
        "swing": spread(known),
        "typical": median(known),
    }


# ---------------------------------------------------------------- demo


def _show(path):
    rows = step1.load_rows(path)
    header, data = split(rows)
    print("batches recorded: %d" % len(data))

    target = target_column(header)
    print("judging: %s" % (target[1] if target else "nothing -- no goal column"))
    if not data or target is None:
        return

    print("\nhow often the estimates are right on batches never seen before:")
    report = reliability(rows)
    if report["tested"]:
        print("  tested on %d held-back batches" % report["tested"])
        print("  right to the point: %.1f%%" % (100.0 * report["accuracy"]))
        print("  average miss: %.3f points" % report["error"])
        print(
            "  always saying a normal score (%g) would be right %.1f%% of the time"
            % (report["typical"], 100.0 * report["baseline"])
        )

    cut = max(1, len(data) // 5)
    train = [header] + data[cut:]
    test = [header] + data[:cut]
    print("\n  on one plain hold-back of %d batches: %.1f%% right" % (len(test) - 1, 100.0 * accuracy(train, test)))

    print("\nreal difference, or just tasting noise?")
    print("  noise floor for this cellar: %.3f points" % noise_floor(train))
    best = step1.remember(rows)
    pairs = []
    if best:
        pairs.append(("best on file", best[0], "first on file", data[0]))
    if len(data) > 1:
        pairs.append(("first on file", data[0], "second on file", data[1]))
    for name_a, batch_a, name_b, batch_b in pairs:
        story = verdict(train, batch_a, batch_b)
        call = {"A": name_a, "B": name_b}.get(story["call"], "too close to call")
        print("\n  %s (%.2f) vs %s (%.2f)" % (name_a, story["a"], name_b, story["b"]))
        print("    winner: %s" % call)
        print("    because %s" % story["why"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python step3.py <batches.csv>")
        raise SystemExit(2)
    _show(sys.argv[1])
