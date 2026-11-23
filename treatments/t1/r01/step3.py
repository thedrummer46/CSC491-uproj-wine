"""Step 3 of the winemaker's assistant.

Rows arrive in the same shape as before: row 0 is the header, every later row
is one batch of numbers, and "Quality+" is the score the tasting panel awards.
The readers from step 1 and the guesser from step 2 are reused rather than
rewritten, so the verdicts here are about the same predictions the winemaker
is already being given.

Two more questions are answered:

    certify(train, a, b) -> 'A', 'B' or 'same'
    accuracy(train, test) -> share of unseen batches guessed right

The first keeps the assistant honest about the difference between "A will
beat B" and "A came out slightly ahead this time".  The second keeps it
honest about how much any of its guesses are worth.
"""

import math
import random
import sys

import step1
import step2

# How many times the cellar is re-drawn when weighing one batch against
# another.  More rounds steady the verdict; this is enough to resolve the 95%
# line without keeping the winemaker waiting.
BOOTSTRAP_ROUNDS = 120

# A winner is declared only when the re-drawn cellars agree this often.
CONFIDENCE = 0.95

# Below this many judged batches there is no cellar to resample, so nothing
# can be called a real difference.
MIN_JUDGED = 4

SEED = 20240303  # fixed, so the same two batches always get the same verdict

# The command-line report holds half the cellar back; on a large cellar it
# tests a thinned-out sample of that half so the printout stays quick.
# accuracy() itself never samples.
REPORT_SAMPLE = 150


def _as_row(batch):
    """One batch as a flat row of numbers, whether or not a header came along.

    Tolerant on purpose: the winemaker may hand over a bare row, or a little
    two-line table with its header still attached.
    """
    if batch and all(isinstance(cell, (list, tuple)) for cell in batch):
        return _as_row(batch[-1])
    return list(batch)


def _round_half_up(value):
    """Round the way a person does: 6.5 goes to 7, not to 6."""
    if value is None:
        return None
    if value >= 0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


def _judged(train):
    """(header, batches the panel has actually scored, index of that score)."""
    header, data = step1._split(train)
    target = step2._quality_column(header)
    if target is None:
        return header, [], None
    judged = [row for row in data if step1._number(row, target) is not None]
    return header, judged, target


def _weigh(train, a, b):
    """(verdict, gap, agreement) for one head-to-head comparison.

    `gap` is how far ahead A's guessed score came out on the cellar as it
    stands (negative means B is ahead); `agreement` is the share of re-drawn
    cellars that pointed the same way.  certify() reports only the verdict;
    the report below shows all three.
    """
    header, judged, target = _judged(train)
    if target is None or len(judged) < MIN_JUDGED:
        return "same", 0.0, 0.0

    left, right = _as_row(a), _as_row(b)
    table = [header] + judged
    gap = step2.guess(table, left) - step2.guess(table, right)
    if gap == 0.0:
        # Nothing to test: on this cellar the two batches are the same wine as
        # far as the panel's score is concerned.
        return "same", 0.0, 0.0

    rng = random.Random(SEED)
    count = len(judged)
    wins_a = 0
    wins_b = 0
    for _round in range(BOOTSTRAP_ROUNDS):
        # Draw a cellar of the same size, with replacement: some past batches
        # come up twice, others not at all.  Each draw is a plausible stand-in
        # for the history that might have been recorded, and the scores it
        # carries include whatever the panel's own day-to-day disagreement put
        # into them.
        resample = [header] + [judged[rng.randrange(count)] for _ in range(count)]
        difference = step2.guess(resample, left) - step2.guess(resample, right)
        if difference > 0:
            wins_a += 1
        elif difference < 0:
            wins_b += 1

    agreement = max(wins_a, wins_b) / float(BOOTSTRAP_ROUNDS)
    if agreement < CONFIDENCE:
        return "same", gap, agreement
    if wins_a > wins_b and gap > 0:
        return "A", gap, agreement
    if wins_b > wins_a and gap < 0:
        return "B", gap, agreement
    return "same", gap, agreement


def certify(train, a, b):
    """Which of two batches will score better -- or 'same' if it is just noise.

    Both batches are scored by step 2's guesser, which leans on the past
    batches whose lab sheets look most like theirs.  A bare difference between
    two such guesses is not worth repeating out loud: shift a handful of
    batches in or out of the cellar and the order can flip.

    So the cellar is re-drawn BOOTSTRAP_ROUNDS times, each time sampling the
    judged batches with replacement, and both guesses are made again on every
    draw.  A winner is announced only when at least CONFIDENCE of those draws
    put the same batch ahead; otherwise the honest answer is 'same'.  Two
    things get tested at once that way, because the re-drawn scores carry the
    panel's own inconsistency along with them: a gap has to survive both a
    different cellar and a different day's tasting.

    Returns 'A' when A will score higher, 'B' when B will, 'same' when the
    gap is inside the noise -- also the answer when there is no panel score
    on record, or too little history to resample.
    """
    return _weigh(train, a, b)[0]


def _aligned(train_header, test_header):
    """How to read a test row in the training header's column order.

    None when the two headers already agree, which is the usual case.
    """
    if list(train_header) == list(test_header):
        return None
    lookup = {str(name).strip().lower(): index for index, name in enumerate(test_header)}
    return [lookup.get(str(name).strip().lower()) for name in train_header]


def _recolumn(row, order):
    """The same batch with its readings put where the training header expects."""
    if order is None:
        return row
    return [None if index is None else step1._number(row, index) for index in order]


def accuracy(train, test):
    """Share of test batches whose guessed score rounds to the true Quality+.

    The guess for each test batch is made from `train` alone, so this is a
    fair question only if the test batches are genuinely held back -- batches
    the assistant has never seen.  Nothing here checks for that; it is the
    caller's half of the bargain.

    The panel awards whole points, so "right" means the guess rounds to the
    score on the sheet.  Both numbers are rounded half-up before comparing,
    which matters only for a cellar whose scores are not whole numbers.

    Test batches the panel never judged are skipped -- there is no answer to
    mark against.  Returns 0.0 when none are left, or when the test sheet has
    no score column at all.
    """
    train_header, _train_data = step1._split(train)
    test_header, test_data = step1._split(test)
    target = step2._quality_column(test_header)
    if target is None or not test_data:
        return 0.0

    order = _aligned(train_header, test_header)
    hits = 0
    counted = 0
    for row in test_data:
        truth = step1._number(row, target)
        if truth is None:
            continue
        counted += 1
        guessed = step2.guess(train, _recolumn(row, order))
        if _round_half_up(guessed) == _round_half_up(truth):
            hits += 1
    if not counted:
        return 0.0
    return hits / float(counted)


def _honesty(rows, cap=REPORT_SAMPLE):
    """Hold half the cellar back, then mark ourselves against the lazy answer."""
    header, data = step1._split(rows)
    target = step2._quality_column(header)
    if target is None or len(data) < 4:
        return None

    learn = [header] + data[0::2]
    held = [row for row in data[1::2] if step1._number(row, target) is not None]
    if not held or len(learn) < 2:
        return None

    stride = max(1, -(-len(held) // cap))  # thin the held-back half, evenly
    held = held[::stride]

    share = accuracy(learn, [header] + held)

    learnt = [v for v in (step1._number(r, target) for r in data[0::2]) if v is not None]
    typical = _round_half_up(step1._median(learnt)) if learnt else None
    plain = sum(
        1 for row in held if _round_half_up(step1._number(row, target)) == typical
    ) / float(len(held))
    return share, plain, len(held)


def _demos(rows):
    """A couple of head-to-heads worth showing: a clear one and a close one."""
    header, data = step1._split(rows)
    target = step2._quality_column(header)
    if target is None:
        return []
    judged = [row for row in data if step1._number(row, target) is not None]
    if len(judged) < 2:
        return []

    ordered = sorted(judged, key=lambda row: step1._number(row, target))
    middle = len(ordered) // 2
    pairs = [("best-scoring vs middling", ordered[-1], ordered[middle])]
    if len(ordered) > 4 and middle + 1 < len(ordered):
        pairs.append(("two middling batches", ordered[middle], ordered[middle + 1]))
    return pairs


def _report(rows):
    header, data = step1._split(rows)
    print("%d past batches, %d columns" % (len(data), len(header)))
    if step2._quality_column(header) is None:
        print("\nno panel score on record: nothing to guess, nothing to certify")
        return

    checked = _honesty(rows)
    if checked:
        share, plain, tested = checked
        print("\nScoring batches never seen before (%d held back):" % tested)
        print("  right to the nearest point:  %.1f%%" % (100.0 * share))
        print("  always saying 'typical':     %.1f%%" % (100.0 * plain))

    pairs = _demos(rows)
    if pairs:
        print("\nHead to head:")
    for label, a, b in pairs:
        verdict, gap, agreement = _weigh(rows, a, b)
        spoken = {
            "A": "A scores better",
            "B": "B scores better",
            "same": "no real difference, just tasting noise",
        }[verdict]
        print("  %-26s %s" % (label + ":", spoken))
        print("      gap %+.3f points, %.0f%% of re-drawn cellars agree"
              % (gap, 100.0 * agreement))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python step3.py <cellar.csv>", file=sys.stderr)
        raise SystemExit(2)
    _report(step1.load_csv(sys.argv[1]))
