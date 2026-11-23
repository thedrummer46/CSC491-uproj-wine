"""Winemaker's AI assistant -- step 4.

Builds on :mod:`step1` (reading a batch table, telling a goal column from an
ordinary lab reading), :mod:`step2` (the cellar of past lab sheets and the
guess it makes for a new one) and :mod:`step3` (saying when a difference is
real and when it is only tasting noise).  Nothing written there is written
again here.

Steps 1 to 3 all begin by looking at the whole cellar at once.  That is no use
on the line, where batches arrive one at a time and the panel's verdicts
trickle back weeks later.  So step 4 answers two last requests:

``worth(model, row)``
    "As each new batch comes off the line, tell me right away whether it's
    worth sending to the panel."

``learn(model, row)``
    "Keep sharpening your sense of what makes a good batch as this season's
    scores come in, without starting over."

The model is a running one.  It keeps three things, each cheap to update and
none of them rebuilt from the beginning:

* running counts, means and swings for every lab column, so a reading can be
  put on a common footing without a second pass over the data;
* a running set of weights -- one per lab column -- nudged a little by each
  newly scored batch, which is where "what makes a good batch" is stored;
* a bounded remembrance of actual batches, so the step 2 question "which past
  batches does this one resemble?" still has an answer.

The two voices, weights and look-alikes, are blended by how wrong each has
been lately, and every batch is scored *before* it is learned from, so that
record is honest -- no batch ever helps predict itself.
"""

import collections
import math
import random

import step1
import step2
import step3
from step1 import classify, mean, median, number, split
from step2 import target_column

# A batch earns a place in front of the panel at this score or better.
PANEL_THRESHOLD = 6.0

# How hard one newly scored batch is allowed to pull the weights around.
LEARNING_RATE = 0.5

# How many past batches are kept for look-alike comparisons, and how many of
# those are reserved for the most recent arrivals.
MEMORY = 240
RECENT = 40

# A reading this many swings from normal is pinned there; one freak batch
# should not drown out the rest.
CLIP = 4.0

# Fixed, so the same season of batches always yields the same model.
SEED = 20260101


# ---------------------------------------------------------------- the model


def start(names):
    """An empty model that knows only the header row.

    ``names`` is the list of column names -- the ``+`` / ``-`` marks are read
    exactly as in step 1, so the model knows which column the panel fills in
    (``Quality+``) and which are lab readings it will be given up front.  A
    whole table can be handed over instead; only its header row is used.

    Nothing is known about any batch yet, so the model declines to recommend
    anything until it has been shown at least one score.
    """
    if names and isinstance(names[0], (list, tuple)):
        names = names[0]
    header = [str(name).strip() for name in names]

    target = target_column(header)
    target_index = target[0] if target else None
    clues = [i for i in range(len(header)) if i != target_index]

    return {
        "header": header,
        "target": target,
        "target_index": target_index,
        "clues": clues,
        "clue_names": [header[i] for i in clues],
        # running shape of each lab column (Welford)
        "count": [0] * len(clues),
        "mu": [0.0] * len(clues),
        "m2": [0.0] * len(clues),
        # what makes a good batch, so far
        "weights": [0.0] * len(clues),
        # running shape of the panel's scores
        "seen": 0,
        "scored": 0,
        "score_mean": 0.0,
        "score_m2": 0.0,
        "score_low": None,
        "score_high": None,
        "whole": True,
        # a bounded remembrance of real batches
        "recent": collections.deque(maxlen=RECENT),
        "sample": [],
        "dice": random.Random(SEED),
        # how wrong each voice has been, measured before learning
        "miss": {"line": [0.0, 0], "near": [0.0, 0], "blend": [0.0, 0]},
    }


# ---------------------------------------------------------------- footing


def _clues(model, row):
    """One lab sheet as raw clue values, ``None`` where a reading is missing."""
    return [number(row[i]) if i < len(row) else None for i in model["clues"]]


def _swing(model, k):
    """The usual swing of one lab column so far; 0.0 before it can be told."""
    n = model["count"][k]
    if n < 2:
        return 0.0
    return math.sqrt(model["m2"][k] / float(n - 1))


def _footing(model, values):
    """Put a lab sheet on a common footing: swings from normal, per column.

    This replaces step 2's scaling by the lowest and highest ever seen, which
    needs the whole cellar in hand.  A running mean and swing need only what
    has already gone by.  A missing reading sits at normal (0.0), and a freak
    one is pinned at :data:`CLIP` swings out.
    """
    point = []
    for k, value in enumerate(values):
        swing = _swing(model, k)
        if value is None or swing <= 0.0:
            point.append(0.0)
        else:
            point.append(max(-CLIP, min(CLIP, (value - model["mu"][k]) / swing)))
    return tuple(point)


def _watch(model, values):
    """Fold one more lab sheet into the running shape of each column."""
    for k, value in enumerate(values):
        if value is None:
            continue
        model["count"][k] += 1
        delta = value - model["mu"][k]
        model["mu"][k] += delta / float(model["count"][k])
        model["m2"][k] += delta * (value - model["mu"][k])


def _clamp(model, value):
    """Keep an estimate inside the range of scores the panel has ever given."""
    low = model["score_low"]
    high = model["score_high"]
    if low is None or high is None:
        return float(value)
    return float(max(low, min(high, value)))


def grain(model):
    """The step the panel scores in: whole points, or ``None`` if finer.

    The same question step 3 asks of a finished cellar, asked of a running one.
    """
    return 1.0 if model["whole"] else None


# ---------------------------------------------------------------- two voices


def _line(model, point):
    """The weights' estimate: a normal score, pushed about by the readings."""
    if not model["scored"]:
        return None
    return model["score_mean"] + sum(w * x for w, x in zip(model["weights"], point))


def _pool(model):
    """The batches kept for comparison: this season's arrivals, plus a sample.

    The most recent batches are in both buckets, so they count twice over --
    deliberately, since what the line is making now resembles what it made
    last week more than what it made last season.
    """
    return list(model["recent"]) + model["sample"]


def _near(model, point):
    """The look-alikes' estimate, the nearest counting most (as in step 2)."""
    pool = _pool(model)
    if not pool:
        return None

    count = int(round(math.sqrt(len(pool))))
    count = max(1, min(step2.NEIGHBOURS_CAP, count, len(pool)))

    ranked = sorted(
        (step2._distance(point, _footing(model, values)), score)
        for values, score in pool
    )[:count]

    exact = [score for gap, score in ranked if gap <= 0.0]
    if exact:
        return float(mean(exact))

    weight_total = 0.0
    score_total = 0.0
    for gap, score in ranked:
        weight = 1.0 / gap
        weight_total += weight
        score_total += weight * score
    if weight_total <= 0.0:
        return float(mean([score for _gap, score in ranked]))
    return float(score_total / weight_total)


def _trust(model, voice):
    """How much to listen to one voice: less, the more it has been missing by."""
    total, n = model["miss"][voice]
    if n < 2:
        return 1.0
    return 1.0 / (0.05 + total / float(n))


def _blend(model, point):
    """Both voices, weighted by their own record so far."""
    line = _line(model, point)
    near = _near(model, point)
    if line is None:
        return near
    if near is None:
        return line
    trust_line = _trust(model, "line")
    trust_near = _trust(model, "near")
    return (trust_line * line + trust_near * near) / (trust_line + trust_near)


# ---------------------------------------------------------------- answers


def learn(model, row):
    """The model after seeing one more scored batch; ``row`` includes its goals.

    Nothing is refitted.  The column shapes take one more reading each, the
    weights take a single nudge in the direction that would have made this
    batch's estimate right, and the batch itself is filed away within a fixed
    budget of remembered batches.  The cost of a call does not grow with the
    number of batches already seen.

    The batch is estimated *before* any of that happens, and the miss recorded,
    which is how the two voices earn or lose the model's trust.  A batch whose
    panel score has not come back yet still teaches the model what a normal lab
    sheet looks like.

    The same model object is updated and returned, so ``model = learn(model,
    row)`` and a bare ``learn(model, row)`` both do the right thing.
    """
    # A header row handed over by mistake teaches nothing; let it pass through.
    if row and any(isinstance(cell, str) and number(cell) is None for cell in row):
        return model

    index = model["target_index"]
    truth = number(row[index]) if index is not None and index < len(row) else None
    values = _clues(model, row)
    model["seen"] += 1

    # Mark the paper before looking at the answer.
    if truth is not None:
        point = _footing(model, values)
        for voice, estimate in (
            ("line", _line(model, point)),
            ("near", _near(model, point)),
            ("blend", _blend(model, point)),
        ):
            if estimate is None:
                continue
            record = model["miss"][voice]
            record[0] += abs(_clamp(model, estimate) - truth)
            record[1] += 1

    _watch(model, values)

    if truth is None:
        return model

    # Running shape of the panel's scores.
    model["scored"] += 1
    delta = truth - model["score_mean"]
    model["score_mean"] += delta / float(model["scored"])
    model["score_m2"] += delta * (truth - model["score_mean"])
    low = model["score_low"]
    high = model["score_high"]
    model["score_low"] = truth if low is None else min(low, truth)
    model["score_high"] = truth if high is None else max(high, truth)
    if abs(truth - round(truth)) > 1e-9:
        model["whole"] = False

    # One nudge to the weights, sized so a batch with wild readings does not
    # shove them further than a batch with ordinary ones.
    point = _footing(model, values)
    error = (truth - model["score_mean"]) - sum(
        w * x for w, x in zip(model["weights"], point)
    )
    energy = sum(x * x for x in point)
    if energy > 0.0:
        step = LEARNING_RATE * error / (1.0 + energy)
        model["weights"] = [w + step * x for w, x in zip(model["weights"], point)]

    # File the batch itself, inside a fixed budget.
    model["recent"].append((tuple(values), float(truth)))
    _keep(model, values, truth)
    return model


def _keep(model, values, score):
    """Hold on to one batch for comparison, never growing past :data:`MEMORY`.

    Once the budget is full a newcomer takes a place at random, each batch of
    the season standing an equal chance of being the one remembered.  The dice
    are seeded, so the same season gives the same model twice running.
    """
    sample = model["sample"]
    item = (tuple(values), float(score))
    if len(sample) < MEMORY:
        sample.append(item)
        return
    slot = model["dice"].randrange(model["scored"])
    if slot < MEMORY:
        sample[slot] = item


def learn_all(model, rows):
    """Stream a whole table of scored batches through :func:`learn`, in order.

    A convenience for catching a model up on last season; a header row at the
    top is ignored, and the model is left exactly as it would be had the
    batches arrived one by one.
    """
    _header, data = step3._table(rows)
    for row in data:
        learn(model, row)
    return model


def predict(model, row):
    """Predicted ``Quality+`` for one batch, from what has been seen so far.

    Any score already sitting in ``row`` is ignored, as in steps 2 and 3 -- the
    point is to answer as the batch comes off the line.  The estimate is the
    blend of the two voices, held inside the range of scores the panel has
    actually handed out.

    A model that has seen no scores returns ``0.0``: it has nothing to say, and
    saying nothing is better than inventing a number.
    """
    point = _footing(model, _clues(model, row))
    estimate = _blend(model, point)
    if estimate is None:
        return 0.0
    return _clamp(model, estimate)


def worth(model, row, threshold=PANEL_THRESHOLD):
    """True when this batch looks worth sending to the panel.

    "Worth it" means a predicted ``Quality+`` of ``threshold`` or better, 6 by
    default.  A model that has seen no scores recommends nothing; there is no
    sense in spending panel time on the strength of an empty model.
    """
    if not model["scored"]:
        return False
    return predict(model, row) >= threshold - 1e-9


# ---------------------------------------------------------------- the why


def noise_floor(model):
    """How far from the threshold a batch must sit for the call to mean much.

    The running cousin of :func:`step3.noise_floor`: the distance the blended
    estimate has actually been wrong by, lately, and never less than half a
    panel point, because a smaller gap would not change the score they write
    down anyway.
    """
    total, n = model["miss"]["blend"]
    miss = total / float(n) if n else None
    if not miss:
        swing = (
            math.sqrt(model["score_m2"] / float(model["scored"] - 1))
            if model["scored"] > 1
            else 0.0
        )
        miss = swing
    floor = float(miss or 0.0)
    step = grain(model)
    if step:
        floor = max(floor, step / 2.0)
    return floor


def report(model, row, threshold=PANEL_THRESHOLD):
    """The full story behind one :func:`worth` call, for when the cellar asks why.

    Returns the estimate, the threshold it was measured against, the margin
    between them, the distance estimates are usually wrong by, the call itself,
    and whether that call is comfortable or too close to be sure of.
    """
    estimate = predict(model, row)
    floor = noise_floor(model)
    margin = estimate - threshold
    send = worth(model, row, threshold)

    if not model["scored"]:
        why = "nothing learned yet -- no scores have come in"
    elif abs(margin) < floor:
        why = "estimate of %.2f is within our usual miss of %.2f of the %g line" % (
            estimate,
            floor,
            threshold,
        )
    elif send:
        why = "estimate of %.2f clears %g by %.2f, more than our usual miss of %.2f" % (
            estimate,
            threshold,
            margin,
            floor,
        )
    else:
        why = "estimate of %.2f falls %.2f short of %g, more than our usual miss of %.2f" % (
            estimate,
            -margin,
            threshold,
            floor,
        )

    return {
        "guess": estimate,
        "threshold": float(threshold),
        "margin": margin,
        "floor": floor,
        "send": send,
        "sure": model["scored"] > 0 and abs(margin) >= floor,
        "batches_learned": model["scored"],
        "why": why,
    }


def sense(model, most=None):
    """What the model currently thinks makes a good batch.

    A list of ``(name, pull, direction)``, strongest pull first, where ``pull``
    is how many panel points one usual swing of that reading moves the estimate
    and ``direction`` is ``'better'`` or ``'worse'``.  This is the sense that
    :func:`learn` sharpens -- reading it is how the winemaker sees it change.
    """
    out = []
    for k, name in enumerate(model["clue_names"]):
        weight = model["weights"][k]
        out.append((name, abs(weight), "better" if weight > 0 else "worse"))
    out.sort(key=lambda item: (-item[1], item[0]))
    return out[:most] if most else out


def standing(model):
    """How the model is doing: batches learned, and each voice's average miss."""
    out = {"seen": model["seen"], "learned": model["scored"]}
    for voice in ("line", "near", "blend"):
        total, n = model["miss"][voice]
        out[voice] = total / float(n) if n else None
    out["floor"] = noise_floor(model) if model["scored"] else None
    return out


# ---------------------------------------------------------------- demo


def _show(path):
    """Replay a cellar as though it came off the line one batch at a time."""
    rows = step1.load_rows(path)
    header, data = split(rows)
    print("batches on file: %d" % len(data))

    target = target_column(header)
    print("judging: %s" % (target[1] if target else "nothing -- no goal column"))
    if not data or target is None:
        return

    model = start(header)
    index = model["target_index"]

    right = 0
    tested = 0
    sent_well = 0
    sent = 0
    missed_good = 0
    good = 0

    for row in data:
        truth = number(row[index]) if index < len(row) else None

        # Decide before learning: this is the call the line would have got.
        if truth is not None and model["scored"]:
            tested += 1
            estimate = predict(model, row)
            step = grain(model)
            if step3._same_score(estimate, truth, step):
                right += 1
            recommend = worth(model, row)
            deserved = truth >= PANEL_THRESHOLD
            if recommend:
                sent += 1
                if deserved:
                    sent_well += 1
            if deserved:
                good += 1
                if not recommend:
                    missed_good += 1

        learn(model, row)

    print("\nlearning as the season went by, judged only on batches not yet seen:")
    print("  calls made: %d" % tested)
    if tested:
        print("  score right to the point: %.1f%%" % (100.0 * right / tested))
    report_ = standing(model)
    for voice, label in (("line", "weights"), ("near", "look-alikes"), ("blend", "blended")):
        if report_[voice] is not None:
            print("  average miss, %-11s %.3f points" % (label + ":", report_[voice]))
    print("  usual miss to beat before a call means anything: %.3f" % report_["floor"])

    print("\nsending to the panel (a batch is worth it at %g or better):" % PANEL_THRESHOLD)
    print("  recommended: %d batches" % sent)
    if sent:
        print("  of those, the panel agreed about: %.1f%%" % (100.0 * sent_well / sent))
    if good:
        print("  good batches we would have held back: %d of %d" % (missed_good, good))

    print("\nwhat the model thinks makes a good batch:")
    for name, pull, direction in sense(model, 6):
        print("  %-16s %.3f points per usual swing  (%s)" % (name, pull, direction))

    print("\nthe batch now coming off the line (last on file):")
    story = report(model, data[-1])
    print("  estimate: %.2f  ->  %s" % (story["guess"], "send it" if story["send"] else "hold it back"))
    print("  because %s" % story["why"])
    if not story["sure"]:
        print("  (too close to the line to be sure of -- a second opinion is cheap)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python step4.py <batches.csv>")
        raise SystemExit(2)
    _show(sys.argv[1])
