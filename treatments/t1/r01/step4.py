"""Step 4 of the winemaker's assistant.

The cellar's records still arrive in the same shape -- a header row whose
names carry their own meaning ("Quality+" is the score the tasting panel
awards, "Alcohol-" a goal to keep low, the rest plain lab readings) -- but
this step does not wait for a finished spreadsheet.  Batches come off the
line one at a time, and the panel's verdicts trickle in through the season
long after the wine was made.

Four questions are answered:

    start(names)         -> a model that knows only the column names
    learn(model, row)    -> the same model, one scored batch wiser
    predict(model, row)  -> the score the panel will probably give a batch
    worth(model, row)    -> is this batch worth a place at the tasting?

Unlike step 2's guesser, which reads the whole cellar every time it is asked,
nothing here keeps the past batches around.  All the model holds is a running
tally per column and one weight per reading, and `learn` nudges those with
the single batch in front of it.  So the cost of a new score is the same on
the thousandth batch as on the first, and this season's scores sharpen the
model as they arrive instead of forcing a fresh fit from scratch.
"""

import math
import sys

import step1
import step2

# The winemaker's own bar: a batch earns its place at the tasting when the
# panel is expected to award it this much or better.
WORTH_THRESHOLD = 6.0

# How hard one batch is allowed to pull on the weights.  It starts brisk, so
# the first dozen scores of a season move the model somewhere useful, and
# settles towards a floor rather than to zero -- a cellar's chemistry drifts,
# and a model that stops listening is a model that goes stale.
STEP_START = 0.5
STEP_FLOOR = 0.05
STEP_HALF_LIFE = 50.0

# Guard against dividing by a spread of nothing on the very first batches.
TINY = 1e-12


def _names(names):
    """The header as a clean list of strings, however it was handed over."""
    if not names:
        return []
    first = names[0]
    if isinstance(first, (list, tuple)):  # a whole table, header included
        return _names(first)
    return [str(name).strip() for name in names]


def _fresh_tally():
    """A running count / mean / sum-of-squared-deviations, Welford style."""
    return [0, 0.0, 0.0]


def _note(tally, value):
    """Fold one more number into a running tally, in constant time."""
    tally[0] += 1
    delta = value - tally[1]
    tally[1] += delta / tally[0]
    tally[2] += delta * (value - tally[1])


def _spread(tally):
    """The tally's standard deviation; 0.0 until there is any spread to see."""
    count, _mean, squares = tally
    if count < 2 or squares <= 0.0:
        return 0.0
    return math.sqrt(squares / count)


def start(names):
    """An empty model: it knows the column names and nothing else.

    The lab sheet is every column except the panel's score -- alcohol
    included, since its reading is on the bench long before the panel sits;
    it is written "Alcohol-" because the winemaker wants it low, not because
    it is unknown.  Only "Quality+" is held out, being the thing predicted.

    A model this empty has no opinion worth hearing: predict() answers 0.0
    and worth() answers False until the first scored batch arrives.
    """
    header = _names(names)
    target = step2._quality_column(header)
    features = [index for index in range(len(header)) if index != target]
    return {
        "names": header,
        "target": target,
        "features": features,
        # one running tally per reading, so each can be centred and put on a
        # comparable scale without a second pass over the data
        "readings": [_fresh_tally() for _ in features],
        "weights": [0.0] * len(features),
        "scores": _fresh_tally(),  # the panel's scores, as a running tally
        "lowest": None,
        "highest": None,
        "learned": 0,  # scored batches folded in so far
        "seen": 0,     # batches seen at all, scored or not
    }


def _reading(row, index):
    return step1._number(row, index)


def _shape(model, row):
    """One batch as (position, comparable reading) pairs, gaps left out.

    Each reading is centred on its running mean and divided by its running
    spread, so a column measured in hundreds cannot shout down one measured
    in tenths.  A missing reading simply does not vote, which amounts to
    treating it as typical.
    """
    shaped = []
    for position, index in enumerate(model["features"]):
        value = _reading(row, index)
        if value is None:
            continue
        tally = model["readings"][position]
        spread = _spread(tally)
        if spread <= TINY:
            continue  # nothing learned about this column's range yet
        shaped.append((position, (value - tally[1]) / spread))
    return shaped


def predict(model, row):
    """The score the panel will probably give this batch, from its lab sheet.

    The answer is the average score the model has seen so far, adjusted by
    each reading's learned pull: a batch whose volatile acidity sits well
    above the cellar's norm is marked down by however much high volatile
    acidity has cost in the past, and so on across the sheet.

    The batch's own Quality+ is never read, so an unjudged batch straight off
    the line is exactly what this expects.  The result is never outside the
    range of scores the panel has actually awarded -- an unusual lab sheet
    should not conjure up a score nobody has ever given.
    """
    if model["learned"] <= 0:
        return 0.0

    predicted = model["scores"][1]
    for position, scaled in _shape(model, row):
        predicted += model["weights"][position] * scaled

    low, high = model["lowest"], model["highest"]
    if low is not None:
        predicted = max(low, min(high, predicted))
    return float(predicted)


def _step_size(learned):
    """How hard to pull on the weights after `learned` scored batches."""
    eagerness = STEP_HALF_LIFE / (STEP_HALF_LIFE + learned)
    return STEP_FLOOR + (STEP_START - STEP_FLOOR) * eagerness


def learn(model, row):
    """Fold one more batch into the model and hand the model back.

    Nothing is refitted.  The running tallies take in the new readings in a
    single pass, and the weights are nudged by the model's own error on this
    one batch: readings that stood out on a batch the model under-rated gain
    pull, readings that stood out on one it over-rated lose it.  The nudge is
    divided by how far the batch strayed from typical overall, so a freak
    sample moves the weights no further than an ordinary one.

    A batch the panel has not scored yet is not wasted: its readings still
    teach the model what normal looks like, even though there is no verdict
    to learn from.  Hand it over again once the score comes in.
    """
    row = list(row) if row else []
    model["seen"] += 1

    # The readings first, so this batch counts towards "what normal looks
    # like" whether or not a score ever follows.
    for position, index in enumerate(model["features"]):
        value = _reading(row, index)
        if value is not None:
            _note(model["readings"][position], value)

    target = model["target"]
    truth = None if target is None else _reading(row, target)
    if truth is None:
        return model  # nothing judged, nothing to learn from

    # Where the model stood before this batch -- that gap is the lesson.
    guessed = predict(model, row)
    shaped = _shape(model, row)

    _note(model["scores"], truth)
    model["learned"] += 1
    model["lowest"] = truth if model["lowest"] is None else min(model["lowest"], truth)
    model["highest"] = truth if model["highest"] is None else max(model["highest"], truth)

    if not shaped:
        return model  # only the running average of scores can improve here

    error = truth - guessed
    strayed = 1.0 + sum(scaled * scaled for _position, scaled in shaped)
    pull = _step_size(model["learned"]) * error / strayed
    for position, scaled in shaped:
        model["weights"][position] += pull * scaled
    return model


def worth(model, row):
    """True when this batch looks worth a place at the next tasting.

    The bar is the winemaker's: a predicted Quality+ of 6 or more.  The
    prediction comes from the lab sheet alone, so the question can be
    answered as the batch comes off the line.

    A model with no scores behind it says False -- not a verdict on the wine,
    just an admission that it has nothing to go on yet.  Feed it this
    season's scores as they arrive and the answer gets sharper.
    """
    return predict(model, row) >= WORTH_THRESHOLD - 1e-9


def fit(rows):
    """Stream a whole cellar through a fresh model, oldest batch first.

    A convenience for picking up where the records leave off; it is the same
    one-batch-at-a-time learning, just run in a loop.
    """
    header, data = step1._split(rows)
    model = start(header)
    for row in data:
        learn(model, row)
    return model


def _round_half_up(value):
    """Round the way a person does: 6.5 goes to 7, not to 6."""
    if value is None:
        return None
    if value >= 0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


def _stream_check(rows):
    """Mark the model as the season would: guess first, learn after.

    Every batch is predicted by a model that has never seen it, then handed
    over to be learned from.  No batch is ever both taught and tested, so
    this is an honest account of what the winemaker would have experienced
    reading the cellar's history one batch at a time.
    """
    header, data = step1._split(rows)
    target = step2._quality_column(header)
    if target is None:
        return None

    model = start(header)
    tested = hits = plain_hits = 0
    missed = 0.0
    sent = sent_good = good = good_caught = 0

    for row in data:
        truth = step1._number(row, target)
        if truth is None:
            learn(model, row)
            continue
        if model["learned"] > 0:
            guessed = predict(model, row)
            tested += 1
            missed += abs(guessed - truth)
            if _round_half_up(guessed) == _round_half_up(truth):
                hits += 1
            if _round_half_up(model["scores"][1]) == _round_half_up(truth):
                plain_hits += 1
            deserved = truth >= WORTH_THRESHOLD
            if worth(model, row):
                sent += 1
                sent_good += 1 if deserved else 0
            if deserved:
                good += 1
                good_caught += 1 if worth(model, row) else 0
        learn(model, row)

    if not tested:
        return None
    return {
        "model": model,
        "tested": tested,
        "right": hits / float(tested),
        "plain": plain_hits / float(tested),
        "miss": missed / float(tested),
        "sent": sent,
        "sent_good": sent_good,
        "good": good,
        "good_caught": good_caught,
    }


def _report(rows):
    header, data = step1._split(rows)
    print("%d past batches, %d columns" % (len(data), len(header)))
    if step2._quality_column(header) is None:
        print("\nno panel score on record: nothing to learn from")
        return

    checked = _stream_check(rows)
    if not checked:
        print("\ntoo few scored batches to say anything yet")
        return

    print("\nReading the cellar one batch at a time (%d guessed before being"
          " learned from):" % checked["tested"])
    print("  right to the nearest point:  %.1f%%" % (100.0 * checked["right"]))
    print("  always saying 'typical':     %.1f%%" % (100.0 * checked["plain"]))
    print("  average miss:                %.3f points" % checked["miss"])

    print("\nWorth sending to the panel (bar: Quality+ of %g or better):"
          % WORTH_THRESHOLD)
    print("  batches waved through:   %d" % checked["sent"])
    if checked["sent"]:
        print("  of those, really good:   %d (%.0f%%)"
              % (checked["sent_good"], 100.0 * checked["sent_good"] / checked["sent"]))
    if checked["good"]:
        print("  good batches caught:     %d of %d (%.0f%%)"
              % (checked["good_caught"], checked["good"],
                 100.0 * checked["good_caught"] / checked["good"]))

    model = checked["model"]
    print("\nWhat the model has come to believe (pull per reading, high is"
          " better):")
    width = max((len(model["names"][i]) for i in model["features"]), default=0)
    ranked = sorted(
        zip(model["features"], model["weights"]),
        key=lambda pair: -abs(pair[1]),
    )
    for index, weight in ranked:
        print("  %-*s  %+.3f" % (width, model["names"][index], weight))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python step4.py <cellar.csv>", file=sys.stderr)
        raise SystemExit(2)
    _report(step1.load_csv(sys.argv[1]))
