"""Step 5 of the winemaker's assistant.

The cellar's records arrive in the same shape as before -- row 0 is the
header, every later row is one batch -- but with one important change: the
goal cells are blank.  Quality+ and Alcohol- are both written '?' until the
batch has actually been tasted, because both numbers come off the tasting
bench, not the lab bench.

    label(row) -> the same batch with its goal cells filled in

Every call to `label` is one real tasting: a panel sitting down with a glass.
That is the scarce thing here, so the questions are no longer "what will the
panel say" but "whose glass is worth pouring".

    taste(rows, label, budget=20) -> the batches chosen, in the order tasted
    choose(rows, label, budget=20) -> the best batch found within that budget

Which means a deliberate break with step 2 and step 4: there, Alcohol- was
fair game as an input, since its reading sits on the bench before the panel
does.  Here it does not -- it is '?' like Quality+ -- so only the lab
readings guide the choosing, and alcohol is something to be discovered by
tasting rather than read off a sheet.

The strategy is the one a careful winemaker would use with twenty glasses:
pour the first few from batches as unlike each other as possible, so the
whole cellar gets a hearing; then let what those glasses taught you pick the
rest, leaning harder towards the promising end as the glasses run out.
"""

import math
import sys

import step1
import step2

# What the goal cells say before a batch has been tasted.
UNTASTED = "?"

# The panel's month: twenty glasses.
BUDGET = 20

# Roughly this share of the budget is spent spreading out across the cellar
# before any batch is judged promising.  Too few and the search commits to
# whichever corner it happened to start in; too many and the glasses run out
# before the promising corner is explored properly.
SEED_FRACTION = 0.35
MIN_SEED = 2

# A prediction leans on the closest tasted batches, never more than this many.
NEIGHBOURS_CAP = 10

# How much a batch's unfamiliarity counts alongside how good it looks.  Both
# sit on a 0..1 scale; this weight fades to nothing as the budget empties, so
# the last glasses are poured from the best batch rather than the oddest one.
EXPLORE_WEIGHT = 0.5

# Hidden truth for the demo bench, armed by hide() and read by label().
_BENCH = None


def _goals(header):
    """Columns the winemaker judges by, as (index, direction).  +1 = high is good."""
    return step1._goal_columns(header)


def _reading_columns(header, goals):
    """The lab sheet: every column that is not a goal, so every column known
    before a glass is poured."""
    hidden = {index for index, _direction in goals}
    return [index for index in range(len(header)) if index not in hidden]


def _fingerprint(header, row, goals):
    """A batch identified by its lab sheet alone, which tasting cannot change."""
    hidden = {index for index, _direction in goals}
    return tuple(
        step1._number(row, index)
        for index in range(len(header))
        if index not in hidden
    )


def hide(rows):
    """Blind a complete cellar and arm `label` to reveal it one glass at a time.

    Takes records that already carry the panel's verdicts, hands back
    (blinded rows, label) where the blinded rows have '?' in every goal cell.
    This is how the winemaker's situation is reproduced from a finished CSV:
    the truth exists, but the only way to see any of it is to taste.
    """
    global _BENCH
    header, data = step1._split(rows)
    goals = _goals(header)
    hidden = {index for index, _direction in goals}

    blind = [list(header)]
    truth = {}
    for row in data:
        asked = list(row)
        for index in hidden:
            if index < len(asked):
                asked[index] = UNTASTED
        blind.append(asked)
        truth.setdefault(_fingerprint(header, row, goals), list(row))

    _BENCH = {"header": header, "goals": goals, "truth": truth, "tastings": 0}
    return blind, label


def label(row):
    """Taste one batch: the same row back, with its goal cells filled in.

    One call is one tasting, and tastings are what the budget counts.  A
    batch the bench has never heard of comes back exactly as it went in --
    there is no glass to pour from it -- but the tasting is still counted,
    because the panel's time was still spent.
    """
    if _BENCH is None:
        raise RuntimeError("no tasting bench: call hide(rows) first")
    _BENCH["tastings"] += 1
    known = _BENCH["truth"].get(_fingerprint(_BENCH["header"], row, _BENCH["goals"]))
    return list(known) if known is not None else list(row)


def tastings():
    """How many glasses the armed bench has poured so far."""
    return 0 if _BENCH is None else _BENCH["tastings"]


def _taste_one(label, row):
    """Hand one batch to the panel and take back whatever comes.

    Tolerant on purpose: a `label` that fills the row in place and returns
    nothing is treated the same as one that returns the filled row.
    """
    answered = label(row)
    if isinstance(answered, (list, tuple)):
        return list(answered)
    return list(row)


def _predict(wanted, known):
    """One goal's value for an untasted batch, from the tasted batches near it.

    The same reasoning step 2 uses for Quality+, applied to whichever goal is
    asked for: average what the closest tasted batches turned out to be,
    leaning hardest on the closest, and never answer outside the range of
    values actually tasted.  None when nothing comparable has been tasted.
    """
    measured = []
    for vector, value in known:
        distance = step2._distance(wanted, vector)
        if distance is not None:
            measured.append((distance, value))
    if not measured:
        return None

    measured.sort(key=lambda pair: pair[0])
    count = max(1, min(NEIGHBOURS_CAP, int(round(math.sqrt(len(measured))))))
    near = measured[:count]

    # The furthest neighbour kept sets the scale of "close"; softening by a
    # tenth of it keeps an all-but-identical batch from drowning out the rest.
    softener = 0.1 * near[-1][0] if near[-1][0] > 0 else 0.1
    predicted = 0.0
    weight_total = 0.0
    for distance, value in near:
        weight = 1.0 / (distance + softener)
        predicted += weight * value
        weight_total += weight
    if weight_total <= 0:
        return None

    values = [value for _distance, value in measured]
    return min(max(values), max(min(values), predicted / weight_total))


def _nearest_tasted(vector, tasted_vectors):
    """How far this batch sits from the nearest glass already poured."""
    distances = [
        distance
        for distance in (step2._distance(vector, other) for other in tasted_vectors)
        if distance is not None
    ]
    return min(distances) if distances else 0.0


def _even_pick(count, rounds, step, untasted):
    """The next of `rounds` evenly spaced batches that has not been tasted yet.

    The fallback for a cellar whose lab readings say nothing -- no readings at
    all, or none that vary.  Walking the records at a steady stride at least
    samples the whole season rather than only its opening weeks.
    """
    if rounds > 1:
        wanted = int(round(step * (count - 1) / float(rounds - 1)))
    else:
        wanted = 0
    best = None
    for index in sorted(untasted):
        gap = abs(index - wanted)
        if best is None or gap < best[0]:
            best = (gap, index)
    return best[1]


def _spread_pick(vectors, order, untasted):
    """The untasted batch least like anything tasted so far.

    The first pour goes to the most typical batch in the cellar -- the one
    nearest the median lab sheet -- because it is the single most informative
    glass when nothing is known.  After that, each pour goes to whichever
    batch sits furthest from every glass already poured, which fans the
    tastings out over the cellar instead of bunching them in one corner.
    """
    candidates = sorted(untasted)
    if not order:
        best = None
        for index in candidates:
            # vectors are centred on the cellar's medians, so the batch
            # closest to the origin is the most ordinary one on the books
            typicality = sum(value * value for value in vectors[index])
            if best is None or typicality < best[0]:
                best = (typicality, index)
        return best[1]

    tasted_vectors = [vectors[position] for position in order]
    best = None
    for index in candidates:
        gap = _nearest_tasted(vectors[index], tasted_vectors)
        if best is None or gap > best[0]:
            best = (gap, index)
    return best[1]


def _learned(vectors, order, tasted, goals):
    """What the glasses poured so far have taught us, per goal column.

    Returns (known, spans): the tasted (lab sheet, value) pairs for each goal
    that can be predicted at all, and the range of values seen for it.  A
    goal whose tasted values are all identical is left out -- it cannot tell
    two batches apart yet, so it should not pretend to.
    """
    known = {}
    spans = {}
    for index, _direction in goals:
        pairs = []
        for position, row in zip(order, tasted):
            value = step1._number(row, index)
            if value is not None:
                pairs.append((vectors[position], value))
        if not pairs:
            continue
        values = [value for _vector, value in pairs]
        low, high = min(values), max(values)
        if high > low:
            known[index] = pairs
            spans[index] = (low, high)
    return known, spans


def _promise(vector, goals, known, spans):
    """How good an untasted batch looks: 0.0 (worst tasted) to 1.0 (best tasted).

    Every goal is predicted from the nearby tasted batches, rescaled onto
    0..1 over the range tasted so far, flipped where lower is better, then
    averaged -- the same yardstick step 1 used to single out exceptional
    batches, so "high Quality+, low Alcohol-" means here exactly what it
    meant there.  None when no goal can be predicted yet.
    """
    parts = []
    for index, direction in goals:
        if index not in known:
            continue
        value = _predict(vector, known[index])
        if value is None:
            continue
        low, high = spans[index]
        fraction = (value - low) / (high - low)
        fraction = max(0.0, min(1.0, fraction))
        parts.append(fraction if direction > 0 else 1.0 - fraction)
    if not parts:
        return None
    return sum(parts) / len(parts)


def _model_pick(vectors, order, tasted, untasted, goals, explore):
    """The next batch worth a glass, weighing promise against unfamiliarity.

    A batch is scored on how good it looks from what has been tasted, plus a
    bonus for sitting in a part of the cellar no glass has reached yet.  The
    bonus fades as the budget empties: early glasses are allowed to be
    curious, the last ones are spent on the best wine we know of.

    None when the glasses so far have taught nothing usable -- all the same
    score, say -- in which case the caller goes back to spreading out.
    """
    known, spans = _learned(vectors, order, tasted, goals)
    if not known:
        return None

    candidates = sorted(untasted)
    tasted_vectors = [vectors[position] for position in order]
    novelty = {index: _nearest_tasted(vectors[index], tasted_vectors) for index in candidates}
    widest = max(novelty.values()) if novelty else 0.0

    best = None
    for index in candidates:
        promise = _promise(vectors[index], goals, known, spans)
        if promise is None:
            continue
        unfamiliar = 0.0 if widest <= 0 else novelty[index] / widest
        score = promise + explore * unfamiliar
        if best is None or score > best[0]:
            best = (score, index)
    return None if best is None else best[1]


def taste(rows, label, budget=BUDGET):
    """The batches to put in front of the panel, in the order to taste them.

    Answers "the panel can only taste 20 this month -- which 20?".  At most
    `budget` glasses are poured, one per batch, and no batch is poured twice.
    The returned rows are the labelled ones, goals filled in, in tasting
    order, so the winemaker can read the session back as it happened.

    The choosing is adaptive: roughly the first third of the budget fans out
    over the cellar by lab sheet, giving every style of wine a hearing, and
    each glass after that is chosen in the light of every glass before it --
    mostly from wherever the tasted batches suggest the best wine lives, with
    a shrinking allowance for curiosity.  That is what makes this different
    from picking twenty batches off the shelf in advance: a fixed list cannot
    learn from the first nineteen glasses.
    """
    header, data = step1._split(rows)
    if not data or not budget or budget <= 0:
        return []

    goals = _goals(header)
    scales = step2._scales(data, _reading_columns(header, goals))
    if scales:
        vectors = [step2._filled(step2._vector(row, scales)) for row in data]
    else:
        vectors = [[] for _ in data]

    rounds = int(min(budget, len(data)))
    if rounds <= 0:
        return []

    seed = min(rounds, max(MIN_SEED, int(round(rounds * SEED_FRACTION))))
    if not scales or not goals:
        seed = rounds  # nothing to model from: just spread the glasses out

    order = []
    tasted = []
    untasted = set(range(len(data)))

    for step in range(rounds):
        if not untasted:
            break
        pick = None
        if step >= seed:
            explore = EXPLORE_WEIGHT * (rounds - step) / float(rounds)
            pick = _model_pick(vectors, order, tasted, untasted, goals, explore)
        if pick is None:
            if scales:
                pick = _spread_pick(vectors, order, untasted)
            else:
                pick = _even_pick(len(data), rounds, step, untasted)

        tasted.append(_taste_one(label, data[pick]))
        order.append(pick)
        untasted.discard(pick)

    return tasted


def choose(rows, label, budget=BUDGET):
    """The best batch found -- high Quality+, low Alcohol- -- within the budget.

    Answers "find me a batch that scores high but stays low in alcohol, while
    tasting as few batches as possible".  The glasses are poured by taste()
    above, so the search spends its budget where the wine looks best rather
    than reading down the list; then the winner is picked from what was
    actually tasted, on measured goals only.  Nothing is claimed about a
    batch no one has drunk.

    Both goals are rescaled over the range tasted and averaged, so neither
    the score nor the alcohol can win on its units alone: the batch handed
    back is the best compromise found, not the highest Quality+ regardless of
    strength.  Ties go to the glass poured first, that being the one the
    search thought more promising.  An empty cellar, or a budget of nothing,
    gives [].
    """
    header, _data = step1._split(rows)
    tasted = taste(rows, label, budget)
    if not tasted:
        return []

    goals = _goals(header)
    spans = {}
    for index, _direction in goals:
        values = [
            value
            for value in (step1._number(row, index) for row in tasted)
            if value is not None
        ]
        spans[index] = (min(values), max(values)) if values else (None, None)

    best = None
    for position, row in enumerate(tasted):
        score = step1._batch_score(row, goals, spans)
        if score is None:
            continue
        if best is None or score > best[0]:
            best = (score, position)
    if best is None:
        return tasted[0]  # one glass, or no spread to judge by: that is the answer
    return tasted[best[1]]


def _cellar_yardstick(rows):
    """Score every batch in a fully known cellar, for marking our own homework."""
    header, data = step1._split(rows)
    goals = _goals(header)
    spans = {}
    for index, _direction in goals:
        values = step1._column(data, index)
        spans[index] = (min(values), max(values)) if values else (None, None)
    scored = [step1._batch_score(row, goals, spans) for row in data]
    return goals, spans, scored


def _standing(score, scored):
    """Where one score sits in the cellar: 1.0 means nothing beats it."""
    usable = [value for value in scored if value is not None]
    if score is None or not usable:
        return None
    beaten = sum(1 for value in usable if value <= score)
    return beaten / float(len(usable))


def _blind_even(blind, budget):
    """The lazy plan, for comparison: a fixed evenly spaced list, chosen up front."""
    _header, data = step1._split(blind)
    rounds = int(min(budget, len(data)))
    if rounds <= 0:
        return []
    if rounds == 1:
        return [data[0]]
    return [data[int(round(step * (len(data) - 1) / float(rounds - 1)))] for step in range(rounds)]


def _report(rows, budget=BUDGET):
    header, data = step1._split(rows)
    goals = _goals(header)
    print("%d batches on the books, %d columns" % (len(data), len(header)))
    print("goals: " + (", ".join(header[index] for index, _d in goals) or "none"))
    if not goals or not data:
        print("\nnothing to taste for: no goal columns on the sheet")
        return

    _goals_again, spans, scored = _cellar_yardstick(rows)

    blind, taster = hide(rows)
    print("\nThe panel's month (%d glasses):" % budget)
    session = taste(blind, taster, budget)
    print("  glasses poured: %d" % tastings())
    width = max((len(name) for name in header), default=0)
    for number, row in enumerate(session, start=1):
        shown = ", ".join(
            "%s %.4g" % (header[index], step1._number(row, index))
            for index, _d in goals
            if step1._number(row, index) is not None
        )
        print("  %2d. %s" % (number, shown or "nothing tasted"))

    blind, taster = hide(rows)
    picked = choose(blind, taster, budget)
    print("\nThe batch it brings back (%d glasses):" % tastings())
    if not picked:
        print("  none")
        return
    for index, name in enumerate(header):
        value = step1._number(picked, index)
        mark = " <- goal" if any(index == g for g, _d in goals) else ""
        print("  %-*s  %s%s" % (width, name, "n/a" if value is None else "%.4g" % value, mark))

    mine = step1._batch_score(picked, goals, spans)
    standing = _standing(mine, scored)
    if standing is not None:
        print("\n  better than %.1f%% of the whole cellar" % (100.0 * standing))

    plain = _blind_even(blind, budget)
    best_plain = None
    for row in plain:
        score = step1._batch_score(row, goals, spans)
        if score is not None and (best_plain is None or score > best_plain):
            best_plain = score
    plain_standing = _standing(best_plain, scored)
    if plain_standing is not None:
        print("  tasting %d spread evenly instead: %.1f%%" % (len(plain), 100.0 * plain_standing))
    print("  tasting all %d: 100.0%% (and %d times the glasses)"
          % (len(data), max(1, len(data) // max(1, budget))))


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("usage: python step5.py <cellar.csv> [budget]", file=sys.stderr)
        raise SystemExit(2)
    _report(step1.load_csv(sys.argv[1]),
            int(sys.argv[2]) if len(sys.argv) == 3 else BUDGET)
