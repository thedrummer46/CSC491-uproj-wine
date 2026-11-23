"""Winemaker's AI assistant -- step 5.

Builds on :mod:`step1` (reading a batch table, telling a goal column from an
ordinary lab reading, and picking out the exceptional batches), :mod:`step2`
(the cellar of past lab sheets, the guess it makes for a new one, and the few
kinds of batch it contains), :mod:`step3` (saying when a difference is real and
when it is only tasting noise) and :mod:`step4` (learning as the season goes
by).  Nothing written there is written again here.

Every step so far has assumed the panel's verdict was already on the sheet.
This one does not.  Here the goal cells -- ``Alcohol-`` and ``Quality+`` -- are
``'?'`` on every row, and the only way to fill them in is to taste the batch:

``label(row)``
    The winemaker's own hand: hand it one batch and it comes back with its
    goals filled in.  Each call is one tasting, and tastings are the expensive
    thing -- the panel sits once a month and can only work through so many
    glasses.

So two requests, both about spending a small tasting budget well:

``taste(rows, label, budget=20)``
    "The panel can only taste 20 batches this month.  Which 20 should they
    taste to find our best wine?"

``choose(rows, label, budget=20)``
    "Find me a batch that scores high but stays low in alcohol, while tasting
    as few batches as possible."

The plan is the obvious one, and it is the plan a good cellar master already
follows.  Spend the first few tastings getting the lay of the land -- one batch
from each kind the cellar contains, so no corner of it goes unvisited.  After
that, every tasting is chosen on what has already been poured: the untasted
batches are all guessed at from the tasted ones, and the glass that goes to the
panel next is the one that looks most promising, counting a batch we are unsure
about as promising too.  Being unsure is itself a reason to taste something.
"""

import random

import step1
import step2
import step3
import step4
from step1 import classify, mean, number, split, spread
from step2 import Cellar

# How many batches the panel can get through in a month.
BUDGET = 20

# The share of the budget spent getting the lay of the land before any
# guessing is done.
WARMUP_SHARE = 0.25

# How much an unsure guess is flattered.  At 1.0 a batch we cannot call is
# treated as being as promising as its most hopeful reading allows.
CURIOSITY = 1.0

# On a very large cellar, this many untasted batches are weighed each round.
# Looking at every single one changes the pick almost never and costs plenty.
CANDIDATES = 500

# Replays used when reporting how a budget did against plain luck.
TRIALS = 50

# Fixed, so the same cellar always yields the same tasting plan.
SEED = step4.SEED

UNKNOWN = "?"


# ---------------------------------------------------------------- the sheet


def _marks(goals):
    """The column positions of the goal cells -- the ones tasting fills in."""
    return [index for index, _name, _direction in goals]


def _cell(row, index):
    """One cell as a number, or ``None`` when it is missing or still ``'?'``."""
    return number(row[index]) if index < len(row) else None


def tasted(row, goals):
    """True when this batch has already been through the panel.

    A batch counts as tasted once every goal cell carries a number; an
    untasted one still has ``'?'`` sitting there.
    """
    return all(_cell(row, index) is not None for index in _marks(goals))


def blank(rows, mark=UNKNOWN):
    """Turn a finished cellar back into an untasted one, plus the hand to taste it.

    Returns ``(blind_rows, label)``: the same table with every goal cell
    replaced by ``mark``, and a ``label`` that hands back the real row.  This
    is how a past season is replayed to see what a month's budget would have
    found -- the honest way, since the plan never sees a goal it has not paid
    for.
    """
    header, data = split(rows)
    goals, _readings = classify(header)
    marks = _marks(goals)

    blind = [list(header)]
    truth = {}
    for row in data:
        hidden = list(row)
        for index in marks:
            if index < len(hidden):
                hidden[index] = mark
        blind.append(hidden)
        truth[id(hidden)] = list(row)

    def label(row):
        """One tasting: the same batch, with its goals filled in."""
        full = truth.get(id(row))
        return list(full) if full is not None else list(row)

    return blind, label


# ---------------------------------------------------------------- one goal at a time


def _view(header, readings, goal, rows):
    """A one-goal table: the lab readings, with a single goal column last.

    Step 2's cellar answers "what score will this batch get?" for one column
    at a time, so each goal gets its own view of the tasted batches.  The other
    goal is left out of the view entirely rather than being passed off as a lab
    reading: it is not on the sheet when the batch comes off the line either.
    """
    index, name, _direction = goal
    keep = [i for i, _n in readings]
    table = [[header[i] for i in keep] + [name]]
    for row in rows:
        table.append([_cell(row, i) for i in keep] + [_cell(row, index)])
    return keep, table


def _view_row(row, keep):
    """One untasted batch laid out to match a one-goal view."""
    return [_cell(row, i) for i in keep] + [None]


class _Guide(object):
    """What one goal looks like from the batches tasted so far.

    Holds step 2's cellar for that goal, the range of values tasting has
    actually turned up, and which way is better.  Everything it is asked --
    "what would this untasted batch score?", "how sure are we?" -- is answered
    by step 2 and step 3; this only puts the answers on a 0..1 footing where 1
    is always better, so two goals pulling opposite ways can be compared.
    """

    def __init__(self, header, readings, goal, rows):
        self.index, self.name, self.direction = goal
        self.keep, table = _view(header, readings, goal, rows)
        self.cellar = Cellar(table)
        known = [s for s in self.cellar.scores if s is not None]
        self.low = min(known) if known else 0.0
        self.high = max(known) if known else 0.0
        self.span = self.high - self.low

    def _oriented(self, value):
        """A goal value as 0..1 with 1 the better end, as step 1 does it."""
        if self.span <= 0.0:
            return 0.5
        fraction = (value - self.low) / self.span
        fraction = min(1.0, max(0.0, fraction))
        return fraction if self.direction > 0 else 1.0 - fraction

    def hope(self, row):
        """``(promise, doubt)`` for one untasted batch, both on the 0..1 footing.

        ``promise`` is what this goal is expected to come back as; ``doubt`` is
        how far apart the look-alike batches behind that expectation were.  A
        batch nothing in the glass so far resembles carries a large doubt, and
        that is what earns it a tasting.
        """
        view = _view_row(row, self.keep)
        promise = self._oriented(self.cellar.guess(view))
        if self.span <= 0.0:
            return promise, 0.0
        swing = spread(step3._neighbourhood(self.cellar, view))
        return promise, min(1.0, swing / self.span)


def _guides(header, readings, goals, rows):
    return [_Guide(header, readings, goal, rows) for goal in goals]


def _promise(guides, row):
    """How worth tasting one untasted batch looks: promise, flattered by doubt.

    Both goals count equally -- a wine that scores well and stays light is
    wanted, not one that wins on either alone -- and a batch we are unsure
    about is given the benefit of that doubt, which is what stops the plan
    pouring twenty glasses of the same thing.
    """
    hopes = [guide.hope(row) for guide in guides]
    promise = mean([p for p, _d in hopes]) or 0.0
    doubt = mean([d for _p, d in hopes]) or 0.0
    return promise + CURIOSITY * doubt


# ---------------------------------------------------------------- the lay of the land


def _warmup(budget):
    """How many of the month's glasses go on looking around before choosing."""
    share = int(round(budget * WARMUP_SHARE))
    return max(2, min(budget, share))


def _lay_of_the_land(points, count):
    """``count`` batches spread across the cellar, one from each kind of wine.

    Step 2 already knows how to sort a cellar into its few kinds without any
    scores at all -- it groups on the lab sheets alone.  Here that grouping is
    asked for exactly as many kinds as there are opening tastings, and the
    batch nearest the middle of each kind is the one poured: the most typical
    example of each sort of wine the cellar makes.
    """
    count = max(1, min(count, len(points)))
    if count == 1:
        middle = step2._centre(points)
        return [min(range(len(points)), key=lambda i: (step2._distance(points[i], middle), i))]

    labels = step2._settle(points, step2._seeds(points, count))
    picked = []
    for group in range(count):
        members = [i for i, label in enumerate(labels) if label == group]
        if not members:
            continue
        centre = step2._centre([points[i] for i in members])
        picked.append(min(members, key=lambda i: (step2._distance(points[i], centre), i)))
    return picked


def _points(data, readings):
    """Every lab sheet as a 0..1 point, scaled as in step 2."""
    scales = step2._scales(data, [i for i, _n in readings])
    return [step2._point(row, scales) for row in data]


def _shortlist(data, done, dice):
    """The untasted batches weighed this round."""
    pool = [i for i in range(len(data)) if i not in done]
    if len(pool) <= CANDIDATES:
        return pool
    return sorted(dice.sample(pool, CANDIDATES))


# ---------------------------------------------------------------- answers


def taste(rows, label, budget=BUDGET):
    """The batches to put in front of the panel, in the order to taste them.

    ``rows`` is a table in the step 1 shape -- header row first -- whose goal
    cells are all still ``'?'``.  ``label`` is one tasting: hand it a row and
    it returns that batch with its goals filled in.  ``label`` is called at
    most ``budget`` times, and never twice on the same batch.

    The first few glasses go on the lay of the land: one typical batch from
    each kind of wine the cellar makes, so the plan starts out knowing
    something about all of it rather than a lot about one corner.  Each glass
    after that is chosen on everything poured so far -- the most promising
    untasted batch, with an unsure guess counted as promising, which is how the
    plan keeps looking around instead of circling one good batch it has already
    found.

    Returns the labelled batches, in the order they were tasted.  The last ones
    in the list are the ones the plan grew most hopeful about; to pick a single
    winner out of them, see :func:`choose`.
    """
    header, data = split(rows)
    if not data or not budget or budget < 1:
        return []

    goals, readings = classify(header)
    if not goals or not readings:
        return []

    budget = min(int(budget), len(data))
    dice = random.Random(SEED)
    points = _points(data, readings)

    poured = []
    done = set()

    def pour(index):
        done.add(index)
        out = label(data[index])
        poured.append(list(out) if out is not None else list(data[index]))

    for index in _lay_of_the_land(points, _warmup(budget)):
        if len(poured) >= budget:
            break
        if index not in done:
            pour(index)

    while len(poured) < budget:
        pool = _shortlist(data, done, dice)
        if not pool:
            break
        guides = _guides(header, readings, goals, poured)
        best = max(pool, key=lambda i: (_promise(guides, data[i]), -i))
        pour(best)

    return poured


def shortlist(rows, label, budget=BUDGET, fraction=0.25):
    """The handful of tasted batches worth talking about, best first.

    The month's tastings are made by :func:`taste` and then judged by step 1's
    own reckoning of exceptional: the batches nothing else beats on both goals
    at once come first, ranked by how well they do across them.
    """
    header, _data = split(rows)
    poured = taste(rows, label, budget)
    if not poured:
        return []
    best = step1.remember([header] + poured, fraction=fraction)
    return best or poured[:1]


def choose(rows, label, budget=BUDGET):
    """The best batch found: high ``Quality+``, low ``Alcohol-``.

    Spends at most ``budget`` tastings -- the plan from :func:`taste` -- and
    then hands back the single best of the batches actually tasted, judged on
    both goals together by step 1's ``remember``.  So the answer is always a
    real batch with real goal values on it, never a guess: the winner has been
    in the glass.

    Returns ``None`` only when there is nothing to taste or no goal column to
    judge by.
    """
    header, _data = split(rows)
    poured = taste(rows, label, budget)
    if not poured:
        return None
    best = step1.remember([header] + poured)
    return best[0] if best else poured[0]


# ---------------------------------------------------------------- the why


def report(rows, label, budget=BUDGET):
    """The full story behind one :func:`choose`, for when the winemaker asks why.

    Returns the chosen batch, its goal values, how many tastings were spent,
    the handful it was picked out of, and -- because the plan can only speak
    for what it poured -- the goals of every batch tasted, so the month's work
    can be read off at a glance.
    """
    header, _data = split(rows)
    goals, _readings = classify(header)
    poured = taste(rows, label, budget)

    story = {
        "tasted": len(poured),
        "budget": int(budget),
        "batches": poured,
        "goals": [name for _i, name, _d in goals],
        "chosen": None,
        "chosen_goals": None,
        "why": "nothing to taste",
    }
    if not poured or not goals:
        return story

    best = step1.remember([header] + poured)
    pick = best[0] if best else poured[0]
    story["chosen"] = pick
    story["chosen_goals"] = {name: _cell(pick, index) for index, name, _d in goals}
    story["shortlist"] = step1.remember([header] + poured, fraction=0.25)
    story["why"] = "best of the %d batches tasted on %s together" % (
        len(poured),
        " and ".join(name for _i, name, _d in goals),
    )
    return story


def check(rows, budget=BUDGET):
    """How a month's budget did, measured against the whole cellar and against luck.

    The cellar is blinded with :func:`blank`, a month of tastings is spent on
    it, and the batch chosen is then looked up in the answers nobody was
    allowed to see.  Reports how good that batch was on both goals together,
    how good the very best batch on file was, where the choice sits among all
    the batches, and what the same number of glasses poured at random would
    have turned up -- because a plan is only worth having if it beats luck.
    """
    header, data = split(rows)
    goals, _readings = classify(header)
    empty = {"tasted": 0, "found": None, "best": None, "share": None, "luck": None}
    if not data or not goals:
        return empty

    points = step1._oriented(data, goals)
    scores = [mean(point) or 0.0 for point in points]

    worth = {}
    for row, score in zip(data, scores):
        key = tuple(_cell(row, i) for i in range(len(header)))
        worth[key] = max(score, worth.get(key, score))

    blind, label = blank(rows)
    pick = choose(blind, label, budget)
    if pick is None:
        return empty

    key = tuple(_cell(pick, i) for i in range(len(header)))
    found = worth.get(key)

    dice = random.Random(SEED)
    spend = min(int(budget), len(data))
    luck = []
    for _trial in range(TRIALS):
        drawn = dice.sample(range(len(data)), spend)
        luck.append(max(scores[i] for i in drawn))

    share = None
    if found is not None:
        share = sum(1 for score in scores if score <= found) / float(len(scores))

    return {
        "tasted": spend,
        "found": found,
        "best": max(scores),
        "share": share,
        "luck": mean(luck),
        "chosen": pick,
    }


# ---------------------------------------------------------------- demo


def _show(path):
    """Replay a finished cellar as though none of it had been tasted yet."""
    rows = step1.load_rows(path)
    header, data = split(rows)
    goals, _readings = classify(header)
    print("batches on file: %d" % len(data))
    print("goals: %s" % ", ".join(name for _i, name, _d in goals) if goals else "no goals")
    if not data or not goals:
        return

    blind, label = blank(rows)
    plan = taste(blind, label)

    print("\nthe %d batches to put in front of the panel this month, in order:" % len(plan))
    print("  %-4s %s" % ("#", "  ".join("%10s" % name for _i, name, _d in goals)))
    for turn, row in enumerate(plan, 1):
        cells = "  ".join(
            "%10s" % ("?" if _cell(row, index) is None else "%g" % _cell(row, index))
            for index, _name, _d in goals
        )
        print("  %-4d %s" % (turn, cells))

    best = step1.remember([header] + plan)
    if best:
        pick = best[0]
        print("\nthe batch to keep, out of those tasted:")
        for index, name, direction in goals:
            print("  %-16s %-8s (%s)" % (
                name,
                "?" if _cell(pick, index) is None else "%g" % _cell(pick, index),
                "maximize" if direction > 0 else "minimize",
            ))

    print("\nhow the month's %d glasses did against the whole cellar:" % BUDGET)
    report_ = check(rows)
    if report_["found"] is not None:
        print("  batch found scores %.3f on both goals together" % report_["found"])
        print("  the best batch on file scores %.3f" % report_["best"])
        print("  it beats or matches %.1f%% of every batch on file" % (100.0 * report_["share"]))
        print("  %d glasses poured at random would have found %.3f" % (report_["tasted"], report_["luck"]))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python step5.py <batches.csv>")
        raise SystemExit(2)
    _show(sys.argv[1])
