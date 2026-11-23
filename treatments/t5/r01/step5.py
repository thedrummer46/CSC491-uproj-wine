#!/usr/bin/env python3 -B
"""
step5.py: two more answers for the winemaker, built on ezr.

  taste(rows, label, budget=20) -> the batches to pour, in the
                                   order we poured them
  choose(rows, label, budget=20) -> the best batch found

Rows arrive as before: rows[0] is the header, `+` marks what to
maximize, `-` what to minimize, the rest are lab readings.  What
is new here is that the goal cells (Alcohol-, Quality+) start out
as '?' for every batch.  `label(row)` hands back that batch with
its goals filled in, and each call is one tasting.  Tastings are
the scarce thing: 20 of them, over 1279 batches.

So both answers are the same hunt, run on a budget.  ezr already
scores a tasted batch for us: `ydist` is how far its goals sit
from the best they could be, over Quality+ and Alcohol- at once,
so "high quality, low alcohol" needs no weights from us -- small
ydist is the thing we want, and the two requests above are one
request.

How the hunt goes.  Four tastings to open, chosen to sit far
apart on the lab sheet, so we start with a view of the whole
cellar rather than four neighbours.  After that, every tasting
is aimed: sort what we have tasted by ydist, call the top
sqrt(n) of it `best` and the remainder `rest`, take the middle
of each, and pour the untasted batch that sits nearest the
first middle and farthest from the second.  This is ezr's
`acquire` strategy, which we re-run here rather than call,
for two reasons: `acquire` looks at `the.Few`=128 candidate
rows, and with so few tastings to spend we would rather
consider all 1279; and the oracle has to be ours, so that a
batch is poured once, by us, and counted.

What never happens: no goal cell is read before its batch is
tasted.  The lab readings of every batch are free (they are on
the sheet already), so the x columns are summarized over all
1279 rows and the distances above use that; the y columns are
summarized over the tasted rows only, and that is the only
place a panel score is ever read.

The difference between the two answers is only when to stop.
The panel asked for 20 slots, so `taste` spends all 20 and the
list it returns is the list to pour.  `choose` was asked to
taste as few batches as possible, so it quits early once the
hunt has gone quiet -- PATIENCE tastings with no new best, and
never before SOON of the budget is gone -- and returns the best
batch it holds.

Speed.  One tasting costs one scan of the untasted batches (two
distances each) plus a sort of the handful we have tasted; 20
tastings is well under a second, and the oracle is called at
most `budget` times, so the whole workload is bounded by the
caller's own labelling cost.
"""
import sys, time, random
from math import sqrt, floor
import ezr

START    = 4     # opening tastings, spread across the cellar
PATIENCE = 6     # choose: quit after this many dull tastings
SOON     = 0.6   # choose: but never before this much budget

#-- the batches --------------------------------------------
def _header(rows: ezr.ROWS) -> list:
  "The header row, or a complaint."
  if not rows or not rows[0]: raise ValueError("step5: no rows")
  if not all(isinstance(s, str) for s in rows[0]):
    raise ValueError("step5: rows[0] must be the header")
  return rows[0]

def _table(rows: ezr.ROWS) -> ezr.TBL:
  """A TBL over every batch.  Its x columns are real: they come
  off the lab sheet.  Its y columns are empty until we taste."""
  return ezr.Tbl([_header(rows)] + list(rows[1:]))

def _tasted(tbl: ezr.TBL, row: ezr.ROW) -> bool:
  "Has the panel filled in every goal of this batch?"
  for c in tbl.y:
    v = row[c.at] if c.at < len(row) else "?"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
      return False
  return True

def _ask(label: ezr.ORACLE, row: ezr.ROW) -> ezr.ROW:
  "One tasting.  Keep whatever the oracle hands back."
  out = label(row)
  return row if out is None else out

#-- which batch next ---------------------------------------
def _ranked(tbl: ezr.TBL, got: ezr.ROWS) -> tuple:
  """What we have tasted, best first, plus a TBL that knows those
  scores (and so can normalize them)."""
  lab  = ezr.clone(tbl, got)
  rows = [r for r in got if _tasted(lab, r)]
  rows.sort(key=lambda r: ezr.ydist(lab, r))
  return lab, rows

def _far(tbl: ezr.TBL, todo: ezr.ROWS, got: ezr.ROWS) -> int:
  "The untasted batch sitting farthest from all we have poured."
  out, best = 0, -1
  for j, r in enumerate(todo):
    d = min(ezr.xdist(tbl, r, p) for p in got)
    if d > best: out, best = j, d
  return out

def _next(tbl: ezr.TBL, todo: ezr.ROWS, rows: ezr.ROWS) -> int:
  """The untasted batch most like our best tastings and least
  like the rest of them."""
  n  = max(1, int(sqrt(len(rows))))
  cb = ezr.mids(ezr.clone(tbl, rows[:n]))
  cr = (ezr.mids(ezr.clone(tbl, rows[n:]))
        if len(rows) > n else None)
  def key(r: ezr.ROW) -> float:
    near = ezr.xdist(tbl, r, cb)
    return ezr.xdist(tbl, r, cr) - near if cr else -near
  return max(range(len(todo)), key=lambda j: key(todo[j]))

#-- the hunt -----------------------------------------------
def _hunt(rows: ezr.ROWS, label: ezr.ORACLE, budget: int,
          patience: int | None = None) -> tuple:
  """Spend up to BUDGET tastings hunting small ydist.  With
  PATIENCE, stop once the hunt goes quiet."""
  tbl    = _table(rows)
  data   = list(tbl.rows)
  budget = min(int(budget), len(data))
  if budget <= 0: return tbl, []
  todo = random.sample(data, len(data))   # no order to trust
  got  = [_ask(label, todo.pop())]
  while todo and len(got) < min(START, budget):
    got += [_ask(label, todo.pop(_far(tbl, todo, got)))]
  dull = 0
  while todo and len(got) < budget:
    _, best = _ranked(tbl, got)
    if not best:               # nothing came back scored yet
      got += [_ask(label, todo.pop())]
      continue
    dull = 0 if best[0] is got[-1] else dull + 1
    if (patience and dull >= patience
        and len(got) >= floor(SOON * budget)): break
    got += [_ask(label, todo.pop(_next(tbl, todo, best)))]
  return tbl, got

#-- taste --------------------------------------------------
def taste(rows: ezr.ROWS, label: ezr.ORACLE,
          budget: int = 20) -> ezr.ROWS:
  """The batches to pour for the panel, in the order to pour
  them: each one chosen in the light of the ones before it.
  Calls LABEL at most BUDGET times, and returns those tasted
  batches with their goals filled in."""
  return _hunt(rows, label, budget)[1]

#-- choose -------------------------------------------------
def choose(rows: ezr.ROWS, label: ezr.ORACLE,
           budget: int = 20) -> ezr.ROW:
  """The best batch we can find -- high Quality+, low Alcohol- --
  while pouring as little as we can get away with.  Never more
  than BUDGET tastings, often fewer."""
  tbl, got = _hunt(rows, label, budget, PATIENCE)
  if not got:                  # no budget: nothing was tasted
    if not tbl.rows: raise ValueError("step5: no batches")
    return tbl.rows[0]
  _, best = _ranked(tbl, got)
  return best[0] if best else got[0]

#-- demo ---------------------------------------------------
def _hide(rows: ezr.ROWS) -> tuple:
  "Blank every goal cell, and keep the oracle that fills it in."
  names = _header(rows)
  ys    = [i for i, s in enumerate(names) if s[-1] in "+-!"]
  truth, blind, calls = {}, [], [0]
  for row in rows[1:]:
    b = list(row)
    for i in ys: b[i] = "?"
    truth[id(b)] = row
    blind += [b]
  def label(row: ezr.ROW) -> ezr.ROW:
    calls[0] += 1
    was, out = truth.get(id(row)), list(row)
    if was:
      for i in ys: out[i] = was[i]
    return out
  return [names] + blind, label, calls

def main(args: list[str]) -> None:
  "step5.py [file.csv] -- hide the panel's notes, then hunt."
  rows = list(ezr.csv(args[0] if args else ezr.the.File))
  random.seed(ezr.the.Seed)
  full = ezr.Tbl(rows)
  ys   = sorted(ezr.ydist(full, r) for r in full.rows)
  N    = len(full.rows)
  beat = lambda y: sum(1 for v in ys if v < y)
  goals = [i for i, s in enumerate(rows[0]) if s[-1] in "+-!"]

  blind, label, calls = _hide(rows)
  t0  = time.time()
  got = taste(blind, label)
  print(f"# taste: {len(got)} batches of {N},"
        f" {calls[0]} tastings, in {time.time()-t0:.1f}s")
  best = min(got, key=lambda r: ezr.ydist(full, r))
  y    = ezr.ydist(full, best)
  print(f"  best poured: ydist {ezr.say(y,3)},"
        f" beaten by {beat(y)} of {N}")
  print("   ", ezr.say({rows[0][i]: best[i] for i in goals}, 2))

  blind, label, calls = _hide(rows)
  t1   = time.time()
  pick = choose(blind, label)
  y    = ezr.ydist(full, pick)
  print(f"# choose: {calls[0]} tastings"
        f" in {time.time()-t1:.1f}s")
  print(f"  pick: ydist {ezr.say(y,3)},"
        f" beaten by {beat(y)} of {N}")
  print("   ", ezr.say({rows[0][i]: pick[i] for i in goals}, 2))

  tries = [min(ezr.ydist(full, r)
               for r in random.sample(full.rows, 20))
           for _ in range(20)]
  print(f"# as if 20 poured at random: best ydist"
        f" {ezr.say(sum(tries)/len(tries),3)} (mean of 20 runs)")
  print(f"  the whole cellar: best {ezr.say(ys[0],3)},"
        f" middle {ezr.say(ys[N//2],3)}")

if __name__ == "__main__": main(sys.argv[1:])
