#!/usr/bin/env python3 -B
"""
step5.py: a winemaker's assistant, built on ezr.py, step1..step4.

Two questions from a month with a budget:

  "The panel can only taste 20 batches this month.  Which 20
   should they taste to find our best wine?"
      -> taste(rows, label, budget=20)

  "Find me a batch that scores high but stays low in alcohol,
   while tasting as few batches as possible."
      -> choose(rows, label, budget=20)

ROWS is [header, *data], as in step1..step4: the header names the
columns (trailing "+" = maximize, "-" = minimize, the rest are
lab readings).  What is new here is that the goal cells of every
data row -- Alcohol- and Quality+ -- arrive as "?".  The only way
to fill them is

      label(row) -> the same batch, goals filled in

and each call is one tasting.  So a label is the scarce thing,
and both answers are measured in labels spent, not in CPU.

Why not predict the score, as step2..step4 do?  Because nothing
is scored yet.  With an empty cellar the question is not "what
will this batch score" but "which batch should we open next",
and that is exactly ezr's `acquire`: label a few at random, sort
them by `ydist` (distance from a row's goals to the ideal corner
-- Quality high, Alcohol low), keep a small `best` table and a
larger `rest` one, and then always open the untasted batch that
sits nearest best's centroid and furthest from rest's.  Each
tasting moves those two centroids, so the next pick is already
better informed.  One goal column or two makes no difference:
`ydist` folds Alcohol- and Quality+ into one number, so "high
score, low alcohol" needs no extra machinery.

`taste` hands the whole budget to `acquire` and reports the
batches in the order they were opened: the early ones are the
random probes that map the cellar, the later ones are the picks,
so the list reads as a tasting flight.

`choose` wants one batch, not a flight, so it holds CHECK labels
back.  It spends the rest the same way, grows ezr's tree on what
it learned (`ranker`), walks every *untasted* batch down that
tree, and opens the few that land in the lowest-ydist leaf.  That
last probe reaches rows the steering never got to, and the batch
it returns is always one we actually tasted -- a verdict with a
score behind it, not a guess.

Two honest limits.  Scores are normalized against the labelled
rows only (they are the only scores in existence), so "best" here
means best of what this budget could see.  And the candidate pool
per step is capped at POOL rows, which keeps one step O(POOL) no
matter how big the cellar gets.

Costs: `budget` calls to label, plus (budget - START) sorts of at
most POOL rows over the x columns, plus one small tree.  On 1279
rows with a budget of 20 that is a fraction of a second.
"""
# pylint: disable=invalid-name

import sys
from random import Random
from typing import Any

import ezr
from ezr import (Tbl, clone, acquire, grabs, ranker, ydist,
                 wins, o, the, say)
import step1
from step1 import names, data
import step2
from step2 import poke
import step3
from step3 import isHeader
from step4 import PANEL

#-- types --------------------------------------------------
type ROW    = list[Any]
type ROWS   = list[ROW]
type ORACLE = Any        # Callable[[ROW], ROW]

#-- dials --------------------------------------------------
START = 4      # random tastings before we start steering
CHECK = 3      # labels held back for `choose`'s last probe
POOL  = 4096   # most candidates weighed per tasting

#-- rows in ------------------------------------------------
def prep(rows: ROWS) -> tuple[list[str], ROWS]:
  "ROWS as (header, data).  The header carries the markers."
  rows = rows or []
  if len(rows) < 2: return (names(rows) if rows else []), []
  return names(rows), [r for r in data(rows) if not isHeader(r)]

def goals(ns: list[str]) -> list[int]:
  "Which columns the panel owns: the ones we must pay to see."
  return [at for at, s in enumerate(ns) if s[-1:] in "+-!"]

def xkey(row: ROW, ats: set[int]) -> tuple:
  "A batch's lab sheet: what stays the same after a tasting."
  return tuple(v for i, v in enumerate(row) if i not in ats)

def done(row: ROW, ats: list[int]) -> bool:
  "Has every goal cell come back from the panel?"
  return all(not isinstance(row[at], bool)
             and isinstance(row[at], (int, float))
             for at in ats if at < len(row))

#-- the month's budget -------------------------------------
class Spent(Exception):
  "Raised when the last tasting is used up."

def budgeted(label: ORACLE, budget: int, log: ROWS) -> ORACLE:
  """LABEL, but it counts, keeps its answers, and then stops.

  The stop is an exception, not a shrug: a silent "?" would feed
  an unlabelled row into `ydist` and quietly poison the search.
  """
  def fn(row: ROW) -> ROW:
    if len(log) >= budget: raise Spent()
    out = label(row)
    log.append(row if out is None else out)
    return log[-1]
  return fn

def steer(tbl: Any, cap: int, fn: ORACLE, pool: int) -> None:
  "Spend CAP labels the way `acquire` wants to spend them."
  if cap < 1: return
  with poke(Stop=cap, Start=max(1, min(START, cap)),
            Few=max(1, min(pool, POOL))):
    acquire(tbl, cap, fn)

#-- request 1: the month's tasting flight ------------------
def taste(rows: ROWS, label: ORACLE, budget: int = 20) -> ROWS:
  """The batches to open this month, in the order to open them.

  Calls LABEL at most BUDGET times and returns exactly what came
  back: the first few are random probes, the rest are chosen in
  the light of everything tasted before them.
  """
  ns, d = prep(rows)
  if not ns or not d or budget < 1: return []
  budget = min(budget, len(d))
  log: ROWS = []
  try: steer(Tbl([ns] + d), budget,
             budgeted(label, budget, log), len(d))
  except Spent: pass
  return log

#-- request 2: one batch, high score and low alcohol -------
def probe(tbl: Any, d: ROWS, ns: list[str], log: ROWS,
          fn: ORACLE, k: int) -> None:
  "Spend the last K labels on the tree's best untasted batches."
  if k < 1 or len(log) <= the.Leaf: return
  ats  = set(goals(ns))
  seen = {xkey(r, ats) for r in log}
  todo = [r for r in d if xkey(r, ats) not in seen][:POOL]
  if not todo: return
  todo.sort(key=ranker(clone(tbl, log)))   # low ydist first
  for r in todo[:k]: fn(r)

def best(tbl: Any, ns: list[str], log: ROWS) -> ROW | None:
  "Of the batches we tasted, the one nearest the ideal corner."
  ats = goals(ns)
  got = [r for r in log if done(r, ats)]
  if not got: return log[0] if log else None
  lab = clone(tbl, got)                    # scores we have paid for
  return min(got, key=lambda r: ydist(lab, r))

def choose(rows: ROWS, label: ORACLE,
           budget: int = 20) -> ROW | None:
  """The best batch found inside BUDGET tastings.

  Most of the budget steers toward the good corner; CHECK labels
  are held back to open the top picks of a tree grown on what
  the steering learned.  The batch returned has been tasted, so
  its Quality+ and Alcohol- are real numbers, not estimates.
  """
  ns, d = prep(rows)
  if not ns or not d or budget < 1: return None
  budget = min(budget, len(d))
  k      = CHECK if budget >= 3 * CHECK else 0
  log: ROWS = []
  fn     = budgeted(label, budget, log)
  tbl    = Tbl([ns] + d)
  try:
    steer(tbl, budget - k, fn, len(d))
    probe(tbl, d, ns, log, fn, budget - len(log))
  except Spent: pass
  return best(tbl, ns, log)

#-- show me ------------------------------------------------
def eg_h() -> None:
  "Print help."
  print(__doc__)

def Cellar(ns: list[str], d: ROWS) -> tuple[ROWS, ORACLE, Any]:
  "The cellar, blinded: '?' goals, plus an oracle that counts."
  ats   = goals(ns)
  aset  = set(ats)
  blind, truth = [], {}
  for r in d:
    b = list(r)
    for at in ats: b[at] = "?"
    blind += [b]
    truth[id(b)] = r
  byKey = {xkey(r, aset): r for r in d}
  tally = o(n=0)
  def label(row: ROW) -> ROW:
    tally.n += 1
    full = truth.get(id(row)) or byKey[xkey(row, aset)]
    out  = list(row)
    for at in ats: out[at] = full[at]
    return out
  return blind, label, tally

def eg_all() -> None:
  "Blind ezr's -File, then spend 20 tastings on it, two ways."
  rows  = list(ezr.csv(ezr.the.File))
  ns, d = prep(rows)
  ats   = goals(ns)
  truth = Tbl([ns] + d)              # for marking only
  w     = wins(truth)                # 100 = best, 0 = average
  cells = ", ".join(ns[at] for at in ats)
  print(f"\n-- {len(d)} batches, {cells} hidden until tasted ----")

  blind, label, tally = Cellar(ns, d)
  got = taste([ns] + blind, label, 20)
  print(f"\n-- taste: the month's flight ({tally.n} tastings) ----")
  print("     #   wins   " + "   ".join(ns[at] for at in ats))
  seen = -1e30
  for i, r in enumerate(got, 1):
    mark = "*" if w(r) > seen else " "
    seen = max(seen, w(r))
    print(f"  {i:>4} {say(w(r),1):>6} {mark}  " +
          "   ".join(f"{say(r[at],2):>8}" for at in ats))

  blind, label, tally = Cellar(ns, d)
  pick = choose([ns] + blind, label, 20)
  print(f"\n-- choose: one batch ({tally.n} tastings) ----------")
  print("  " + ",".join(str(s) for s in ns))
  print("  " + ",".join(say(v, 3) for v in pick))
  for at in ats: print(f"  {ns[at]:>16} : {say(pick[at],2)}")
  print(f"  {'wins':>16} : {say(w(pick),1)} "
        f"(100 = the cellar's best, 0 = average)")
  print(f"  {'to the panel?':>16} : "
        f"{'yes' if pick[ns.index(ns[ats[-1]])] >= PANEL else 'no'}"
        f"   (step4's bar is {say(PANEL,1)})")

  blind, label, tally = Cellar(ns, d)
  tbl  = Tbl([ns] + blind)
  dumb = best(tbl, ns, grabs(tbl, 20, label))
  print(f"\n-- the straw man: 20 at random ---------------------")
  print(f"  wins {say(w(dumb),1)} from {tally.n} tastings")

if __name__ == "__main__":
  ezr.main(sys.argv[1:], globals())
