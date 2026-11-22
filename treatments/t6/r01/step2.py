#!/usr/bin/env python3 -B
"""
step2.py: two more answers for the winemaker, built on ezr.py
and step1.py.

  guess(train, row) -> float
      The score the panel will probably give one new batch,
      read from its lab sheet alone.
  kinds(train) -> [(group_rows, expected_quality), ...]
      The few kinds of batch in the cellar, each with the
      score to expect from it.

Rows arrive as a list of lists: rows[0] is the header of
strings, the rest are numbers.  A name ending in `+` is a goal
to maximize, `-` a goal to minimize, the rest are lab readings.

How the guessing works
----------------------
ezr's `tree` splits rows on their lab readings, choosing cuts
whose two sides have the tightest goals.  So rows that land in
the same leaf have similar lab sheets *and* similar panel
scores.  To guess a new batch, walk it down that tree and
report the average Quality+ of the batch's leaf.  That is the
whole method: no new maths, just a reading of ezr's tree over
the column the winemaker asked about.

Two cautions, both deliberate:

  * Only the lab readings steer the walk.  A new batch has no
    panel score yet, and `tree` only ever tests `tbl.x`
    columns, so nothing about Quality+ is needed to place a
    row.  The score comes back out of the leaf, never into it.
  * A leaf's average is the honest answer, not a precise one.
    `kinds` hands back each group's rows so the winemaker can
    see the spread behind the number.

Usage:  python3 step2.py [wines.csv]
"""
# pylint: disable=bad-indentation,invalid-name

import sys
from ezr import Tbl, tree, leaf, leafs, show, say, csv, the
from step1 import goalp, label, which  # reading the header

#-- settings -----------------------------------------------
Goal = "Quality+"   # the column the panel scores; see `scored`

#-- finding the panel's column -----------------------------
def scored(tbl, want: str = Goal) -> int:
  """Which column holds the panel's score?

  The named one, if the cellar has it; else the first goal to
  maximize; else the first goal of any kind.  Written this way
  so a CSV with a differently spelled header still works."""
  names = tbl.names
  for at, name in enumerate(names):
    if name == want: return at
  for at, name in enumerate(names):
    if name[-1] == "+": return at
  for at, name in enumerate(names):
    if goalp(name): return at
  raise ValueError("no goal column in the header")

def expected(rows: list[list], at: int) -> float:
  """Mean of column AT over ROWS, skipping unknowns (`?`).
  Returns 0 for an empty group, which `kinds` never makes."""
  vs = [r[at] for r in rows if r[at] != "?"]
  return sum(vs) / len(vs) if vs else 0

#-- answers ------------------------------------------------
def guess(train: list[list], row: list) -> float:
  """The score to expect from one new batch.

  Grow ezr's tree on TRAIN, walk ROW down it on lab readings
  alone, and return the average panel score of the batch it
  lands beside.  ROW may be short or padded with `?` in its
  goal cells: nothing here reads them."""
  tbl  = Tbl(train)
  at   = scored(tbl)
  node = leaf(tree(tbl, tbl.rows), row)
  return expected(node.rows, at)

def kinds(train: list[list]) -> list:
  """The kinds of batch in this cellar, best first.

  Each leaf of the tree is one kind: a pocket of batches whose
  lab sheets agree.  We return (rows, expected score) per kind,
  sorted by that score, highest first -- for Quality+ that is
  best kind first."""
  tbl   = Tbl(train)
  at    = scored(tbl)
  out   = [(k.rows, expected(k.rows, at))
           for k in leafs(tree(tbl, tbl.rows)) if k.rows]
  return sorted(out, key=lambda z: -z[1])

#-- checking the guesses -----------------------------------
def errors(rows: list[list], folds: int = 5) -> dict:
  """How wrong is `guess`?  Hold out every fifth batch in
  turn, guess it from the rest, and compare with the panel.

  Reported beside the error of always saying the cellar mean,
  so the winemaker can see whether the lab sheet helped."""
  names, data = rows[0], rows[1:]
  at   = scored(Tbl(rows))
  mine, dumb = [], []
  for f in range(folds):
    test  = [r for i, r in enumerate(data) if i % folds == f]
    train = [r for i, r in enumerate(data) if i % folds != f]
    if not test or len(train) < 2 * the.Leaf: continue
    flat = expected(train, at)
    tbl  = Tbl([names] + train)
    node = tree(tbl, tbl.rows)
    for r in test:
      if r[at] == "?": continue
      mine += [abs(expected(leaf(node, r).rows, at) - r[at])]
      dumb += [abs(flat - r[at])]
  n = len(mine) or 1
  return o_(seen=len(mine), mine=sum(mine)/n, dumb=sum(dumb)/n)

def o_(**kw) -> dict:
  "A plain dict, spelled like ezr's `o` but without the dots."
  return dict(**kw)

#-- reporting ----------------------------------------------
def report(rows: list[list]) -> None:
  "Print both answers for one batch list."
  names = rows[0]
  tbl   = Tbl(rows)
  at    = scored(tbl)
  name  = label(names[at])
  n     = len(rows) - 1

  print(f"\n# Guessing {name} from the lab sheet   "
        f"({n} batches in the cellar)\n")
  print(f"  the panel scores : {names[at]}   ({which(names[at])})")
  print(f"  cellar average   : {say(expected(rows[1:], at), 3)}")
  e = errors(rows)
  if e["seen"]:
    print(f"  average error    : {say(e['mine'],3)} "
          f"per batch, held out {e['seen']} times")
    print(f"  (always guessing the average is off by "
          f"{say(e['dumb'],3)})")
  else:
    print("  too few batches to check the guessing.")

  groups = kinds(rows)
  print(f"\n# The kinds of batch   ({len(groups)} kinds, "
        f"best {name} first)\n")
  print(f"  {'kind':>4} {'batches':>8} {'expect '+name:>18}")
  for i, (group, score) in enumerate(groups):
    print(f"  {i+1:>4} {len(group):>8} {say(score,3):>18}")
  print("\n  (each kind's own rows are the return value of kinds())")

  print(f"\n# How the kinds were told apart\n")
  show(tbl, tree(tbl, tbl.rows))
  print("\n  (ydist is distance to the ideal mix of all goals:"
        " small is good)")

  worst, best = groups[-1][0][0], groups[0][0][0]
  print(f"\n# Two worked guesses\n")
  for tag, row in (("a batch of the best kind ", best),
                   ("a batch of the worst kind", worst)):
    blind = [("?" if goalp(nm) else v) for nm, v in zip(names, row)]
    print(f"  {tag}: guess {say(guess(rows, blind),3):>8}"
          f"   panel said {say(row[at],3)}")

def main(args: list[str]) -> None:
  "Read a CSV -- named on the command line, or ezr's default."
  file = args[0] if args else the.File
  rows = list(csv(file))
  if len(rows) < 2:
    print(f"{file}: no data rows found."); return
  print(f"file: {file}")
  report(rows)

if __name__ == "__main__": main(sys.argv[1:])
