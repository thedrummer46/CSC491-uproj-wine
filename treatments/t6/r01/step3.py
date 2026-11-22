#!/usr/bin/env python3 -B
"""
step3.py: two answers about trust, built on ezr.py, step1.py
and step2.py.

  certify(train, a, b) -> 'A', 'B' or 'same'
      Which of two batches will score better -- or 'same',
      when the gap between them is only tasting noise.
  accuracy(train, test) -> float
      Share of unseen batches whose guessed score rounds to
      the true Quality+.

Rows arrive as a list of lists: rows[0] is the header of
strings, the rest are numbers.  A name ending in `+` is a goal
to maximize, `-` a goal to minimize, the rest are lab readings.

Where the noise comes from
--------------------------
step2 guesses a batch by walking it down ezr's tree and
reporting the average Quality+ of the leaf it lands in.  That
leaf is not one number: it is a pocket of real batches whose
lab sheets agree, and the panel scored them over a range.  So
the leaf *is* the noise estimate.  Two batches that land in
the same leaf have no story to tell them apart at all; two
that land in different leaves differ only as much as those two
pockets of scores differ.

So `certify` asks ezr's own statistics -- `same`, which is
`cliffs` and `ks` together, plus Cohen's small-effect test --
whether the two leaves' score distributions are
distinguishable.  If they are not, the honest answer is
'same'.  If they are, the higher-scoring leaf wins.  Quality+
is maximized, so higher is better; were the panel's column a
`-` goal, the comparison flips (see `better`).

How often the guesses are right
-------------------------------
`accuracy` is deliberately strict in one way and forgiving in
another.  Strict: a batch counts only if the guess rounds to
exactly the panel's score, which for the usual integer wine
scores means "named the right score".  Forgiving: `train` and
`test` are kept apart, so every batch counted was never seen
while the tree was grown.  Pass disjoint row lists; the
function does not check that for you, and will happily report
a flattering number if you hand it the same rows twice.

Usage:  python3 step3.py [wines.csv]
"""
# pylint: disable=bad-indentation,invalid-name

import sys, random
from ezr import Tbl, tree, leaf, leafs, same, adds, div, say, csv, the
from step1 import label
from step2 import scored, expected, guess

#-- settings -----------------------------------------------
Eps   = 0.35   # Cohen effect size: how big a gap must be to count
Split = 0.5    # for the demo: share of the cellar used to train

#-- the panel's scores inside one leaf ---------------------
def scores(rows: list[list], at: int) -> list:
  "The panel's scores for ROWS, skipping unknowns (`?`)."
  return [r[at] for r in rows if r[at] != "?"]

def better(name: str) -> int:
  "+1 if a higher score is better for this column, else -1."
  return -1 if name[-1] == "-" else 1

#-- answers ------------------------------------------------
def certify(train: list[list], a: list, b: list) -> str:
  """'A', 'B', or 'same': which batch will score better.

  Grow ezr's tree on TRAIN and walk both batches down it on
  their lab readings.  Each lands in a leaf; that leaf's panel
  scores are the spread we expect around our guess.  Then ask
  ezr's `same` -- cliffs delta, Kolmogorov-Smirnov, and a
  Cohen check at EPS -- whether those two spreads can be told
  apart.  No: 'same', the difference is tasting noise.  Yes:
  the batch with the better expected score.

  Two batches in the same leaf always come back 'same': the
  lab sheets gave us nothing to separate them by.  Neither
  batch needs a panel score of its own; only `tbl.x` columns
  steer the walk."""
  tbl  = Tbl(train)
  at   = scored(tbl)
  root = tree(tbl, tbl.rows)
  la, lb = leaf(root, a), leaf(root, b)
  if la is lb: return "same"
  xs, ys = scores(la.rows, at), scores(lb.rows, at)
  if not xs or not ys: return "same"
  if same(xs, ys, eps=Eps): return "same"
  sign = better(tbl.names[at])
  ma, mb = expected(la.rows, at), expected(lb.rows, at)
  if ma == mb: return "same"
  return "A" if sign * (ma - mb) > 0 else "B"

def accuracy(train: list[list], test: list[list]) -> float:
  """Share of TEST batches whose guessed score rounds to the
  panel's score.  0..1; 1 means every guess named the right
  score.

  TRAIN and TEST are both full row lists, header first.  The
  tree is grown once on TRAIN, then every TEST batch with a
  panel score is walked down it.  Batches whose score the
  panel never gave (`?`) are not counted -- there is nothing
  to be right about.  An empty test set scores 0."""
  tbl  = Tbl(train)
  at   = scored(tbl)
  root = tree(tbl, tbl.rows)
  hits = n = 0
  for row in test[1:]:
    if at >= len(row) or row[at] == "?": continue
    n   += 1
    hits += round(expected(leaf(root, row).rows, at)) == round(row[at])
  return hits / n if n else 0.0

#-- how wide is a guess? -----------------------------------
def spread(train: list[list], row: list) -> float:
  "Standard deviation of the panel scores around one guess."
  tbl = Tbl(train)
  at  = scored(tbl)
  return div(adds(scores(leaf(tree(tbl, tbl.rows), row).rows, at)))

#-- reporting ----------------------------------------------
def split(rows: list[list], at: float = Split) -> tuple:
  "Shuffle the cellar into (train, test), header on both."
  data = random.sample(rows[1:], len(rows) - 1)
  n    = int(len(data) * at)
  return [rows[0]] + data[:n], [rows[0]] + data[n:]

def blind(names: list, row: list) -> list:
  "One batch as it arrives: lab sheet only, no panel score."
  from step1 import goalp
  return [("?" if goalp(nm) else v) for nm, v in zip(names, row)]

def pairs(groups: list, k: int = 4) -> list:
  "A few (kind i, kind j) pairs: far apart, then close together."
  out = []
  for i in range(min(k, len(groups) - 1)):
    out += [(0, len(groups) - 1 - i)]        # best against worst
  for i in range(min(k, len(groups) - 1)):
    out += [(i, i + 1)]                      # neighbour kinds
  return [(i, j) for i, j in out if i != j]

def report(rows: list[list]) -> None:
  "Print both answers for one cellar."
  names = rows[0]
  tbl   = Tbl(rows)
  at    = scored(tbl)
  name  = label(names[at])
  train, test = split(rows)

  print(f"\n# Right on batches never seen   "
        f"({len(train)-1} to learn from, {len(test)-1} held back)\n")
  acc = accuracy(train, test)
  print(f"  the panel scores : {names[at]}")
  print(f"  exactly right    : {say(100*acc,1)}% of held-back batches")
  flat = round(expected(train[1:], at))
  dumb = [r for r in test[1:] if r[at] != "?" and round(r[at]) == flat]
  base = len(dumb) / max(1, len([r for r in test[1:] if r[at] != "?"]))
  print(f"  (always saying {flat} would be right "
        f"{say(100*base,1)}% of the time)")
  print(f"\n  Rounded to the nearest whole score: 'exactly right'"
        f"\n  means we named the score the panel gave.")

  from step2 import kinds
  groups = kinds(train)
  print(f"\n# Real difference, or tasting noise?   "
        f"({len(groups)} kinds of batch)\n")
  print(f"  {'A':>4} {'B':>4}   {'expect A':>9} {'expect B':>9}"
        f"   {'verdict':>8}")
  for i, j in pairs(groups):
    a = blind(names, groups[i][0][0])
    b = blind(names, groups[j][0][0])
    v = certify(train, a, b)
    said = {"A": "A wins", "B": "B wins", "same": "noise"}[v]
    print(f"  {i+1:>4} {j+1:>4}   {say(groups[i][1],2):>9}"
          f" {say(groups[j][1],2):>9}   {said:>8}")
  print("\n  'noise' means the two pockets of panel scores overlap"
        "\n  too much to call -- cliffs delta, KS, and Cohen all"
        f"\n  agree, at an effect size of {Eps}.")

  big = groups[0][0][0]
  print(f"\n# One batch, read honestly\n")
  print(f"  guess  : {say(guess(train, blind(names, big)),3)} {name}")
  print(f"  give or take {say(spread(train, blind(names, big)),3)}"
        f" (the spread inside its kind)")
  print(f"  panel said {say(big[at],3)}")

def main(args: list[str]) -> None:
  "Read a CSV -- named on the command line, or ezr's default."
  random.seed(the.Seed)
  file = args[0] if args else the.File
  rows = list(csv(file))
  if len(rows) < 2:
    print(f"{file}: no data rows found."); return
  print(f"file: {file}")
  report(rows)

if __name__ == "__main__": main(sys.argv[1:])
