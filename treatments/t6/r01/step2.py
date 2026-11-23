#!/usr/bin/env python3 -B
"""
step2.py: a winemaker's assistant, built on ezr.py and step1.py.

Two more questions from the cellar:

  "Before the panel sits, estimate what score this new batch
   will get from its lab sheet alone."
      -> guess(train, row)

  "Group our batches into a few kinds, and tell me what score
   to expect from each kind."
      -> kinds(train)

Both take TRAIN = [header, *data], as in step1: the header names
the columns (trailing "+" = maximize, "-" = minimize, the rest
are lab readings) and every later row holds numbers.

What counts as the answer?  The panel scores "Quality+", so that
is the one column we predict.  Everything else on the sheet --
including "Alcohol-", which is a goal but is also just a reading
the lab hands us -- is an input.  So we re-label the header: the
score keeps its "+" (ezr's one goal here), and every other name
loses its marker and becomes an x column.

How?  ezr's `tree` already splits rows so that each side has the
tightest goals; with one goal, that is a regression tree on the
score.  We grow a few of them on bootstrap samples (bagging, to
calm a single tree's twitchiness), hang the mean score on each
leaf, and average.  Leaf means are nudged toward the global mean
so a four-row leaf cannot shout.

`kinds` grows one more tree, this time with a leaf size of about
n/8, so the cellar falls into a handful of readable groups rather
than dozens.  Each group reports the score to expect from it.
"""
# pylint: disable=invalid-name

import sys
from typing import Any
from contextlib import contextmanager
from random import Random

import ezr
from ezr import Tbl, clone, o, the, tree, leaf, leafs, say
import step1
from step1 import names, data

#-- types --------------------------------------------------
type ROW   = list[Any]
type ROWS  = list[ROW]
type MODEL = o           # o(at, n, mu, lo, hi, tbl, trees)

#-- dials --------------------------------------------------
TREES = 8      # bagged trees behind one guess
PRIOR = 1.0    # pseudo-rows of global mean, added to each leaf
KINDS = 8      # aim for about this many groups in `kinds`

#-- header, re-read ----------------------------------------
def goalAt(ns: list[str]) -> int:
  "Which column does the panel score?  Quality+, given a choice."
  ups = [at for at, s in enumerate(ns) if s[-1:] == "+"]
  for at in ups:
    if "quality" in ns[at].lower(): return at
  if ups: return ups[-1]
  rest = [at for at, s in enumerate(ns) if s[-1:] in "-!"]
  return rest[-1] if rest else len(ns) - 1

def xname(s: Any) -> str:
  "A lab reading: drop any marker, so ezr sees an x column."
  s = str(s).strip()
  if s and s[-1] in "+-!X": s = s[:-1]
  return s or " "

def yname(s: Any) -> str:
  "The one goal: capitalized (so ezr makes it a Num) and a `+`."
  s = xname(s).strip() or "Y"
  return s[0].upper() + s[1:] + "+"

def score(row: ROW, at: int) -> float | None:
  "The panel's score for ROW; None if we were never told it."
  v = row[at] if 0 <= at < len(row) else "?"
  if isinstance(v, bool) or not isinstance(v, (int, float)):
    return None
  return float(v)

@contextmanager
def poke(**kw: Any):
  "Lend `the` some new settings, then give them back."
  old = {k: the[k] for k in kw}
  the.update(kw)
  try: yield
  finally: the.update(old)

#-- the model ----------------------------------------------
def est(m: MODEL, rows: ROWS) -> float:
  "Mean score of ROWS, pulled gently toward the global mean."
  ys = [y for y in (score(r, m.at) for r in rows) if y is not None]
  return (sum(ys) + PRIOR * m.mu) / (len(ys) + PRIOR)

def grow(m: MODEL, tbl: Any) -> Any:
  "One regression tree, each leaf labelled with its mean score."
  node = tree(tbl, tbl.rows)
  for lf in leafs(node): lf.est = est(m, lf.rows)
  return node

def Model(train: ROWS) -> MODEL | None:
  "A small forest, grown once per train set."
  ns   = names(train)
  if not ns: return None
  at   = goalAt(ns)
  hdr  = [yname(s) if i == at else xname(s)
          for i, s in enumerate(ns)]
  rows = [r for r in data(train) if score(r, at) is not None]
  if not rows: return None
  ys   = [score(r, at) for r in rows]
  tbl  = Tbl([hdr] + rows)
  m    = o(at=at, n=len(ns), mu=sum(ys)/len(ys),
           lo=min(ys), hi=max(ys), tbl=tbl, trees=[])
  rnd  = Random(the.Seed)
  many = TREES if len(rows) >= 4 * the.Leaf else 1
  for i in range(many):      # tree 0 sees all rows, the rest bag
    sub = tbl if i == 0 else clone(tbl, rnd.choices(rows,
                                                    k=len(rows)))
    m.trees += [grow(m, sub)]
  return m

_cache: dict[int, tuple[ROWS, MODEL | None]] = {}

def model(train: ROWS) -> MODEL | None:
  "Model for TRAIN, built once: guess is called many times."
  hit = _cache.get(id(train))
  if hit and hit[0] is train: return hit[1]
  out = Model(train) if train and len(train) > 1 else None
  if len(_cache) > 2: _cache.clear()   # one split at a time
  _cache[id(train)] = (train, out)     # keep TRAIN alive, so
  return out                           # its id stays its own

#-- request 1: the score this batch will get ---------------
def guess(train: ROWS, row: ROW) -> float:
  """Predicted Quality+ for one batch, from its readings alone.

  Averages the forest's leaf means, then clips to the range of
  scores the panel has actually handed out.
  """
  m = model(train)
  if m is None: return 0.0
  r  = list(row) + ["?"] * max(0, m.n - len(row))
  ys = [leaf(node, r).est for node in m.trees]
  return max(m.lo, min(m.hi, sum(ys) / len(ys)))

#-- request 2: a few kinds of batch ------------------------
def kindsTree(train: ROWS) -> tuple[MODEL, Any] | None:
  "One coarse tree: leaves of about n/KINDS rows."
  m = model(train)
  if m is None or not m.tbl.rows: return None
  with poke(Leaf=max(the.Leaf, len(m.tbl.rows) // KINDS)):
    return m, tree(m.tbl, m.tbl.rows)

def kinds(train: ROWS) -> list[tuple[ROWS, float]]:
  """The cellar, split into a few kinds of batch.

  Returns (group_rows, expected_quality) pairs, best kind first.
  The groups are a partition: every train row lands in one.
  """
  found = kindsTree(train)
  if not found: return []
  m, node = found
  out = [(lf.rows, est(m, lf.rows)) for lf in leafs(node)]
  out.sort(key=lambda z: -z[1])
  return out

#-- show me ------------------------------------------------
def eg_h() -> None:
  "Print help."
  print(__doc__)

def eg_all() -> None:
  "Load ezr's -File, split it in half, and answer both questions."
  rows  = list(ezr.csv(ezr.the.File))
  ns, d = names(rows), data(rows)
  d     = Random(the.Seed).sample(d, len(d))
  n     = len(d) // 2
  at    = goalAt(ns)
  train = [ns] + d[:n]
  test  = [r for r in d[n:] if score(r, at) is not None]

  print(f"\n-- predicting {ns[at]} "
        f"({n} train, {len(test)} test) ----------")
  ys  = [y for y in (score(r, at) for r in d[:n]) if y is not None]
  mu  = sum(ys) / len(ys)
  got = [(score(r, at), guess(train, r)) for r in test]
  mae = sum(abs(a - b) for a, b in got) / len(got)
  dumb = sum(abs(a - mu) for a, _ in got) / len(got)
  print(f"  mean abs error : {say(mae,3)}")
  print(f"  always say {say(mu,3):>5} : {say(dumb,3)}  (baseline)")

  found = kindsTree(train)
  if found:
    m, node = found
    print(f"\n-- kinds of batch -----------------------------")
    for group, want in kinds(train):
      print(f"  {len(group):>4} batches, expect "
            f"{say(want,2):>5} {ns[at]}")
    print()
    ezr.show(m.tbl, node)

if __name__ == "__main__":
  ezr.main(sys.argv[1:], globals())
