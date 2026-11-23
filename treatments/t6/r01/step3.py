#!/usr/bin/env python3 -B
"""
step3.py: a winemaker's assistant, built on ezr.py, step1, step2.

Two last questions from the cellar:

  "When you say batch A will beat batch B, tell me whether
   that's a real difference or just tasting noise."
      -> certify(train, a, b)   ->  'A', 'B' or 'same'

  "How often are your score estimates right on batches you've
   never seen before?"
      -> accuracy(train, test)  ->  0..1

TRAIN is [header, *data], as in step1 and step2: the header names
the columns (trailing "+" = maximize, "-" = minimize, the rest
are lab readings) and every later row holds numbers.  TEST is the
same shape; a header there is optional, so a bare block of rows
works too.

Where does the verdict come from?  step2 already grows a small
forest that predicts the panel's score (Quality+).  Each tree
drops a batch into a leaf -- a pocket of past batches that look
like it on the bench -- and the panel scores of those rows are
the scores batches like this one have actually earned.  Pool
those scores over the forest and you have a *spread*, not a
point: the tasting noise around one guess.

So: `guess` says which way the call goes, and the two spreads say
whether the call is worth making.  Two spreads count as alike
when all three of ezr's tests agree -- cliffs (rank imbalance),
ks (biggest gap between the two curves) and cohen (means inside
one third of a pooled standard deviation).  Alike means 'same'.

Costs: certify is called for hundreds of pairs against one train
set, so the forest is built once (step2 caches it) and each
batch's spread is cached on the model, sorted, ready for the
tests.  A pair then costs two tree walks per tree and one linear
pass per test.
"""
# pylint: disable=invalid-name

import sys
from math import floor
from random import Random
from typing import Any

import ezr
from ezr import o, the, leaf, cliffs, ks, cohen, say
import step1
from step1 import names
import step2
from step2 import goalAt, model, guess, score

#-- types --------------------------------------------------
type ROW   = list[Any]
type ROWS  = list[ROW]
type MODEL = o

#-- dials --------------------------------------------------
EFFECT = 0.35     # cohen: mean gaps smaller than this are noise
CACHE  = 4096     # most spreads to remember, per model

#-- rows in ------------------------------------------------
def isHeader(row: ROW) -> bool:
  "A row of names: every cell a string.  Data rows carry numbers."
  return bool(row) and all(isinstance(v, str) for v in row)

def body(rows: ROWS) -> ROWS:
  "The data rows of ROWS, with any header dropped."
  rows = rows or []
  if not rows: return []
  return list(rows[1:]) if isHeader(rows[0]) else list(rows)

def pad(m: MODEL, row: ROW) -> ROW:
  "ROW, cut or padded to the width the model was trained on."
  r = list(row)[:m.n]
  return r + ["?"] * (m.n - len(r))

#-- the spread behind one guess ----------------------------
def leafYs(m: MODEL, lf: o) -> list[float]:
  "Panel scores of this leaf's rows, sorted, kept on the leaf."
  if lf.get("ys") is None:
    lf.ys = sorted(y for y in (score(r, m.at) for r in lf.rows)
                   if y is not None)
  return lf.ys

def spread(m: MODEL, row: ROW) -> list[float]:
  """Scores earned by past batches that look like ROW, sorted.

  One pocket of look-alikes per tree, pooled.  Cached on the
  model, since the same batch shows up in many pairs.
  """
  r   = pad(m, row)
  key = tuple(r)
  if m.get("spreads") is None: m.spreads = {}
  hit = m.spreads.get(key)
  if hit is not None: return hit
  out: list[float] = []
  for node in m.trees: out += leafYs(m, leaf(node, r))
  out.sort()
  if len(m.spreads) > CACHE: m.spreads.clear()
  m.spreads[key] = out
  return out

def alike(xs: list[float], ys: list[float]) -> bool:
  """Are two spreads indistinguishable?

  This is ezr's `same`, with the sort hoisted out into the cache
  above, and with cohen always in the jury: a difference has to
  survive rank, curve and effect size before we report it.
  """
  if not xs or not ys: return True
  return (cliffs(xs, ys) and ks(xs, ys)
          and cohen(xs, ys, d=EFFECT))

#-- request 1: is the difference real? ---------------------
def certify(train: ROWS, a: ROW, b: ROW) -> str:
  """'A' or 'B' for the batch that will score better, else 'same'.

  Direction comes from `guess` (the score column is a goal to
  maximize, so the bigger guess wins).  The verdict comes from
  the two spreads: if look-alikes of A and of B earned scores we
  cannot tell apart, the gap between the guesses is noise.
  """
  m = model(train)
  if m is None: return "same"
  ya, yb = guess(train, a), guess(train, b)
  if ya == yb: return "same"
  if alike(spread(m, a), spread(m, b)): return "same"
  return "A" if ya > yb else "B"

#-- request 2: how often is the guess right? ---------------
def near(x: float) -> int:
  "Nearest whole score, halves rounding up (the panel's scale)."
  return floor(x + 0.5)

def accuracy(train: ROWS, test: ROWS) -> float:
  """Share of unseen batches whose guess rounds to the true score.

  Batches with no score of their own are skipped: there is
  nothing there to be right about.  No scores at all returns 0.
  """
  m = model(train)
  if m is None: return 0.0
  hits = n = 0
  for r in body(test):
    y = score(r, m.at)
    if y is None: continue
    n   += 1
    hits += near(guess(train, r)) == near(y)
  return hits / n if n else 0.0

#-- show me ------------------------------------------------
def eg_h() -> None:
  "Print help."
  print(__doc__)

def eg_all() -> None:
  "Load ezr's -File, split it in half, and answer both questions."
  rows  = list(ezr.csv(ezr.the.File))
  ns, d = names(rows), step1.data(rows)
  rnd   = Random(the.Seed)
  d     = rnd.sample(d, len(d))
  n     = len(d) // 2
  at    = goalAt(ns)
  train = [ns] + d[:n]
  test  = [ns] + [r for r in d[n:] if score(r, at) is not None]
  got   = body(test)

  acc = accuracy(train, test)
  print(f"\n-- guessing {ns[at]} ({n} train, {len(got)} test) ----")
  print(f"  rounds to the panel : {say(100*acc,1)}% of the time")

  pairs = [(rnd.choice(got), rnd.choice(got)) for _ in range(500)]
  tally = {"A": 0, "B": 0, "same": 0}
  for a, b in pairs: tally[certify(train, a, b)] += 1
  print(f"\n-- 500 pairs, certified ------------------------")
  for k, v in tally.items():
    print(f"  {k:>4} : {v:>4} ({say(100*v/500,1)}%)")

  a, b = min(got, key=lambda r: guess(train, r)), \
         max(got, key=lambda r: guess(train, r))
  print(f"\n-- the two extremes ----------------------------")
  print(f"  worst guess {say(guess(train,a),3)} vs "
        f"best {say(guess(train,b),3)} -> "
        f"{certify(train, a, b)}")
  print(f"  a batch against itself -> {certify(train, a, a)}")

if __name__ == "__main__":
  ezr.main(sys.argv[1:], globals())
