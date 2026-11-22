#!/usr/bin/env python3 -B
"""
step2.py: answers two more questions from the winemaker.

  3. "Before the panel sits, estimate what score this new
     batch will get from its lab sheet alone."
         -> guess(train, row)

  4. "Group our batches into a few kinds, and tell me what
     score to expect from each kind."
         -> kinds(train)

Rows arrive as in step1: row[0] is the header of strings,
the rest are numbers.  A name ending in `+` is a goal to
maximize, `-` a goal to minimize, and the rest are lab
readings.

The panel score is the `+` goal (here, `Quality+`).  A lab
sheet holds lab readings only, so the other goals (here,
`Alcohol-`) are hidden from the model: we re-tag them `X`,
which is how ezr.Tbl is told to skip a column.  With one
goal left, ezr's `ydist` is simply "how far from the best
score", so the splits found by ezr.tree gather rows of like
score, and each leaf of that tree is one "kind" of batch.

A kind's expected score is the mean of the real scores
inside it, not its ydist: ydist is normalized to 0..1, and
the winemaker wants an answer in their own units.
"""
import random

import ezr


def _at(names: list[str]) -> int:
  "Where the panel score sits: the first column to maximize."
  for at, s in enumerate(names):
    if s[-1] == "+": return at
  raise ValueError("step2: no `+` goal column to predict")


def _header(names: list[str], at: int) -> list[str]:
  "Keep the lab readings and column AT, hide other goals."
  return [s if i == at or s[-1] not in "+-!" else s + "X"
          for i, s in enumerate(names)]


def _known(row: list, at: int) -> bool:
  "Does ROW report a score?"
  return len(row) > at and row[at] != "?"


def _tbl(rows: list[list]) -> ezr.TBL:
  "An ezr table whose one goal is the panel score."
  if not rows:
    raise ValueError("step2: no header row")
  at  = _at(rows[0])
  tbl = ezr.Tbl([_header(rows[0], at)] +
                [r for r in rows[1:] if _known(r, at)])
  tbl.at = at            # the score lives here, in real units
  return tbl


def _mu(tbl: ezr.TBL, rows: list[list]) -> float | None:
  "Mean real score of ROWS; None if no row reports one."
  vs = [r[tbl.at] for r in rows if _known(r, tbl.at)]
  return sum(vs) / len(vs) if vs else None


_fits: dict = {}     # cache of one: training sets are big


def _fit(train: list[list]) -> ezr.o:
  "Grow (and briefly remember) TRAIN's tree of kinds."
  key = (id(train), len(train))
  if key not in _fits:
    tbl = _tbl(train)
    if not tbl.rows:
      raise ValueError("step2: no past batches with a score")
    _fits.clear()
    _fits[key] = ezr.o(tbl=tbl, mu=_mu(tbl, tbl.rows),
                       node=ezr.tree(tbl, tbl.rows))
  return _fits[key]


def guess(train: list[list], row: list) -> float:
  """The score the panel should give this one batch.

  ROW falls down TRAIN's tree of kinds -- on lab readings
  alone -- and we report the mean score of the kind it
  lands in.  Only TRAIN is consulted: whatever ROW says in
  its own score column is never read.
  """
  fit = _fit(train)
  mu  = _mu(fit.tbl, ezr.leaf(fit.node, row).rows)
  return float(fit.mu if mu is None else mu)


def kinds(train: list[list]) -> list[tuple[list[list], float]]:
  """The few kinds of batch, and what to expect from each.

  One (rows, expected score) pair per leaf of the tree,
  best kind first.  Every kind holds at least `the.Leaf`
  batches, so these are groups worth talking about, and
  together they cover all of TRAIN that carries a score.
  """
  fit = _fit(train)
  out = []
  for node in ezr.leafs(fit.node):
    mu = _mu(fit.tbl, node.rows)
    out += [(node.rows, float(fit.mu if mu is None else mu))]
  return sorted(out, key=lambda two: -two[1])


def main() -> None:
  "Read -File: hold back half the batches, guess, then group."
  src  = list(ezr.csv(ezr.the.File))
  head, at = src[0], _at(src[0])
  rows  = random.sample(src[1:], len(src) - 1)
  n     = len(rows) // 2
  train, test = [head] + rows[:n], [r for r in rows[n:]
                                    if _known(r, at)]
  print("# guesses vs the panel")
  print(f"{'guess':>8} {'actual':>8}")
  gaps = []
  for row in test:
    got = guess(train, row)
    gaps += [abs(got - row[at])]
    if len(gaps) <= 10: print(f"{got:8.2f} {row[at]:8.2f}")
  if gaps:
    print(f"\nmean error {sum(gaps)/len(gaps):.2f} "
          f"over {len(gaps)} held-back batches")
  print("\n# kinds of batch")
  print(f"{'score':>8} {'n':>6}")
  for group, mu in kinds(train):
    print(f"{mu:8.2f} {len(group):6}")


if __name__ == "__main__":
  ezr.run(main)
