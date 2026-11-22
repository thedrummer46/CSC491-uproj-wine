#!/usr/bin/env python3 -B
"""
step1.py: two answers for the winemaker, built on ezr.py.

  remember(rows) -> the handful of truly exceptional batches
                    (at most 5% of the data rows).
  summary(rows)  -> what a normal batch looks like: one
                    typical value per column.

Rows arrive as a list of lists: rows[0] is the header of
strings, the rest are numbers.  The header says what each
column is for: a name ending in `+` is a goal to maximize,
`-` a goal to minimize, and the rest are lab readings.
ezr reads those suffixes for us, so "exceptional" here means
"closest to the best possible mix of every goal at once" --
for this cellar, high Quality+ with low Alcohol-.

Usage:  python3 step1.py [wines.csv]
"""
# pylint: disable=bad-indentation,invalid-name

import sys
from ezr import Tbl, mids, ydist, say, csv, the

#-- settings -----------------------------------------------
Pick = 0.05        # "a handful": the top 5% of batches, max

#-- answers ------------------------------------------------
def remember(rows: list[list]) -> list[list]:
  """The exceptional batches, best first.

  Every batch is scored by `ydist`: its distance to the ideal
  corner where all goals are at their best.  Small is good.
  We keep the 5% with the smallest distance -- rounded *down*,
  so the result is never more than 5% of the data rows (a
  cellar of under 20 batches has no standouts, just batches)."""
  tbl = Tbl(rows)
  k   = int(len(tbl.rows) * Pick)
  return sorted(tbl.rows, key=lambda r: ydist(tbl, r))[:k]

def summary(rows: list[list]) -> dict:
  """A picture of a normal batch: each column name mapped to
  its typical value (the mean of the numbers, or the most
  common symbol).  Unknown cells (`?`) are left out of the
  reckoning rather than counted as zero."""
  tbl = Tbl(rows)
  if not tbl.rows: return {name: "?" for name in tbl.names}
  return {tbl.names[at]: v for at, v in mids(tbl).items()}

#-- reading the header -------------------------------------
def goalp(name: str) -> bool:
  "Is this column a goal to chase, rather than a lab reading?"
  return name[-1] in "+-!"

def which(name: str) -> str:
  "How to read this column, in the winemaker's words."
  return {"+": "goal: higher is better",
          "-": "goal: lower is better"}.get(name[-1], "lab reading")

def label(name: str) -> str:
  "The column name, without its machine-readable suffix."
  return name[:-1] if goalp(name) else name

#-- reporting ----------------------------------------------
def report(rows: list[list]) -> None:
  "Print both answers for one batch list."
  names = rows[0]
  n     = len(rows) - 1

  print(f"\n# A normal batch   ({n} batches in the cellar)\n")
  typical = summary(rows)
  wide    = max(len(label(k)) for k in typical) if typical else 0
  for name, v in typical.items():
    print(f"  {label(name):<{wide}}  {say(v,3):>8}   {which(name)}")

  keep = remember(rows)
  print(f"\n# Truly exceptional batches   "
        f"(best {len(keep)} of {n}, best first)\n")
  if not keep:
    print("  Too few batches to single any out.")
    return
  goals = [at for at, name in enumerate(names) if goalp(name)]
  head  = "   ".join(f"{label(names[at]):>12}" for at in goals)
  print(f"  {'rank':>4}   {head}")
  for i, row in enumerate(keep):
    cells = "   ".join(f"{say(row[at],3):>12}" for at in goals)
    print(f"  {i+1:>4}   {cells}")
  print("\n  (full rows are the return value of remember())")

def main(args: list[str]) -> None:
  "Read a CSV -- named on the command line, or ezr's default."
  file = args[0] if args else the.File
  rows = list(csv(file))
  if len(rows) < 2:
    print(f"{file}: no data rows found."); return
  print(f"file: {file}")
  report(rows)

if __name__ == "__main__": main(sys.argv[1:])
