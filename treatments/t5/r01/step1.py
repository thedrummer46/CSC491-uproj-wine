#!/usr/bin/env python3 -B
"""
step1.py: two answers for the winemaker, built on ezr.

  remember(rows) -> the handful of truly exceptional batches
  summary(rows)  -> what a normal batch looks like

Rows arrive as a list of lists: row[0] is the header (strings),
the rest are numbers.  The header marks say what each column is
for: `+` maximize, `-` minimize, the rest are lab readings.  ezr
reads those marks when it builds the table, so `ydist` already
means "how far this batch is from the best possible one" over
Quality+ and Alcohol-.  Exceptional therefore == small ydist.
"""
import sys
import ezr

MOST = 0.05   # remember: never return more than this share of rows
HAND = 5      # remember: a handful is five

def table(rows: ezr.ROWS) -> ezr.TBL:
  "A TBL from ROWS, whose first row names the columns."
  if not rows: raise ValueError("step1: no rows given")
  return ezr.Tbl(rows)

def scored(tbl: ezr.TBL) -> list[tuple[float, ezr.ROW]]:
  "Each row with its distance-to-best; rows missing a goal skip."
  out = []
  for row in tbl.rows:
    if any(row[c.at] == "?" for c in tbl.y): continue
    out.append((ezr.ydist(tbl, row), row))
  return out

def remember(rows: ezr.ROWS) -> ezr.ROWS:
  """The truly exceptional past batches: the ones closest to the
  best mix of high Quality+ and low Alcohol-.  Best first.  Never
  more than a handful, and never more than 5% of the data rows."""
  tbl  = table(rows)
  pairs = scored(tbl)
  if not pairs: return []
  k = max(1, min(HAND, int(MOST * len(tbl.rows))))
  pairs.sort(key=lambda z: z[0])
  return [row for _, row in pairs[:k]]

def summary(rows: ezr.ROWS, p: int = 4) -> dict:
  """A quick picture of a normal batch: every column's middle --
  the mean of the numbers, or the most common symbol."""
  tbl = table(rows)
  return {tbl.names[at]: (round(v, p) if isinstance(v, float) else v)
          for at, v in ezr.mids(tbl).items()}

#-- demo ---------------------------------------------------
def main(args: list[str]) -> None:
  "step1.py [file.csv] -- show both answers for one file."
  rows = list(ezr.csv(args[0] if args else ezr.the.File))
  print("# a normal batch")
  for k, v in summary(rows).items(): print(f"  {k:>16} {ezr.say(v,4)}")
  print(f"# exceptional batches (of {len(rows)-1})")
  for row in remember(rows):
    print("  ", ", ".join(ezr.say(v, 4) for v in row))

if __name__ == "__main__": main(sys.argv[1:])
