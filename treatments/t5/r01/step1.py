#!/usr/bin/env python3 -B
"""
step1.py: answers two questions from the winemaker.

  1. "Out of all our past batches, show me the handful that
     were truly exceptional."
         -> remember(rows)

  2. "Give me a quick picture of a normal batch from our
     cellar: what each lab reading usually looks like."
         -> summary(rows)

Rows arrive as a list of lists: row[0] is the header of
strings, and the rest are numbers.  Header names ending in
`+` are goals to maximize, `-` goals to minimize, and the
rest are lab readings.

ezr.py does the work.  Its `Tbl` reads the same header
convention (`+-!` marks a goal), and its `ydist` scores a
row by how far its goals sit from the best they could be:
0 is perfect, 1 is as bad as it gets.  So "exceptional"
means "smallest ydist", and "normal" means each column's
middle (`mid`): the mean of a number, the most common value
of a symbol.
"""
import math

import ezr


def _tbl(rows: list[list]) -> ezr.TBL:
  "Build an ezr table from [header] + data rows."
  if not rows:
    raise ValueError("step1: no header row")
  return ezr.Tbl(rows)


def remember(rows: list[list]) -> list[list]:
  """The exceptional batches: best on the goal columns.

  At most 5% of the data rows, sorted best first.  With any
  data at all we return at least one row -- a cellar always
  has a best batch -- but never more than the 5% cap.
  """
  tbl = _tbl(rows)
  if not tbl.rows: return []
  keep = min(len(tbl.rows),                # 5%, at least one
             max(1, math.floor(0.05 * len(tbl.rows))))
  best = sorted(tbl.rows, key=lambda r: ezr.ydist(tbl, r))
  return best[:keep]


def summary(rows: list[list]) -> dict:
  """A typical batch: every column's name -> its middle.

  Numbers report their mean, symbols their most common
  value, and a column of nothing but "?" reports "?".
  Goal columns are included, since the winemaker wants the
  whole picture of a normal batch.
  """
  tbl = _tbl(rows)
  return {tbl.names[at]: ezr.mid(col)
          for at, col in tbl.cols.items()}


def main() -> None:
  "Read -File, print the handful and the picture."
  src  = list(ezr.csv(ezr.the.File))
  head = src[0]
  print("# exceptional batches")
  print(", ".join(map(str, head)))
  for row in remember(src):
    print(", ".join(ezr.say(v) for v in row))
  print("\n# a normal batch")
  for name, v in summary(src).items():
    print(f"{name:>16} = {ezr.say(v)}")


if __name__ == "__main__":
  ezr.run(main)
