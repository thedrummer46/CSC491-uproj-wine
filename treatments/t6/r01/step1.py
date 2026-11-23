#!/usr/bin/env python3 -B
"""
step1.py: a winemaker's assistant, built on ezr.py.

Two questions from the cellar:

  "Out of all our past batches, show me the handful that
   were truly exceptional."
      -> remember(rows)

  "Give me a quick picture of a normal batch from our cellar:
   what each lab reading usually looks like."
      -> summary(rows)

Both take ROWS = [header, *data], where the header names the
columns (trailing "+" = maximize, "-" = minimize, the rest are
lab readings) and every later row holds numbers.

"Exceptional" is not one number.  A batch is exceptional when
all its goals are close to the best they could be, so we score
each batch with ezr's `ydist`: the distance from a row's goals
to the ideal corner (Quality high, Alcohol low).  Small is good.
"""
# pylint: disable=invalid-name

from typing import Any
from math import floor

import ezr
from ezr import Tbl, Col, add, mid, ydist

#-- types --------------------------------------------------
type ROW  = list[Any]
type ROWS = list[ROW]

#-- how big is "a handful"? --------------------------------
KEEP = 0.05          # at most this fraction of the data rows

#-- helpers ------------------------------------------------
def names(rows: ROWS) -> list[str]:
  "The header, tidied."
  return [str(s).strip() for s in rows[0]]

def data(rows: ROWS) -> ROWS:
  "Everything below the header."
  return list(rows[1:])

def table(rows: ROWS) -> Any:
  "An ezr TBL: header plus data, so goals know their direction."
  return Tbl([names(rows)] + data(rows))

#-- request 1: the exceptional batches ---------------------
def scored(rows: ROWS) -> list[tuple[float, ROW]]:
  "Every batch, with its distance-to-ideal.  Best first."
  tbl = table(rows)
  out = [(ydist(tbl, r), r) for r in tbl.rows]
  out.sort(key=lambda z: z[0])
  return out

def remember(rows: ROWS) -> ROWS:
  """The standout past batches, best first.

  Returns at most 5% of the data rows (16 of 320), floored, so
  the list stays a handful and never pads itself with mediocrity.
  """
  rows = rows or []
  if len(rows) < 2: return []
  n = len(data(rows))
  k = floor(KEEP * n)                  # "at most", so round down
  if k < 1: return []
  return [r for _, r in scored(rows)[:k]]

#-- request 2: the picture of a normal batch ---------------
def summary(rows: ROWS) -> dict[str, Any]:
  """One typical value per column, in header order.

  Numbers report their mean, symbols their most common value,
  and an all-unknown column reports "?".  Columns ezr would
  skip are summarized here too: the winemaker asked about
  every reading, not just the modelled ones.
  """
  rows = rows or []
  if not rows: return {}
  ns   = names(rows)
  cols = {at: Col(s or " ", at) for at, s in enumerate(ns)}
  for row in data(rows):
    for at, col in cols.items():
      if at < len(row): add(col, row[at])
  return {ns[at]: mid(col) for at, col in cols.items()}

#-- show me ------------------------------------------------
def eg_h() -> None:
  "Print help."
  print(__doc__)

def eg_all() -> None:
  "Load ezr's -File and answer both questions."
  rows = list(ezr.csv(ezr.the.File))
  print("\n-- a normal batch ------------------------------")
  for k, v in summary(rows).items():
    print(f"  {k:>16} : {ezr.say(v, 3)}")
  best = remember(rows)
  print(f"\n-- {len(best)} exceptional batches "
        f"(of {len(rows)-1}) ------------")
  print("  " + ",".join(str(s) for s in names(rows)))
  for r in best:
    print("  " + ",".join(ezr.say(v, 3) for v in r))

if __name__ == "__main__":
  ezr.main(sys.argv[1:] if (sys := __import__("sys")) else [],
           globals())
