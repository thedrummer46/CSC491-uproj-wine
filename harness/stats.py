"""The lecturer's statistics, so every claim in Results passes `same`.

`same` (cliffs + ks, + cohen when eps>0) comes straight from lib/ezr.py.
`ranks` follows gloss6 (txt.github.io/seai26f/gloss6.html).
Harness code: used to SCORE treatments, never shown to them.
"""
from statistics import median

from lib.ezr import cliffs, cohen, ks, same  # noqa: F401  (re-exported)


def ranks(d, big=True):
    """D = {treatment: [scores over repeats]}. Sort by median (best first);
    a new rank starts only when a treatment is not `same` as the one that
    opened the current rank. Returns {name: rank}; rank 0 = the winners."""
    order = sorted(d, key=lambda k: -median(d[k]) if big else median(d[k]))
    out, r, lead = {}, 0, order[0]
    for k in order:
        if not same(d[lead], d[k]):
            r, lead = r + 1, k
        out[k] = r
    return out
