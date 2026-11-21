"""The lecturer's statistics, so every claim in Results passes `same`.

`cohen`, `cliffs`, `ks`, `same` follow ezr.py (github.com/timm/ezr, ezr.py
stats section); `ranks` follows gloss6 (txt.github.io/seai26f/gloss6.html).
Harness code: used to SCORE treatments, never shown to them.
"""
from math import sqrt
from statistics import mean, median, stdev


def cohen(xs, ys, d=0.35, eps=0):
    "Are the means within D pooled standard deviations?"
    a, b = len(xs), len(ys)
    pool = ((a - 1) * stdev(xs) ** 2 + (b - 1) * stdev(ys) ** 2) / (a + b - 2)
    return abs(mean(xs) - mean(ys)) <= max(eps, d * sqrt(pool))


def cliffs(xs, ys, d=0.197):
    "Sorted in. Is the rank imbalance small enough?"
    gt = lt = j = k = 0
    for x in xs:
        while j < len(ys) and ys[j] < x: j += 1; k = j
        while k < len(ys) and ys[k] <= x: k += 1
        gt += j; lt += len(ys) - k
    return abs(gt - lt) / (len(xs) * len(ys)) <= d


def ks(xs, ys, a=1.36):
    "Sorted in. 95% Kolmogorov-Smirnov."
    n, m, i, j, d = len(xs), len(ys), 0, 0, 0
    while i < n and j < m:
        v = min(xs[i], ys[j])
        while i < n and xs[i] <= v: i += 1
        while j < m and ys[j] <= v: j += 1
        d = max(d, abs(i / n - j / m))
    return d <= a * sqrt((n + m) / (n * m))


def same(xs, ys, eps=0):
    "Indistinguishable? Always cliffs and ks; EPS>0 adds cohen."
    xs, ys = sorted(xs), sorted(ys)
    return cliffs(xs, ys) and ks(xs, ys) and (not eps or cohen(xs, ys, d=eps))


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
