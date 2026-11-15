"""Statistics for the quantization x temperature study.

Contract for each function is in its docstring. Each function is derived from its definition,
with its tests in tests/test_stats.py written first.
Use only the standard library and numpy.
"""

from __future__ import annotations

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a proportion: k successes out of n trials.

    Returns (low, high), both inside [0, 1]. Must behave at k = 0 and k = n, where the
    normal-approximation ("Wald") interval collapses to zero width.
    Before coding: write down why Wald fails near 0 and 1, in two sentences, in notes/log.md.
    """
    raise NotImplementedError


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test for paired yes/no outcomes.

    b = items where condition A is right and condition B is wrong; c = the reverse.
    Items where both agree carry no information and are not passed in.
    Returns the p-value under the null that a discordant item is equally likely to go either way.
    Edge cases: b + c == 0 returns 1.0; the result never exceeds 1.0.
    """
    raise NotImplementedError


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the ORIGINAL order.

    Adjusted values are monotone in the sorted order and capped at 1.0.
    """
    raise NotImplementedError


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0,
                          alpha: float = 0.05) -> tuple[float, float, float]:
    """Difference in mean accuracy between two conditions scored on the SAME items.

    a, b: 1-D arrays of 0/1 (or per-item accuracy in [0, 1]), same length, same item order.
    Resample ITEMS with replacement (the same indices for a and b, which is what makes it paired).
    Returns (mean(a) - mean(b), low, high) with a percentile interval at level 1 - alpha.
    Must be reproducible for a given seed.
    """
    raise NotImplementedError


def decision_flip_rate(outcomes: np.ndarray) -> float:
    """Decision-flip rate (DFR) of one configuration.

    outcomes: 2-D array, shape (items, samples), entries 0/1 = wrong/right for each repeated sample.
    Read section "metrics" of arXiv 2606.29581 and implement THEIR definition. Copy the definition,
    in your own words, into this docstring before you write any code, including what the paper
    compares each sample against.
    Properties any correct version has: unanimous rows contribute no flips; result is in [0, 1].
    """
    raise NotImplementedError


def compound_degradation_index(acc_ref: float, acc_quant: float, acc_temp: float, acc_both: float) -> float:
    """Compound Degradation Index (CDI) from arXiv 2606.29581: do quantization and temperature
    hurt more together than the sum of their separate effects?

    acc_ref   = accuracy at full precision, temperature 0
    acc_quant = quantized, temperature 0
    acc_temp  = full precision, temperature T
    acc_both  = quantized, temperature T
    Read the paper's definition (it is defined on attack success rate; decide and DOCUMENT how you
    map it to accuracy, including the sign convention) and implement it. Then add your own test.
    """
    raise NotImplementedError
