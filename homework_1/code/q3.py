from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from numpy import errstate, isfinite, log, ndarray, sqrt
from numpy.random import binomial, default_rng

rng = default_rng()


def quartile_mean_estimator(sample: np.ndarray) -> float:
    """Estimator: average of sample lower and upper quartiles."""
    q1, q3 = np.percentile(sample, [25, 75])
    return (q1 + q3) / 2


def simulate_precision(n: int) -> None:
    samples = rng.normal(loc=0, scale=1, size=(100_000, n))
    quartile_estimates = [quartile_mean_estimator(sample) for sample in samples]
    mean_estimates = samples.mean(axis=1)
    std_quartile = np.std(quartile_estimates, ddof=1)
    std_mean = np.std(mean_estimates, ddof=1)
    se_mean = 1 / np.sqrt(n)

    print(f"Sample size: {n}")
    print(f"Quartile-based estimator std dev: {std_quartile}")
    print(f"Sample mean std dev: {std_mean}")
    print(f"Theoretical standard error of mean: {se_mean}")
    print("\n")


simulate_precision(n=100)
simulate_precision(n=500)
