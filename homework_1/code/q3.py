import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from numpy import errstate, isfinite, log, ndarray
from numpy.random import binomial, default_rng

rng = default_rng()


def quartile_mean_estimator(sample: np.ndarray) -> float:
    """Estimator: average of sample lower and upper quartiles."""
    q1, q3 = np.percentile(sample, [25, 75])
    return (q1 + q3) / 2


def simulate_precision(n: int, num_samples: int = 100_000) -> None:
    """Simulate quartile estimator vs sample mean."""
    # Generate all samples at once for efficiency
    samples = rng.normal(loc=0, scale=1, size=(num_samples, n))

    # Compute quartile-based estimator for each sample
    quartile_estimates = np.apply_along_axis(quartile_mean_estimator, 1, samples)

    # Compute sample means for comparison
    mean_estimates = samples.mean(axis=1)

    # Compute standard deviations of estimates
    std_quartile = np.std(quartile_estimates, ddof=1)
    std_mean = np.std(mean_estimates, ddof=1)

    # Theoretical standard error of the mean
    se_mean = 1 / np.sqrt(n)

    print(f"Sample size n = {n}")
    print(f"Quartile-based estimator std dev: {std_quartile:.5f}")
    print(f"Sample mean std dev: {std_mean:.5f}")
    print(f"Theoretical standard error of mean: {se_mean:.5f}")
    print("-" * 40)


def histogram(n1: int, n2: int, fname: str) -> None:
    risk_ratio, log_risk_ratio = get_risk_ratio(n1, n2)

    # Combine for consistent binning
    combined = np.concatenate([risk_ratio, log_risk_ratio])
    bins = np.linspace(combined.min(), combined.max(), 101)

    # Create figure
    plt.figure(figsize=(8, 5))

    # Plot histograms
    plt.hist(risk_ratio, bins=bins, alpha=0.5, label="Risk Ratio")
    plt.hist(log_risk_ratio, bins=bins, alpha=0.5, label="Log Risk Ratio")

    # Titles and axis labels
    plt.title(fname)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Keep only left and bottom spines (like gridlines)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    # Format y-axis in thousands
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))

    # Legend
    plt.legend()

    # Save and close
    plt.savefig(
        f"homework_1/plots/{fname.replace('(', '').replace(')', '').replace(' ', '_')}.png",
        bbox_inches="tight",
    )
    plt.close()


# Part (a): n = 100
simulate_precision(n=100)

# Part (b): n = 500
simulate_precision(n=500)
