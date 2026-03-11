import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from numpy import errstate, isfinite, log, ndarray
from numpy.random import binomial


def get_risk_ratio(n1: int, n2: int) -> tuple[ndarray, ndarray]:
    y1 = binomial(n1, 0.20, 1_000_000)
    y2 = binomial(n2, 0.10, 1_000_000)

    pi1_hat = y1 / 50
    pi2_hat = y2 / 50

    with errstate(divide="ignore", invalid="ignore"):
        risk_ratio = pi1_hat / pi2_hat
        log_risk_ratio = log(risk_ratio)

    # keep only valid values
    risk_ratio = risk_ratio[isfinite(risk_ratio)]
    log_risk_ratio = log_risk_ratio[isfinite(log_risk_ratio)]

    return risk_ratio, log_risk_ratio


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


histogram(50, 50, "Q2 (a) Histogram")
histogram(1000, 1000, "Q2 (b) Histogram")
histogram(30, 70, "Q2 (c) Histogram")
histogram(70, 30, "Q2 (d) Histogram")
