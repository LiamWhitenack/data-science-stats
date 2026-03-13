import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint


def plot_confidence_intervals(
    estimates: list[float],
    intervals: list[tuple[float, float]],
    labels: list[str],
    name: str,
) -> None:

    lower_errors = [
        estimate - interval[0] for estimate, interval in zip(estimates, intervals)
    ]

    upper_errors = [
        interval[1] - estimate for estimate, interval in zip(estimates, intervals)
    ]

    y_positions = [0, 0.25]

    plt.figure(figsize=(6, 4))  # much shorter vertically

    plt.errorbar(
        estimates,
        y_positions,
        xerr=[lower_errors, upper_errors],
        fmt="o",
        capsize=4,
    )

    plt.yticks(y_positions, labels)

    plt.xlim(0, 1)
    plt.ylim(-0.1, 0.35)

    plt.xlabel("Proportion Believe In Life After Death")
    plt.title(name)

    ax = plt.gca()

    # Remove box around plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Remove y tick marks (keep labels)
    ax.tick_params(axis="y", length=0)

    plt.savefig(f"homework_1/plots/{name.replace(' ', '_')}.png")


for method in ("normal", "wilson"):
    male_lower, male_upper = proportion_confint(
        count=703,
        nobs=945,
        alpha=0.05,
        method=method,
    )
    female_lower, female_upper = proportion_confint(
        count=1017,
        nobs=1178,
        alpha=0.05,
        method=method,
    )
    print(male_lower, male_upper, female_lower, female_upper)
    plot_confidence_intervals(
        [703 / 945, 1017 / 1178],
        [(male_lower, male_upper), (female_lower, female_upper)],
        ["male", "female"],
        f"{method} confidence intervals".title(),
    )
