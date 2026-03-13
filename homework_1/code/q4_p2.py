from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import savefig, subplots, tight_layout
from pandas import read_csv
from scipy.stats import norm

rng = np.random.default_rng(0)

# Example EARTH theme colors (replace with your actual palette)
EARTH = {
    "accent": "#7f6d5f",  # bar color
    "zero": "#000000",  # edge color
}


# --------------------------------------------------
# Reusable barchart function
# --------------------------------------------------
def barchart(
    x,
    y,
    y_label: str,
    figsize: tuple[int, int] = (8, 5),
    filename: str = "barchart",
    x_label: str | None = None,
) -> None:

    fig, ax = subplots(figsize=figsize)

    # Use EARTH accent for bars (minimal, clean)
    ax.barh(
        x,
        y,
        color=EARTH["accent"],
        linewidth=0.8,
    )

    # --- Minimal styling ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if x_label is not None:
        ax.set_xlabel(x_label)

    ax.set_ylabel(y_label)

    # --- Title from filename ---
    clean_name = Path(filename).stem
    title = clean_name.replace("_", " ").replace("-", " ").title()
    ax.set_title(title, pad=12)

    tight_layout()
    savefig(f"homework_1/plots/{clean_name}.png")
    plt.close()


# --------------------------------------------------
# (i) Problem description
# --------------------------------------------------

print(
    """
(i) Problem Description

The goal of this project is to determine which demographic categories
are associated with higher happiness levels. In particular, we examine
whether happiness varies across marital status and sex categories.

Understanding which groups report higher happiness can provide insight
into how social factors influence well-being.

Unless otherwise specified, all interval estimates use 90% confidence
levels.
"""
)

# --------------------------------------------------
# (ii) Load and describe data
# --------------------------------------------------

df = read_csv("homework_1/data/happy.csv")

print(
    """
(ii) Data Description

The dataset contains the following variables:

id        : unique respondent identifier
happiness : categorical happiness score (1-3)
marital   : marital status category (1-3)
sex       : respondent sex ("male" or "female")

Happiness and marital status are ordinal categorical variables where
larger values represent higher levels of the attribute.
"""
)

print("\nSample size:", len(df))


# --------------------------------------------------
# (iii) Sample distributions
# --------------------------------------------------

print(
    """
(iii) Sample Data Distributions

Bar charts show the distribution of each categorical variable.
"""
)
# Example bar charts for individual variables
for column in ["happiness", "marital", "sex"]:
    counts = df[column].value_counts().sort_index()
    barchart(
        x=counts.index.astype(str),
        y=counts.values,
        filename=f"{column}_distribution",
        x_label="Count",
        y_label=column.title(),
    )


def plot_happiness_by_marital_sex(
    data,
    name: str = "mean_happiness_by_marriage_status_and_sex",
    confidence: float = 0.90,
) -> None:

    # Group by marital × sex
    grouped = (
        data.groupby(["marital", "sex"])["happiness"]
        .agg(["mean", "count", "std"])
        .unstack()
    )

    marital_categories = grouped.index.values  # e.g., 1, 2, 3
    sex_categories = grouped.columns.levels[1]  # 'male', 'female'

    z = norm.ppf((1 + confidence) / 2)

    plt.figure(figsize=(8, 5))

    # Vertical spacing for male/female within each marital category
    offset = 0.15
    colors = {"male": "#7f6d5f", "female": "#a6c48a"}

    y_positions = np.arange(len(marital_categories))

    for i, sex in enumerate(sex_categories):
        estimates = []
        lower_errors = []
        upper_errors = []

        # Compute mean and CI for each marital category
        for marital in marital_categories:
            mean = grouped.loc[marital, ("mean", sex)]
            count = grouped.loc[marital, ("count", sex)]
            std = grouped.loc[marital, ("std", sex)]
            se = std / np.sqrt(count)
            estimates.append(mean)
            lower_errors.append(z * se)
            upper_errors.append(z * se)

        # Shift y positions slightly for male/female
        y_pos = y_positions - offset / 2 if sex == "male" else y_positions + offset / 2

        plt.errorbar(
            x=estimates,
            y=y_pos,
            xerr=[lower_errors, upper_errors],
            fmt="o",
            capsize=4,
            markersize=6,
            color=colors[sex],
            label=sex,
        )

    plt.yticks(y_positions, [f"Marital {m}" for m in marital_categories])
    plt.xlabel("Mean Happiness")
    plt.title(name.replace("_", " ").title())
    plt.xlim(left=1, right=3)
    plt.legend(title="Sex")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    plt.savefig(f"homework_1/plots/{name}.png")
    plt.close()


plot_happiness_by_marital_sex(df)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------


def mean_confidence_interval(values, confidence=0.90):

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    z = norm.ppf((1 + confidence) / 2)

    margin = z * std / np.sqrt(len(values))

    return mean, mean - margin, mean + margin


def proportion_confidence_interval(count, n, confidence=0.90):

    p = count / n

    z = norm.ppf((1 + confidence) / 2)

    se = np.sqrt(p * (1 - p) / n)

    margin = z * se

    return p, p - margin, p + margin


# --------------------------------------------------
# (iv) Mean and proportion estimation
# --------------------------------------------------

mean_happiness, lower_mean, upper_mean = mean_confidence_interval(df["happiness"])

print(
    f"""
(iv) Mean and Proportion Estimation

The estimated mean happiness score is {mean_happiness:.3f}.

The 90% confidence interval for the population mean happiness is
({lower_mean:.3f}, {upper_mean:.3f}).

Interpretation:
We are 90% confident that the true population average happiness lies
within this interval.
"""
)

happy_count = np.sum(df["happiness"] == 3)

p_hat, lower_prop, upper_prop = proportion_confidence_interval(happy_count, len(df))

print(
    f"""
The estimated proportion of individuals reporting the highest happiness
(level 3) is {p_hat:.3f}.

The 90% confidence interval for this proportion is
({lower_prop:.3f}, {upper_prop:.3f}).

Interpretation:
We are 90% confident that the true population proportion of people with
the highest happiness level falls within this interval.
"""
)


# --------------------------------------------------
# (v) Group comparison using groupby
# --------------------------------------------------

group_stats = df.groupby("sex")["happiness"].agg(["mean", "median", "count"])

print(
    """
(v) Comparison Between Groups

Mean and median happiness by sex:
"""
)

print(group_stats)

means = group_stats["mean"]

mean_diff = means.max() - means.min()

print(
    f"""
The difference between the highest and lowest mean happiness across
sex categories is {mean_diff:.3f}.

Interpretation:
This difference indicates how much average happiness varies between
male and female respondents.
"""
)


# --------------------------------------------------
# (vi) Correlation with bootstrap CI
# --------------------------------------------------

print(
    """
(vi) Correlation Analysis

We estimate the correlation between marital status and happiness.
A 95% bootstrap confidence interval is constructed using 1000
bootstrap samples.
"""
)


def bootstrap_correlation(x, y, B=1000):

    correlations = []

    for _ in range(B):
        indices = rng.integers(0, len(x), len(x))

        sample_x = x[indices]
        sample_y = y[indices]

        correlations.append(np.corrcoef(sample_x, sample_y)[0, 1])

    return np.array(correlations)


corr_boot = bootstrap_correlation(df["happiness"].to_numpy(), df["marital"].to_numpy())

corr = np.corrcoef(df["happiness"], df["marital"])[0, 1]

lower_corr = np.percentile(corr_boot, 2.5)
upper_corr = np.percentile(corr_boot, 97.5)

print(
    f"""
The estimated correlation is {corr:.3f}.

The 95% bootstrap confidence interval is
({lower_corr:.3f}, {upper_corr:.3f}).
"""
)


# --------------------------------------------------
# (vii) Bootstrap Pearson skewness CI
# --------------------------------------------------

print(
    """
(vii) Pearson Skewness

We compute the Karl Pearson skewness coefficient

3(mean − median) / standard deviation

and construct a 95% bootstrap confidence interval using
1000 bootstrap samples.
"""
)


def pearson_skew(values):

    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values, ddof=1)

    return 3 * (mean - median) / std


skew_boot = []

for _ in range(1000):
    sample = rng.choice(df["happiness"], size=len(df), replace=True)

    skew_boot.append(pearson_skew(sample))

skew_boot = np.array(skew_boot)

lower_skew = np.percentile(skew_boot, 2.5)
upper_skew = np.percentile(skew_boot, 97.5)

print(
    f"""
The estimated skewness is {pearson_skew(df["happiness"]):.3f}.

The 95% bootstrap confidence interval is
({lower_skew:.3f}, {upper_skew:.3f}).
"""
)

plt.figure(figsize=(6, 4))

plt.hist(skew_boot, bins=40)

plt.xlabel("Bootstrap Skewness")
plt.ylabel("Frequency")
plt.title("Bootstrap Distribution of Pearson Skewness")

plt.savefig("homework_1/plots/skewness_bootstrap.png", bbox_inches="tight")
plt.close()


# --------------------------------------------------
# (viii) Closing discussion
# --------------------------------------------------

print(
    """
(viii) Conclusion

The analysis provides estimates of average happiness and the
proportion of respondents reporting the highest happiness level.

Group comparisons indicate differences in happiness across sex
categories. The correlation analysis suggests whether marital
status and happiness are associated.

Bootstrap methods were used to quantify uncertainty in the
correlation and skewness estimates.

Overall, the analysis illustrates how statistical estimation
methods can be used to understand patterns in categorical
survey data and identify demographic groups associated with
higher happiness levels.
"""
)
