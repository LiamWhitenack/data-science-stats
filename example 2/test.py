from matplotlib import pyplot as plt
from numpy import array, mean, percentile, sum, zeros
from numpy.random import randint, seed
from pandas import read_excel
from scipy.stats import norm

seed(2026)


def calc_skew(x):
    n = len(x)
    m3 = sum((x - mean(x)) ** 3) / n
    m2 = sum((x - mean(x)) ** 2) / n
    return m3 / (m2 ** (3 / 2))


def bias_corrected_ci(x, boot):
    theta_hat = calc_skew(x)

    prop_less = mean(boot < theta_hat)
    z0 = norm.ppf(prop_less)

    alpha1 = norm.cdf(2 * z0 + norm.ppf(0.025))
    alpha2 = norm.cdf(2 * z0 + norm.ppf(0.975))

    lower = percentile(boot, alpha1 * 100)
    upper = percentile(boot, alpha2 * 100)

    return lower, upper


def bootstrap_skew(x, B=5000):
    n = len(x)
    boot = zeros(B)

    for b in range(B):
        indices = randint(0, n, n)
        resample = x[indices]
        boot[b] = calc_skew(resample)

    return boot


def percentile_ci(boot):
    lower = percentile(boot, 2.5)
    upper = percentile(boot, 97.5)
    return lower, upper


df = read_excel("example 2/SurveyCompTimes.xlsx")

completion_times = df["CompTime"]

boot = bootstrap_skew(completion_times, 5000)

perc_ci = percentile_ci(boot)
bc_ci = bias_corrected_ci(completion_times, boot)
plt.hist(boot)

print("Percentile CI:", perc_ci)
print("Bias-corrected CI:", bc_ci)
