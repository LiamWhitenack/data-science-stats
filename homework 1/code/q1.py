from numpy import mean
from itertools import combinations
from pandas import DataFrame


travel_distance = {
    "H1": 35,
    "H2": 50,
    "H3": 20,
    "H4": 8,
    "HA": 5,
    "HB": 15,
    "HC": 22,
    "HD": 58,
    "HE": 35,
    "HF": 50,
}
mpg_for_car_used = {
    "H1": 25,
    "H2": 20,
    "H3": 10,
    "H4": 26,
    "HA": 15,
    "HB": 20,
    "HC": 10,
    "HD": 30,
    "HE": 24,
    "HF": 27,
}

# second block sampling design
second_block_probabilities = {
    ("HA", "HC"): 1 / 4,
    ("HB", "HD"): 7 / 16,
    ("HA", "HE"): 3 / 16,
    ("HC", "HF"): 1 / 8,
}


first_block_houses = ["H1", "H2", "H3", "H4"]
first_block_samples = list(combinations(first_block_houses, 3))
first_block_probability = 1 / 4


def trimmed_mean_20(values):

    sorted_values = sorted(values)

    # remove smallest and largest
    trimmed = sorted_values[1:-1]

    return sum(trimmed) / len(trimmed)


rows = []

for first_sample in first_block_samples:

    for second_sample, second_probability in second_block_probabilities.items():

        sample = list(first_sample) + list(second_sample)

        probability = first_block_probability * second_probability

        distances = [travel_distance[h] for h in sample]
        mpgs = [mpg_for_car_used[h] for h in sample]

        tsm_distance = trimmed_mean_20(distances)
        tsm_mpg = trimmed_mean_20(mpgs)

        confidence_interval = (tsm_distance - 6, tsm_distance + 6)

        rows.append(
            {
                "Sample": sample,
                "P(S)": probability,
                "Values": distances,
                "tsm(.20) Distance": tsm_distance,
                "tsm(.20) MPG": tsm_mpg,
                "Conf Interval": confidence_interval,
            }
        )

table = DataFrame(rows)



# Is tsm.20 unbiased for the population average commute distance? If not, what is the
# bias?
expected_tsm_distance = (table["tsm(.20) Distance"] * table["P(S)"]).sum()
true_average = mean(list(travel_distance.values()))

bias_distance_mean = expected_tsm_distance - true_average

# Is tsm.20 unbiased for the 20% trimmed population commute distance? If not, what is
# the bias?
true_tsm_20_average = mean(sorted(travel_distance.values())[2:8])
bias_distance_trimmed = expected_tsm_distance - true_tsm_20_average

# What is the true sampling variance for tsm.20 for estimating commuter distance under
# the proposed sampling design?
variance_distance = (
    table["P(S)"]
    * (table["tsm(.20) Distance"] - expected_tsm_distance) ** 2
).sum()

# What is the confidence level for the confidence interval tsm.20 ± 6 for estimating the
# true mean commuting distance?
coverage = table.apply(
    lambda row: row["P(S)"]
    if (true_average >= row["tsm(.20) Distance"] - 6)
    and (true_average <= row["tsm(.20) Distance"] + 6)
    else 0,
    axis=1,
).sum()

# Repeat (a) through (d) for average MPG. For part (d) what is the confidence level for
# the confidence interval tsm.20 ± 4 for estimating the true mean MPG?
true_mpg_mean = mean(list(mpg_for_car_used.values()))

true_mpg_trimmed = mean(sorted(mpg_for_car_used.values())[2:8])

expected_tsm_mpg = (table["tsm(.20) MPG"] * table["P(S)"]).sum()

bias_mpg_mean = expected_tsm_mpg - true_mpg_mean

bias_mpg_trimmed = expected_tsm_mpg - true_mpg_trimmed

variance_mpg = (
    table["P(S)"]
    * (table["tsm(.20) MPG"] - expected_tsm_mpg) ** 2
).sum()

coverage_mpg = table.apply(
    lambda row: row["P(S)"]
    if (true_mpg_mean >= row["tsm(.20) MPG"] - 4)
    and (true_mpg_mean <= row["tsm(.20) MPG"] + 4)
    else 0,
    axis=1,
).sum()

print("\n--- Distance Results ---\n")

print(f"E(tsm_.20) = {expected_tsm_distance}")
print(f"Population mean = {true_average}")
print(f"Bias for population mean = {bias_distance_mean}")

print()

print(f"20% trimmed population mean = {true_tsm_20_average}")
print(f"Bias for trimmed population mean = {bias_distance_trimmed}")

print()

print(f"True sampling variance = {variance_distance}")

print()

print(f"Confidence level for tsm(.20) ± 6 = {coverage}")

print("\n--- MPG Results ---\n")

print(f"E(tsm_.20 MPG) = {expected_tsm_mpg}")
print(f"Population mean MPG = {true_mpg_mean}")
print(f"Bias for MPG population mean = {bias_mpg_mean}")

print()

print(f"20% trimmed MPG population mean = {true_mpg_trimmed}")
print(f"Bias for trimmed MPG mean = {bias_mpg_trimmed}")

print()

print(f"True sampling variance (MPG) = {variance_mpg}")

print()

print(f"Confidence level for tsm(.20) ± 4 MPG = {coverage_mpg}")

pass