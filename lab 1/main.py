from statistics import mean, median, stdev
from collections.abc import Iterable
from random import randint
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame

sbp_mmHg: list[float] = read_csv("lab 1/data.csv")["sbp_mmHg"].to_list()

def get_sample(input_list: list[float], n: int = 30) -> Iterable[float]:
    for _ in range(n):
        yield input_list.pop(randint(0, len(input_list) - 1))

def summary_stats(data: list[float]) -> tuple[float, float, float, float]:
    return mean(data), median(data), stdev(data), len([1 for var in data if var > 1.4]) / len(data)

def compare_distributions(sbp_mmHg: list[float]) -> None:
    plt.hist(sbp_mmHg)
    plt.savefig("lab 1/sample_sizes/all.png")
    for column in ("mean", "median", "stdev", "proportion"):
        for sample_size in (10, 30, 50, 100):
            this_spb = sbp_mmHg.copy()
            plt.close()
            df = DataFrame([summary_stats(list(get_sample(this_spb, sample_size))) for _ in range(800)], columns=["mean", "median", "stdev", "proportion"])
            pass

compare_distributions(sbp_mmHg)

