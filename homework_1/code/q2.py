from numpy import log
from numpy.random import binomial

def get_risk_ratio(n1: int, n2: int) -> None:
    y1 = binomial(n1, 0.20, 1_000_000)
    y2 = binomial(n2, 0.10, 1_000_000)

    pi1_hat = y1 / 50
    pi2_hat = y2 / 50

    print(f"risk_ratio = {float(pi1_hat / pi2_hat)}")
    print(f"log_risk_ratio = {float(log(pi1_hat / pi2_hat))}")

import matplotlib.pyplot as plt
from math import sqrt

def histogram(n: int, pi: float, fname: str) -> None:
    x = binomial(n, pi, 1_000_000)
    plt.hist(x, bins=int(sqrt(len(x))))
    plt.savefig(f"homework 1/plots/{fname}.png")
    plt.close()

get_risk_ratio(50, 50)

