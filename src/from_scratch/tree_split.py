"""W03 (release catch-up). One impurity/split routine for decision trees.

Interface (Appendix A):
    entropy(y) -> float                 Shannon entropy in bits of the label array y.
    gini(y) -> float                    Gini impurity of y.
    information_gain(y, mask) -> float  Entropy of y minus the weighted entropy of y[mask], y[~mask].
    best_threshold(x, y) -> (t, gain)   Best split x <= t on one continuous feature by information
                                        gain; t is a midpoint between two consecutive distinct values.
                                        Return (None, 0.0) when x has a single distinct value.
"""
import numpy as np


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p))


def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p ** 2)


def information_gain(y, mask):
    n = len(y)
    y_left = y[mask]
    y_right = y[~mask]

    weighted_entropy = 0.0

    if len(y_left) > 0:
        weighted_entropy += (len(y_left) / n) * entropy(y_left)

    if len(y_right) > 0:
        weighted_entropy += (len(y_right) / n) * entropy(y_right)

    return entropy(y) - weighted_entropy


def best_threshold(x, y):
    values = np.unique(x)

    if len(values) == 1:
        return None, 0.0

    thresholds = (values[:-1] + values[1:]) / 2

    best_t = None
    best_gain = -np.inf

    for t in thresholds:
        mask = x <= t
        gain = information_gain(y, mask)

        if gain > best_gain:
            best_gain = gain
            best_t = t

    return best_t, best_gain
