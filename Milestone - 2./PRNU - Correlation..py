import numpy as np

def normalized_correlation(a, b):
    """
    Normalized cross-correlation between two PRNU residuals.
    """

    a = a.flatten()
    b = b.flatten()

    numerator = np.sum(a * b)
    denominator = np.sqrt(np.sum(a**2) * np.sum(b**2))

    if denominator == 0:
        return 0.0

    return numerator / denominator
