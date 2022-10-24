import metrics as m
import numpy as np


def PR(probabilities, ground_truth):
    rec = np.zeros_like(probabilities)
    prec = np.zeros_like(probabilities)
    for i, threshold in enumerate(sorted(probabilities.copy())):
        pred = probabilities >= threshold
        rec[i] = m.recall(pred, ground_truth)
        prec[i] = m.precision(pred, ground_truth)
    return np.concatenate([[1.], prec]), np.concatenate([[.0], rec])
