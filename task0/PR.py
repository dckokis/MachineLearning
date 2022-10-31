import metrics as m
import numpy as np


def PR(probabilities, ground_truth):
	thresholds = np.unique(probabilities)
	rec = np.zeros_like(thresholds)
	prec = np.zeros_like(thresholds)
	for i, threshold in enumerate(sorted(thresholds)):
		preds = probabilities >= threshold
		rec[i] = m.recall(preds, ground_truth)
		prec[i] = m.precision(preds, ground_truth)
	return np.concatenate([prec, [1.]]), np.concatenate([rec, [.0]])
