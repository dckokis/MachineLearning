import metrics as m
import numpy as np


def ROC(probabilities, ground_truth):
	fpr = np.zeros_like(probabilities)
	tpr = np.zeros_like(probabilities)
	thresholds = sorted(probabilities.copy())
	for i, threshold in enumerate(thresholds):
		try:
			pred = probabilities >= threshold
			fpr[i] = m.FPR(pred, ground_truth)
			tpr[i] = m.TPR(pred, ground_truth)
		except ZeroDivisionError:
			continue
	return np.concatenate([[0.], fpr, [1.]]), np.concatenate([[0.], tpr, [1.]])
