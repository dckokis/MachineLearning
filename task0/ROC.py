import metrics as m
import numpy as np


def ROC(probabilities, ground_truth):
	thresholds = np.insert(np.unique(probabilities), 0, 2)
	fpr = np.zeros_like(thresholds)
	tpr = np.zeros_like(thresholds)
	thresholds = sorted(thresholds)
	for i, threshold in enumerate(thresholds):
		preds = probabilities >= threshold
		fpr[i] = m.FPR(preds, ground_truth)
		tpr[i] = m.TPR(preds, ground_truth)
	return fpr, tpr
