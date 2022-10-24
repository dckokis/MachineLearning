import numpy as np


def TP(y_pred: np.array, y_gt: np.array):
	return np.count_nonzero(np.logical_and(y_pred == 1, y_gt == 1))


def FP(y_pred: np.array, y_gt: np.array):
	return np.count_nonzero(np.logical_and(y_pred == 0, y_gt == 1))


def FN(y_pred: np.array, y_gt: np.array):
	return np.count_nonzero(np.logical_and(y_pred == 1, y_gt == 0))


def TN(y_pred: np.array, y_gt: np.array):
	return np.count_nonzero(np.logical_and(y_pred == 0, y_gt == 0))


def precision(y_pred: np.array, y_gt: np.array):
	TruePositive = TP(y_pred, y_gt)
	FalsePositive = FP(y_pred, y_gt)
	return TruePositive / (TruePositive + FalsePositive)


def recall(y_pred: np.array, y_gt: np.array):
	TruePositive = TP(y_pred, y_gt)
	FalseNegative = FN(y_pred, y_gt)
	return TruePositive / (TruePositive + FalseNegative)


def F1(y_pred: np.array, y_gt: np.array):
	PR = precision(y_pred, y_gt)
	REC = recall(y_pred, y_gt)
	return 2 * PR * REC / (PR + REC)


def TPR(y_pred: np.array, y_gt: np.array):
	return recall(y_pred, y_gt)


def FPR(y_pred: np.array, y_gt: np.array):
	FalsePositive = FP(y_pred, y_gt)
	TrueNegative = TN(y_pred, y_gt)
	return FalsePositive / (FalsePositive + TrueNegative)
