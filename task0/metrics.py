import numpy as np


def TP(y_pred: np.array, y_gt: np.array, ):
	return np.add.reduce((y_pred[:] >= 1.) & (y_gt == 1))


def FP(y_pred: np.array, y_gt: np.array, ):
	return np.add.reduce((y_pred[:] >= 1.) & (y_gt == 0))


def FN(y_pred: np.array, y_gt: np.array):
	return np.add.reduce((y_pred[:] < 1.) & (y_gt == 1))


def TN(y_pred: np.array, y_gt: np.array):
	return np.add.reduce((y_pred[:] < 1.) & (y_gt == 0))


def precision(y_pred: np.array, y_gt: np.array):
	TruePositive = TP(y_pred, y_gt)
	FalsePositive = FP(y_pred, y_gt)
	try:
		prec = TruePositive / (TruePositive + FalsePositive)
	except ZeroDivisionError:
		prec = 0
	return prec


def recall(y_pred: np.array, y_gt: np.array):
	TruePositive = TP(y_pred, y_gt)
	FalseNegative = FN(y_pred, y_gt)
	try:
		rec = TruePositive / (TruePositive + FalseNegative)
	except ZeroDivisionError:
		rec = 1
	return rec


def FPR(y_pred: np.array, y_gt: np.array):
	FalsePositive = FP(y_pred, y_gt)
	AllPositive = np.add.reduce(y_gt == 0)
	if AllPositive != 0:
		fpr = FalsePositive / AllPositive
	else:
		fpr = 1
	return fpr


def TPR(y_pred: np.array, y_gt: np.array):
	TruePositive = TP(y_pred, y_gt)
	tpr = TruePositive / np.add.reduce(y_gt == 1)
	return tpr
