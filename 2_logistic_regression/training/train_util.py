#!/usr/bin/python3

from load_csv import load
import pandas as pd
import sys
import numpy as np
from numpy import ndarray as array
import copy as cp


def normalize(li: array, range: tuple) -> array:
	"""Normalize list value, mapped into range of 0:1"""

	#use numpy way will be more efficient
	return np.array([(num - range[0]) / (range[1] - range[0]) for num in li])


def cross_entropy_loss(predict:array, truth:array) -> None:
	"""Calculate the loss"""

	pass


def predict(res:array, weight:list, fnorm:list) -> array:
	"""Calculate the training prediction"""

	for i, f in enumerate(fnorm):
		res += weight[i + 1] * f
	res += weight[0]

	res = 1.0 / (1 + np.e ** (-res)) #sigmoid
	return res


def preprocess_class_onevsall(arr:array, className: str) -> array:
	"""preprocess class one vs all"""

	return np.array([1 if h == className else 0 for h in arr])


def cal_grad(prediction: array, groundTruth: array, features: list, weight:list) -> list:
	"""Calculate the gradient of loss to weight"""

	diff = prediction - groundTruth
	gradient = []
	length = len(prediction)
	for i in range(len(weight)):
		if i == 0: #bias
			theta0 = float((np.nansum(diff) / length))
			gradient.append(theta0)
		else: #weights
			theta = float((np.nansum(diff * features[i - 1]) / length))
			gradient.append(theta)

	return gradient


def count_correct(prediction:array, truth: array, length: int) -> None:
	"""Count prediction truth"""

	count = 0
	for i, num in enumerate(truth):
		if (num == prediction[i]):
			count += 1

	return f"<CORRECT> {count}   <RATE> {count / float(length) * 100:2f}%"