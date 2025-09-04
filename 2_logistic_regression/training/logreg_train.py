#!/usr/bin/python3

from load_csv import load
import pandas as pd
import sys
import numpy as np
from numpy import ndarray as array
import math
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

	print("INIT RES", res)
	for i, f in enumerate(fnorm):
		res += weight[i + 1] * f
		print("RES", res)
	res += weight[0]

	res = 1.0 / (1 + np.e ** (-res)) #sigmoid
	return res


def cal_grad(prediction: array, groundTruth: array, features: list, weight:list) -> list:
	"""Calculate the gradient of loss to weight"""

	diff = prediction - groundTruth
	gradient = []
	length = len(prediction)
	for i,t in enumerate(weight):
		if i == 0: #bias
			theta0 = float((np.nansum(diff) / length))
			gradient.append(theta0)
		else: #weights
			theta = float((np.nansum(diff * features[i - 1]) / length))
			gradient.append(theta)

	print("\033[33m[GRADIENT]", gradient, "\033[0m")
	return gradient


def train(features: array, results: array) -> None:
	"""Train algorithm for a feature"""

	print("----------------------init----------------------------")
	li_range = [] #tuple list
	li_weight = [] #scalar list
	li_fnorm = [] #array list
	length = len(results)
	li_weight.append(0.0) #theta 0  bias
	for i, f in enumerate(features):
		li_range.append((float(min(f)), float(max(f)))) ##can we use?? min max save the range
		li_fnorm.append(normalize(f, li_range[i]))
		li_weight.append(0.0) #theta i
	
	print("RANGES:  ", li_range)
	print("--------------------init done-------------------------")

	print("----------------------main train----------------------------")
	

	learning_rate = 0.001
	for i in range(10000):
		res = np.zeros(length, dtype=np.float32)
		res = predict(res, li_weight, li_fnorm)
		gradient = cal_grad(res, results, li_fnorm, li_weight)
		
		li_weight_array = np.array(li_weight)
		print("LI",li_weight_array)
		li_weight_array -= (np.array(gradient) * learning_rate)
		li_weight = li_weight_array.astype(float).tolist()

		binary_arr = (res > 0.5).astype(int)

		count = 0
		for i, num in enumerate(results):
			if (num == binary_arr[i]):
				count += 1

		print(res.tolist())
		print("<CORRECT NUMBER>", count)
	
	print("----------------------train done----------------------------")


def main():
	"""Main to train."""

	print("\033[33mUsage: python3 describe.py <path_csv>\033[0m")
	argv = sys.argv
	assert len(argv) == 2, "Wrong argument number."

	pd.set_option('display.float_format', '{:.6f}'.format)
	df = load(argv[1])
	if df is None:
		sys.exit(1)
	# print(df)

	feature1 = np.array(df["Astronomy"])
	feature2 = np.array(df["Herbology"])
	house = np.array(df["Hogwarts House"])

	if len(feature1) != len(house) or len(feature2) != len(house):
		raise("Mismatched length")

	class1 = [1 if h == "Ravenclaw" else 0 for h in house]

	train([feature1, feature2], class1)

	# try:
	# 	print("\033[33mUsage: python3 describe.py <path_csv>\033[0m")
	# 	argv = sys.argv
	# 	assert len(argv) == 2, "Wrong argument number."

	# 	pd.set_option('display.float_format', '{:.6f}'.format)
	# 	df = load(argv[1])
	# 	if df is None:
	# 		sys.exit(1)
	# 	# print(df)

	# 	feature1 = np.array(df["Astronomy"])
	# 	feature2 = np.array(df["Herbology"])
	# 	house = np.array(df["Hogwarts House"])

	# 	if len(feature1) != len(house) or len(feature2) != len(house):
	# 		raise("Mismatched length")

	# 	class1 = [1 if h == "Ravenclaw" else 0 for h in house]

	# 	train([feature1, feature2], class1)

	# except KeyboardInterrupt:
	# 	print("\033[33mStopped by user.\033[0m")
	# 	sys.exit(1)

	# except Exception as e:
	# 	print("Error:", e)


if __name__ == "__main__":
	main()
