#!/usr/bin/python3

# from load_csv import load
import pandas as pd
import sys
import numpy as np
from numpy import ndarray as array
import copy as cp

from train_util import normalize, predict, preprocess_class_onevsall, cal_grad, count_correct


def set_train_goal(df: pd.DataFrame, feature_names:list, class_obj: str, goal: str):
	"""set traning goal"""

	house = np.array(df[class_obj])
	features = []
	for name in feature_names: # add check if length matched
		features.append(np.array(df[name]))
	class1 = preprocess_class_onevsall(house, goal)
	train(features, class1)


def train(features: array, results: array) -> None:
	"""Train algorithm for a feature"""

	DEBUG = False

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
	print("--------------------init done-------------------------")


	print("----------------------main train----------------------------")
	learning_rate = 0.001
	for i in range(100000):
		res = np.zeros(length, dtype=np.float32)
		res = predict(res, li_weight, li_fnorm)
		gradient = cal_grad(res, results, li_fnorm, li_weight)
		li_weight_array = np.array(li_weight)
		li_weight_array -= (np.array(gradient) * learning_rate)
		li_weight = li_weight_array.astype(float).tolist()
		binary_arr = (res > 0.5).astype(int)

		if i % 50 == 0:
			
			print("\033[033m[EPOCH]", int(i / 50), count_correct(binary_arr, results, length), "\033[0m")
			if DEBUG:
				print("\033[033m----------------------------------------------")
				print("\033[034m[GRAD]", gradient, "\033[0m")
				print("\033[035m[WEIS]",li_weight_array, "\033[0m")
				print("\033[032m[RESS]",res, "\033[0m")
				print("----------------------------------------------\033[0m")
			
	print("----------------------train done----------------------------")


