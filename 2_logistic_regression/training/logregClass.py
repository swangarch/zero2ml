#!/usr/bin/python3

import numpy as np
from numpy import ndarray as array
from train_util import normalize, prob_predict, preproc_onevsall, cal_grad, count_correct
from datetime import datetime
from pandas import DataFrame as dataframe


class logreg:
	"""Class to perform logistic regression."""

	def __init__(self, data: dataframe, feature_names:list, className: str, goal: str):
		"""Init logreg."""

		self.classname = className
		self.goal = goal
		catArr = np.array(data[className])
		self.features = []
		self.feature_names = feature_names
		for name in feature_names: # add check if length matched
			self.features.append(np.array(data[name]))
		self.catToTrain = preproc_onevsall(catArr, goal)

		self.ranges = [] #tuple list
		self.weights = [0.0] #scalar list
		self.fnorms = [] #array list

		for i, f in enumerate(self.features):
			self.ranges.append((float(min(f)), float(max(f)))) ##can we use?? min max save the range
			self.fnorms.append(normalize(f, self.ranges[i]))
			self.weights.append(0.0) #theta i

		self.length = len(self.catToTrain)
		print("--------------------initialization done-------------------------")


	def train(self, leanrning_rate=0.0005, max_iter=100000, Debug=False) -> None:
		"""Train algorithm for a feature."""

		startTime = datetime.now()
		learning_rate = leanrning_rate
		for i in range(max_iter):
			res = np.zeros(self.length, dtype=np.float32)
			res = prob_predict(res, self.weights, self.fnorms)
			gradient = cal_grad(res, self.catToTrain, self.fnorms, self.weights)
			self.weights_array = np.array(self.weights)
			self.weights_array -= (np.array(gradient) * learning_rate)
			self.weights = self.weights_array.astype(float).tolist()
			binary_arr = (res > 0.5).astype(int)

			if i % 500 == 0:
				
				print("\033[033m[ITER]", int(i), count_correct(binary_arr, self.catToTrain), "\033[0m")
				if Debug:
					print("\033[033m----------------------------------------------")
					print("\033[034m[GRAD]", gradient, "\033[0m")
					print("\033[035m[WEIS]",self.weights_array, "\033[0m")
					print("\033[032m[RESS]",res, "\033[0m")
					print("----------------------------------------------\033[0m")

		print("----------------------training done----------------------------")
		print(f"\033[031m[TRAINING TIME] {datetime.now() - startTime}\033[0m")
		print(f'\033[031m[WEIGHTS] {self.weights}\033[0m')
		return self.weights


	def predict(self, df_test: dataframe)-> array:
		"""Predict after training then compare with ground truth."""
		
		weights = self.weights
		ranges = self.ranges

		print(f"[Prediction {self.goal}]")
		fnorms = []
		features = []
		for name in self.feature_names: # add check if length matched
			features.append(np.array(df_test[name]))
		for i, f in enumerate(features):
			fnorms.append(normalize(f, ranges[i]))
		length = len(df_test)
		res = res = np.zeros(length, dtype=np.float32)
		prob = prob_predict(res, weights, fnorms)
		trueRes = (prob > 0.5).astype(int)
		print(trueRes)

		print(f"[Ground Truth {self.goal}]")
		house = df_test[self.classname]
		category = preproc_onevsall(house, self.goal)
		print(category)
		print(count_correct(trueRes, category))
		print()
		return prob
	
	def predict_new(self, df_test: dataframe)-> array:
		"""Predict after training then compare with ground truth."""
		
		weights = self.weights
		ranges = self.ranges

		fnorms = []
		features = []
		for name in self.feature_names: # add check if length matched
			features.append(np.array(df_test[name]))
		for i, f in enumerate(features):
			fnorms.append(normalize(f, ranges[i]))
		length = len(df_test)
		res = res = np.zeros(length, dtype=np.float32)
		prob = prob_predict(res, weights, fnorms)
		return prob


def save_weights(weight: array, path:str) -> None:
	"""Save the weights."""

	try:
		np.savetxt(path, weight, delimiter=",", fmt="%f")
	
	except Exception as e:
		print("Error:", e)