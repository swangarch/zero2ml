#!/usr/bin/python3

import pandas as pd
import sys
from load_csv import load
import matplotlib.pyplot as plt


def visualize_hist(data: pd.DataFrame, feature:str, length:int, pos: int) -> None:
	"""Visualize data."""
	
	try:
		plt.subplot(length, length, pos)
		plt.hist(data[feature].values, bins=100, rwidth=1, color='red')
		plt.xticks([])
		plt.yticks([])
	except Exception as e:
		print("Error in histogram visualizer:", e)


def visualize_scatter(data: pd.DataFrame, feature1:str, feature2:str, length:int, pos: int) -> None:
	"""Visualize pair plot data."""
	
	try:
		plt.subplot(length, length, pos)
		plt.scatter(data[feature1].values, data[feature2].values, s=0.1)
		plt.xticks([])
		plt.yticks([])
	except Exception as e:
		print("Error in scatter visualizer:", e)


def pair_plot_scatter(data: pd.DataFrame) -> None:
	"""Show all the pair plots."""

	features = data.iloc[0].index
	num_features = len(features)
	count = 0
	for f1 in features:
		for f2 in features:
			count += 1
			if f1 != f2:
				visualize_scatter(data, f1, f2, num_features, count)
			else:
				visualize_hist(data, f1, num_features, count)
	plt.show()


def main():
	"""Program to visualize data."""

	try:
		print("\033[33mUsage: python3 pair_plot.py <path_csv>\033[0m")
		argv = sys.argv
		assert len(argv) == 2, "Wrong argument number."
		df = load(argv[1])
		if df is None:
			sys.exit(1)

		df_num = df.select_dtypes(include="number")
		pair_plot_scatter(df_num)

	except KeyboardInterrupt:
		print("\033[33mStopped by user.\033[0m")
		sys.exit(1)

	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()

