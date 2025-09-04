#!/usr/bin/python3

import pandas as pd
import sys
from load_csv import load
import matplotlib.pyplot as plt


def visualize_hist(data: pd.DataFrame, feature:str) -> None:
	"""Visualize data."""
	
	try:
		plt.hist(data[feature].values, bins=100, rwidth=1)
		title = f"histogram:{feature}"
		plt.title(title)
		plt.xlabel(feature)
		plt.ylabel("Value")
		plt.show()
	except Exception as e:
		print("Error in scatter visualizer:", e)


def plot_scatter(data: pd.DataFrame) -> None:
	"""Show all the pair plots."""

	features = data.iloc[0].index
	# print(data.iloc[0]) # debug

	count = 0
	for f1 in features:
		count += 1
		print(f"Histogram {count}: {f1}")
		visualize_hist(data, f1)
	
	print("\033[33mDone.\033[0m")
	plt.show()


def main():
	"""Program to visualize data."""

	try:
		print("\033[33mUsage: python3 histogram.py <path_csv>\033[0m")
		argv = sys.argv
		assert len(argv) == 2 or len(argv ), "Wrong argument number."
		df = load(argv[1])
		if df is None:
			sys.exit(1)

		df_num = df.select_dtypes(include="number")
		plot_scatter(df_num)

	except KeyboardInterrupt:
		print("\033[33mStopped by user.\033[0m")
		sys.exit(1)

	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()
