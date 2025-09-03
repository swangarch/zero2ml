import pandas as pd
import sys
from load_csv import load
import matplotlib.pyplot as plt


def visualize_scatter(data: pd.DataFrame, feature1:str, feature2:str) -> None:
	"""Visualize data."""
	
	try:
		plt.scatter(data[feature1].values, data[feature2].values, s=1)
		title = f"Scatter:{feature1} * {feature2}"
		plt.title(title)
		plt.xlabel(feature1)
		plt.ylabel(feature2)
		plt.show()
	except Exception as e:
		print("Error in scatter visualizer:", e)


# def visualize_scatter(data: pd.DataFrame, feature1:str, feature2:str, length:int, pos: int) -> None:
# 	"""Visualize pair plot data."""
	
# 	try:
# 		plt.subplot(length, length, pos)
# 		plt.scatter(data[feature1].values, data[feature2].values, s=1)
# 		# title = f"Scatter:{feature1} * {feature2}"
# 		# plt.title(title)
# 		# plt.xlabel(feature1)
# 		# plt.ylabel(feature2)
# 	except Exception as e:
# 		print("Error in scatter visualizer:", e)


def plot_scatter(data: pd.DataFrame) -> None:
	"""Show all the pair plots."""

	features = data.iloc[0].index
	print(data.iloc[0]) # debug

	num_features = len(features)
	count = 0
	for f1 in features:
		for f2 in features:
			count += 1
			if f1 != f2:
				# print(f1, f2)
				visualize_scatter(data, f1, f2)
	plt.show()


def main():
	"""Program to visualize data."""

	try:
		argv = sys.argv
		assert len(argv) == 2 or len(argv ), "Wrong argument number."
		df = load(argv[1])
		if df is None:
			sys.exit(1)

		df_num = df.select_dtypes(include="number")
		plot_scatter(df_num)

	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()

