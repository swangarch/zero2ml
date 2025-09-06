#!/usr/bin/python3

from load_csv import load
import pandas as pd
import sys
from pandas import DataFrame as dataframe
from logregallClass import logregall
import numpy as np
from numpy import ndarray as array


def clean_data(df: dataframe) -> dataframe:
	"""Clean data, drop nan value line"""

	return df.dropna()


def split_data(df: dataframe) -> tuple:
	"""Split the data into test data and validation data"""

	train_df = df.sample(frac=0.8, random_state=412)
	test_df = df.drop(train_df.index)
	return train_df, test_df


def save_weights(weight: array):
	"""Save the weights."""

	try:
		np.savetxt("weight.csv", weight, delimiter=",", fmt="%f")
	
	except Exception as e:
		print("Error:", e)


def main(): # add try catch
	"""Main to load dataset and train."""

	print("\033[33mUsage: python3 describe.py <path_csv>\033[0m")
	argv = sys.argv
	assert len(argv) == 2, "Wrong argument number."
	pd.set_option('display.float_format', '{:.6f}'.format)


	df = load(argv[1])
	df = clean_data(df)
	df, df_test = split_data(df)
	
	# feature_names = [
	# 				 "Astronomy", "Herbology", 
	# 			     "Arithmancy", "Charms", 
	# 				 "Divination", "Ancient Runes",
	# 				 "Defense Against the Dark Arts",
	# 				 "Muggle Studies", "History of Magic",
	# 				 "Transfiguration", "Potions",
	# 				 "Care of Magical Creatures",
	# 				 "Flying",
	# 				]
	
	feature_names = [
					 "Astronomy", "Herbology", 
				     "Charms", 
					 "Divination", "Ancient Runes",
					 "Defense Against the Dark Arts",
					 "Muggle Studies", "History of Magic",
					 "Transfiguration",
					 "Flying",
					]
	
	classname = "Hogwarts House"
	goals = ["Ravenclaw", "Gryffindor", "Slytherin", "Hufflepuff"]


	lgall = logregall(df, feature_names, classname, goals)
	lgall.train_all()
	lgall.predict_test(df_test)

	df_new = load("dataset_test.csv")

	lgall.predict_new(df_new)


if __name__ == "__main__":
	main()



# def main():
# 	"""Main to train."""

# 	print("\033[33mUsage: python3 describe.py <path_csv>\033[0m")
# 	argv = sys.argv
# 	assert len(argv) == 2, "Wrong argument number."

# 	pd.set_option('display.float_format', '{:.6f}'.format)
# 	df = load(argv[1])
# 	if df is None:
# 		sys.exit(1)
	

# 	feature1 = np.array(df["Astronomy"])
# 	feature2 = np.array(df["Herbology"])
# 	house = np.array(df["Hogwarts House"])

# 	if len(feature1) != len(house) or len(feature2) != len(house):
# 		raise("Mismatched length")

# 	class1 = [1 if h == "Ravenclaw" else 0 for h in house]

# 	train([feature1, feature2], class1)

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