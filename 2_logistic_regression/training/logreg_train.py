#!/usr/bin/python3

from load_csv import load
import pandas as pd
import sys
import numpy as np
from numpy import ndarray as array
import copy as cp

from training import set_train_goal


def main():
	"""Main to train."""

	print("\033[33mUsage: python3 describe.py <path_csv>\033[0m")
	argv = sys.argv
	assert len(argv) == 2, "Wrong argument number."

	pd.set_option('display.float_format', '{:.6f}'.format)
	df = load(argv[1])
	if df is None:
		sys.exit(1)
	
	# feature_names_irrel = [
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
	
	class_name = "Hogwarts House"

	set_train_goal(df, feature_names, class_name, "Ravenclaw")
	set_train_goal(df, feature_names, class_name, "Gryffindor")
	set_train_goal(df, feature_names, class_name, "Slytherin")
	set_train_goal(df, feature_names, class_name, "Hufflepuff")


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