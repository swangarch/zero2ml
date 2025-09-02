import pandas as pd
import sys
from load_csv import load


def main():
	"""Test of reading data and print it."""

	try:
		argv = sys.argv
		assert len(argv) == 2, "Wrong argument number."

		df = load(argv[1])
		if df is None:
			sys.exit(1)

		print(df)
		print("\033[33m", df.describe(), "\033[0m")  ##remove this, forbidden!!!!!!!!!!!!!!!!
 
		df_num = df.select_dtypes(include="number")
		# print(df_num.sum())
		print(df_num)
	
	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()