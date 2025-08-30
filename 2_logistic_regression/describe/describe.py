import pandas as pd
import sys

def main():
	"""Test of reading data and print it."""

	try:
		argv = sys.argv

		data = pd.read_csv(argv[1])
		df = pd.DataFrame(data)

		print(df)
		# print(df.describe())  ##remove this
	
	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()