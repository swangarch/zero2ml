import pandas as pd
import sys
from load_csv import load


def ft_describe(series: pd.Series, title: str) -> None:
	"""Function to describe the data series."""

	try:
		sum = 0.0
		count = 0.0
		mean = 0.0
		min = float('inf')
		max = float('-inf')
		for element in series: 
			if element == element:
				if (element < min):
					min = element
				if (element > max):
					max = element
				count += 1
				sum += element

		mean = sum / count
		print(f"[{title}]")
		# print(f"Sum {sum:6f}")
		print(f"Count {count:6f}")
		print(f"Mean {mean:6f}")
		print(f"Min {min:6f}")
		print(f"Max {max:6f}")
		print()
	except Exception as e:
		print("Error:", e)

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

		for col in df_num:
			ft_describe(df_num[col], col)

		# print(df_num)
	
	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()