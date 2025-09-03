import pandas as pd
import sys
from load_csv import load


def ft_describe(series: pd.Series, title: str, all: dict) -> None:
	"""Function to describe the data series."""

	# try:
	sum = 0.0
	count = 0
	mean = 0.0
	sortedLi = sorted(series)
	err = 0.0
	cleanLi = []
	for elem in sortedLi: 
		if elem == elem:
			sum += elem
			cleanLi.append(elem)

	count = len(cleanLi)
	mean = sum / count

	for num in cleanLi:
		err += (num - mean) ** 2
	
	std = (err / count) ** 0.5
	
	min_num = cleanLi[0]
	max_num = cleanLi[-1]

	percent25 = cleanLi[int((count) / 4)]
	percent50 = cleanLi[int((count) / 2)]
	percent75 = cleanLi[int((count) / 4 * 3)]


	all[title] = [count, mean, std, min_num, percent25, percent50, percent75, max_num]


def main():
	"""Test of reading data and print it."""

	try:
		argv = sys.argv
		assert len(argv) == 2, "Wrong argument number."

		pd.set_option('display.float_format', '{:.6f}'.format)
		df = load(argv[1])
		if df is None:
			sys.exit(1)
		print(df)
		print("--------------------pd describe----------------------")
		print("\033[33m", df.describe(), "\033[0m")  ##remove this, forbidden!!!!!!!!!!!!!!!!
 
		df_num = df.select_dtypes(include="number")
		
		all = {"Feature": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "max"]}
		
		for col in df_num:
			ft_describe(df_num[col], col, all)
		
		print("--------------------my describe----------------------")

		df_describ = pd.DataFrame(all)
		df_describ = df_describ.set_index("Feature")
		df_describ.index.name = None
		print(df_describ)
	
	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()