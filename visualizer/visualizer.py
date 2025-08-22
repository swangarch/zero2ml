import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import sys

def predict(mileage: float, theta0: float, theta1: float) -> float:
    """Given theta and a mileage, pridict its price"""

    return theta0 + mileage * theta1

def draw_lr(process: subprocess.Popen, x: float, y: float) -> None:
    """This function will run training program in the subprocess, and visulize the result."""
    
    plt.ion()
    fig, ax = plt.subplots()

    for line in process.stderr:
        print("Error: Training program error: ", line.strip())
        sys.exit(1)
    
    for line in process.stdout:

        words = line.split(" ")
        theta0 = float(words[0])
        theta1 = float(words[1])
        ax.clear()

        ax.set_title("Car price prediction by mileage")
        ax.set_xlabel("Mileage")
        ax.set_ylabel("Price")

        xlr = [min(x), max(x)]
        ylr = [predict(min(x), theta0, theta1), predict(max(x), theta0, theta1)]
        ax.text(0.8, 0.8, f"NMSE: {float(words[2]):.4f}", transform=ax.transAxes)
        ax.plot(xlr, ylr, color="orange", label="Prediction")
        ax.scatter(x, y, color="cadetblue", label="Real data")
        
        ax.legend(loc="upper right")
        plt.pause(0.01)
        print(line, end="")

    process.wait()  
    print("Done")
    plt.ioff()
    plt.show()

def main():
    """Main fucntion will check args number, read and parse csv, then it will run the traning program, and visulize the training process."""

    args = sys.argv
    if len(args) != 3:
        print("Error: Wrong argv number, usage: <linear_regression_program_path> <data_csv_path>")
        sys.exit(1)

    try:
        data = pd.read_csv(args[2])

        mileages = list(data['km'])
        prices = list(data['price'])
    except Exception as e:
        print("Error: Failed to read training data", e)
        sys.exit(1)

    if len(mileages) != len(prices):
        print("Error: Mismatched data in datatable")
        sys.exit(1)

    try:
        process = subprocess.Popen(
            [args[1], args[2]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        print("Error: Failed to run tranning program", e)
        sys.exit(1)

    try:
        draw_lr(process, mileages, prices)
    except Exception as e:
        print("Error: ", e)
        sys.exit(1)


if __name__ == "__main__":
    main()