import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import sys

def predict(mileage, p0, p1):
    return p0 + mileage * p1

def draw_lr(process, x, y):
    plt.ion()
    fig, ax = plt.subplots()
    
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

    args = sys.argv
    if len(args) != 3:
        print("Error: Wrong argv number usage: <path linear_regression_program> <data data.csv>")
        sys.exit(1)

    try:
        data = pd.read_csv(args[2])

        mileages = list(data['km'])
        prices = list(data['price'])
    except Exception as e:
        print("Error: Failed to read training data.", e)
        sys.exit(1)

    try:
        process = subprocess.Popen(
            [args[1], args[2]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except:
        print("Error: Failed to open tranning program, make sure you use right linear_regression.")
        sys.exit(1)

    try:
        draw_lr(process, mileages, prices)
    except Exception as e:
        print("Error: ", e)
        sys.exit(1)


if __name__ == "__main__":
    main()