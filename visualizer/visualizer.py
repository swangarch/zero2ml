import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import sys

# data = pd.read_csv("./data.csv")
# print(list(data['km']))
# print(list(data['price']))

def predict(mileage, p0, p1):
    return p0 + mileage * p1

def draw_lr(process, x, y):
    plt.ion()
    fig, ax = plt.subplots()
    
    for line in process.stdout:
        words = line.split(" ")
        p0 = float(words[0])
        p1 = float(words[1])
        ax.clear()

        ax.set_title("Car price prediction by mileage")
        ax.set_xlabel("Mileage")
        ax.set_ylabel("Price")

        xlr = [0, 1]
        ylr = [predict(0, p0, p1), predict(1, p0, p1)]
        ax.text(0.75, 0.75, f"Mean square error: {float(words[2]):.4f}")
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

    mileages = [
            1.0, 0.53846365, 0.5877495, 0.74910295, 0.70520633, 0.4233099,
            0.66282976, 0.3044712, 0.5601126, 0.28144044, 0.27236173, 0.18498763,
            0.23537892, 0.3436235, 0.20313586, 0.24470638, 0.11670145, 0.3228958,
            0.17526405, 0.19702812, 0.1432559, 0.2100451, 0.0, 0.17913322,
        ]

    prices = [
            0.0, 0.032327585, 0.16163793, 0.1724138, 0.3448276, 0.36637932,
            0.46336207, 0.50431037, 0.50625, 0.54956895, 0.5905172, 0.5905172,
            0.63577586, 0.6788793, 0.6788793, 0.70043105, 0.70043105, 0.7198276,
            0.82758623, 0.8415948, 0.9353448, 0.9353448, 0.9353448, 1.0,
        ]

    try:
        process = subprocess.Popen(
            ["./linear_regression"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except:
        print("Error: failed to open tranning program, make sure you put your linear_regression at the same dir.")
        sys.exit(1)

    try:
        draw_lr(process, mileages, prices)
    except Exception as e:
        print("Error: ", e)
        sys.exit(1)


if __name__ == "__main__":
    main()