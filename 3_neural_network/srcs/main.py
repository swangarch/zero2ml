from utils.activation_func import relu, sigmoid
from utils.data_generation import generate_data_1d, generate_data_3d
from utils.nnClass import NN


def main():
    try:
        net_shape = (1, 64, 64, 1)
        activation_funcs = (relu, relu, None)

        nn = NN(net_shape, activation_funcs)

        inputs, truths = generate_data_1d(142, 200)
        test_inputs, test_truths = generate_data_1d(123, 80)


        nn.train(inputs, truths, 20000, 0.01)
        nn.test(inputs, truths, test_inputs, test_truths)
        nn.show_loss()

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()