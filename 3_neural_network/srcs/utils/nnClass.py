import numpy as np
from utils.activation_func import relu, relu_deriv, sigmoid, sigmoid_deriv, activ_deriv
from utils.nnUtils import forward_layer, network, loss, mse_loss, create_bias, gradient_descent, mean_gradients, shuffle_data, split_dataset
import matplotlib.pyplot as plt
from datetime import datetime
import os


class NN:
    """Neural network class, which can perform training and prediction"""

    def __init__(self, shape, activation_functions):
        """Init a multi layer perceptron."""
        
        if len(shape) < 4:
            raise ValueError("Net shape to short.")
        if len(shape) != len(activation_functions) + 1:
            raise ValueError("Mismatched net shape and activation functions.")
        
        self.net_shape = shape  ##
        self.activ_funcs = activation_functions  ##

        self.nets = network(self.net_shape)
        self.len_nets = len(self.nets)
        self.len_out = self.net_shape[-1]
        self.biases = create_bias(self.net_shape, 0.1) 

        self.graph_loss_train = []
        self.graph_loss_test = []
        self.graph_epoch = []
        self.grads_weights = []
        self.grads_biases = []
        self.actives = []

        self.loss_threshold = 0.000000001
        self.loss_test = None
        self.loss_train = None
        self.plt = plt

        self.deriv_map = dict()
        self.deriv_map[relu] = relu_deriv
        self.deriv_map[sigmoid] = sigmoid_deriv


    def check_train_params(self, inputs, truths):
        """Check training parameters."""

        if len(inputs) != len(truths):
            raise ValueError("Mismatched training dataset")


    def train_batch(self, inputs, truths, learning_rate=0.01):
        """"""

        num_data = len(inputs)
        self.grads_weights = []
        self.grads_biases = []
        for i_data, input in enumerate(inputs):
            # -----------------------------forward --------------------------------
            activ = input
            self.actives = [input]
            for i in range(self.len_nets):
                activ = forward_layer(self.nets[i], activ, self.biases[i], self.activ_funcs[i])
                self.actives.append(activ)
            # -----------------------------forward end-----------------------------

            # -----------------------------back probab --------------------------------
            Wgrads = []  #collect the output gradient matrixs
            Bgrads = []
            diff = (self.actives[-1] - truths[i_data] ) * activ_deriv(self.activ_funcs[-1], self.actives[-1], self.deriv_map) # last layer difference
            Bgrads.append(diff) #gradient for bias
            Wgrads.append(diff @ self.actives[-2].T)
            for i in range(self.len_nets - 1, 0, -1): #exclude index == 0
                loss_prev_layer = self.nets[i].T @ diff  #cal the loss of prev layer
                diff = loss_prev_layer * activ_deriv(self.activ_funcs[i - 1], self.actives[i], self.deriv_map)
                Bgrads.append(diff) #gradient for bias
                Wgrads.append(diff @ self.actives[i - 1].T) # add gradient for weights to grads 
            # -----------------------------back probab end-----------------------------
            self.grads_weights.append(Wgrads)
            self.grads_biases.append(Bgrads)

        Wgrads_mean, Bgrads_mean = mean_gradients(self.net_shape, self.grads_weights, self.grads_biases, num_data)
        gradient_descent(self.nets, self.biases, Wgrads_mean, Bgrads_mean, learning_rate)

    
    def inference(self, inputs):
        """After training, use weights to do inference."""

        result = []
        for input in inputs:
            activ = input
            for i in range(self.len_nets):
                activ = forward_layer(self.nets[i], activ, self.biases[i], self.activ_funcs[i])
            result.append(activ)
        return result


    def train(self, inputs, truths, max_iter=10000, learning_rate=0.01, batch_size=20, visualize=True, test_ratio = 0.8, threshold=None, animation=False):
        """Train a dataset."""

        self.check_train_params(inputs, truths)
        self.prepare(visualize, inputs, truths, threshold)

        inputs_train, truths_train, inputs_test, truths_test = split_dataset(inputs, truths, test_ratio)
        startTime = datetime.now()
        for epoch in range(max_iter):
            inputs_shuffled, truths_shuffled = shuffle_data(inputs_train, truths_train)
            # mini_batch_training
            count = 0
            while count < len(inputs_train):
                inputs_batch = inputs_shuffled[count: count + batch_size]
                truths_batch = truths_shuffled[count: count + batch_size]
                self.train_batch(inputs_batch, truths_batch, learning_rate)
                count += batch_size
            # mini_batch_training
            stop = self.show_record(epoch, inputs_train, inputs_test, truths_train, truths_test, startTime, animation)
            if stop == True:
                break
        print()
        print("[TRAINING DONE]")

        self.plt.ioff()
        self.plt.close()


    def test(self, inputs, truths, test_inputs, test_truths):
        """Test for a new dataset."""
        test_result = self.inference(test_inputs)
        plt.scatter(inputs[:, 0], np.array(truths)[:, 0], c="blue", label="Prediction", s=0.5)
        plt.scatter(test_inputs[:, 0], np.array(test_result)[:, 0], c="red", label="Prediction", s=0.5)
        plt.legend(loc="lower left")
        plt.savefig("visualize/prediction.png", dpi=300, bbox_inches='tight')
        plt.close()

    
    def test_animation(self, test_inputs, test_truths):
        """Test for a new dataset."""

        test_result = self.inference(test_inputs)
        self.plt.clf()
        self.plt.scatter(test_inputs[:, 0], np.array(test_truths)[:, 0], c="blue", label="Truth", s=5)
        self.plt.scatter(test_inputs[:, 0], np.array(test_result)[:, 0], c="red", label="Prediction", s=5)
        self.plt.legend(loc="lower left")
        self.plt.pause(0.1)
        self.plt.show()


    def show_record(self, epoch, inputs_train, inputs_test, truths_train, truths_test, startTime, animation): #return a boolean to determine if training continue
        """Show and record the loss"""

        if epoch % 10 == 0:
            predicts_train = self.inference(inputs_train)
            predicts_test = self.inference(inputs_test)
            loss_train = loss(mse_loss, truths_train, predicts_train)
            loss_test = loss(mse_loss, truths_test, predicts_test)

            if loss_train < 1 and loss_test < 1:
                self.graph_loss_train.append(loss_train)
                self.graph_loss_test.append(loss_test)
            self.graph_epoch.append(epoch)

            if animation == True and epoch % 50 == 0:
                self.test_animation(inputs_test[:50], truths_test[:50])
            if epoch % 100 == 0:
                time = str(datetime.now() - startTime).split(".")[0]
                print(f"\033[?25l[EPOCH] {epoch}  [LOSS_TRAIN] {loss_train:8f} [LOSS_TEST] {loss_test:8f}  [TIME] {time}\033[?25l", end="\r")
                

            if self.loss_train  is not None and self.loss_test is not None:
                if abs(self.loss_train - loss_train) < self.loss_threshold and abs(self.loss_test - loss_test) < self.loss_threshold:
                    return True
            
            self.loss_train = loss_train
            self.loss_test = loss_test

            
        return False


    def show_loss(self):
        """Show loss func."""

        plt.plot(self.graph_epoch, self.graph_loss_train, c="cyan", lw=0.5, label="Training loss")
        plt.plot(self.graph_epoch, self.graph_loss_test, c="orange", lw=0.5, label="Test loss")
        plt.legend(loc="upper right")
        plt.savefig("visualize/loss.png", dpi=300, bbox_inches='tight')
        plt.close()


    def prepare(self, visualize, inputs, truths, threshold):
        """Create plt scatter."""

        if visualize == True:
            os.makedirs("visualize", exist_ok=True)
            plt.ion()
        if threshold is not None:
            self.loss_threshold = threshold