import numpy as np
import matplotlib.pyplot as plt
import random
import csv

# Network class
class Network:

    def __init__(self, alpha, bias, filename, layer_sizes, epochs):
        self.input, self.expected = self.read(filename)
        self.alpha = alpha
        self.bias = bias
        self.layer_sizes = layer_sizes
        self.epochs = epochs
        self.y_outputs = []
        self.x_outputs = []

        self.layers = [0] * (len(layer_sizes) + 1)
        for layer_num in range(len(self.layers)):
            if layer_num < 1:
                self.layers[layer_num] = self.input[0]
            else:
                temp = [0] * layer_sizes[layer_num-1]
                self.layers[layer_num] = temp
        self.weights = self.init_weights(len(self.input[0]), layer_sizes)

        for i in range(self.epochs):
            for index in range((len(self.input))-1):
                self.layers[0] = self.input[index]
                self.run_through_layers()
                self.backpropagation(index)
            self.accuracy()
        self.plot()


    # Initiates the weights
    def init_weights(self, input_size, layer_sizes):
        weights = []

        neuron_weights = 2 * np.random.random((input_size*layer_sizes[0])) -1
        weights.append(neuron_weights)

        for layer in range(len(layer_sizes)-1):
            temp = np.random.random(layer_sizes[layer]*layer_sizes[layer+1])
            weights.append(temp)

        return weights
    # ----------------------------------------------------------

    # Used to find the accuracy
    def accuracy(self):
        my_randoms = []
        mse = 0
        for i in range(100):
            my_randoms.append(random.randrange(0, len(self.input), 1))
        for element in my_randoms:
            out = self.layers[len(self.layers) - 1]
            self.y_outputs.append(out[0])
            self.x_outputs.append(element)
            mse = mse + np.mean((out[0] - self.expected[element]) ** 2)
            self.run_through_layers()
        print((mse/100))
    # ----------------------------------------------------------


    def plot(self):
        plt.axis([-1, 1, -2, 2])
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.plot(self.input, self.expected)
        plt.plot(self.x_outputs, self.y_outputs)
        plt.show()


    # Transfer function
    def sigmoid(self, x):
        output = 1 / 1*(1 + np.exp(-x))
        return output


    def relu(self, x):
        return np.maximum(0, x)
    # ----------------------------------------------------------


    # derivative for backpropagation
    def sigmoid_derivative(self, output):
        return output * (1 - output)


    def relu_derivative(self, output):
        if output > 0:
            return 1
        else:
            return 0
    # ----------------------------------------------------------


    #Method for running through the layers calculating the new output
    def run_through_layers(self):
        for layer_num in range(len(self.layers)-1):
            for output_neuron_num in range(self.layer_sizes[layer_num]):
                sum = 0
                for neuron_num in range(len(self.layers[layer_num])):
                    sum = sum + (self.layers[layer_num][neuron_num] * self.weights[layer_num][output_neuron_num])
                self.layers[layer_num+1][output_neuron_num] = self.relu((sum+ self.bias))
    # ----------------------------------------------------------


    # Backpropagation changes the weights depending on the learning rate and how far off the result was
    def backpropagation(self, index):
        deltas = []
        for layer_index in reversed(range(len(self.layers))):
            if layer_index == (len(self.layers) - 1):
                delta_layer = []
                for neuron_index in range(len(self.layers[layer_index])):
                    output = self.layers[layer_index][neuron_index]
                    delta_layer.append((self.expected[index] - output) * self.relu_derivative(output))
                output_delta = delta_layer
            else:
                slices = self.layer_sizes[layer_index]
                layer_weights = np.split(self.weights[layer_index ], slices)
                output_delta = np.asarray([deltas[-1]]).dot(np.asarray(layer_weights))
            deltas.extend(output_delta)

        num_layers = len(self.layers)
        reversed_deltas = list(reversed(deltas))
        for i in reversed(range(num_layers-1)):
            weights_to_update = self.weights[i]
            for neuron in range(self.layer_sizes[i]):
                for weight in range(len(weights_to_update)//self.layer_sizes[i]):
                    self.weights[i][weight+(neuron*self.layer_sizes[i-1])] = self.alpha * reversed_deltas[i+1][neuron]
    # ----------------------------------------------------------


    # Read the data
    def read(self, filename):
        input = []
        expected = []
        with open(filename, 'r') as file:
            reader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                input.append(row[0:(len(row)-1)])
                expected.append([row[(len(row)-1)]])
        return input, expected
    # ----------------------------------------------------------


# Init network and run
network = Network(0.2, 0, 'a.csv', [100,  1], 1000)
# ----------------------------------------------------------