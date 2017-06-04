import numpy as np
import csv

# Network class
class Network:

    def __init__(self, alpha, bias, filename, layer_sizes):
        self.input, self.expected = self.read(filename)
        self.alpha = alpha
        self.bias = bias
        self.layer_sizes = layer_sizes

        self.layers = [0] * (len(layer_sizes) + 1)
        for layer_num in range(len(self.layers)):
            if layer_num < 1:
                self.layers[layer_num] = self.input[0]
            else:
                temp = [0] * layer_sizes[layer_num-1]
                self.layers[layer_num] = temp
        self.weights = self.init_weights(len(self.input[0]), layer_sizes)

        for index in range((len(self.input))-1):
            self.layers[0] = self.input[index]
            self.run_through_layers()
            self.backpropagation(index)
            print('One input:')
            print(self.layers)

    # Initiates the weights
    def init_weights(self, input_size, layer_sizes):
        weights = []

        neuron_weights = np.random.random((input_size*layer_sizes[0]))
        weights.append(neuron_weights)

        for layer in range(len(layer_sizes)-1):
            temp = np.random.random(layer_sizes[layer]*layer_sizes[layer+1])
            weights.append(temp)

        return weights
    # ----------------------------------------------------------


    # Transfer function and derivative for backpropagation
    def sigmoid(self, x):
        output = 1 / 1*(1 + np.exp(-x))
        return output


    def sigmoid_derivative(self, output):
        return output * (1 - output)
    # ----------------------------------------------------------

    #Method for running through the layers calculating the new output
    def run_through_layers(self):
        for layer_num in range(len(self.layers)-1):
            for output_neuron_num in range(self.layer_sizes[layer_num]):
                sum = 0
                for neuron_num in range(len(self.layers[layer_num])):
                    sum = sum + (self.layers[layer_num][neuron_num] * self.weights[layer_num][output_neuron_num])
                self.layers[layer_num+1][output_neuron_num] = self.sigmoid(sum)
    # ----------------------------------------------------------

    # Backpropagation changes the weights depending on the learning rate and how far off the result was
    def backpropagation(self, index):
        for layer_index in reversed(range(len(self.layers)-1)):
            for neuron_index in range(len(self.layers[layer_index+1])):
                output = self.layers[layer_index+1][neuron_index]
                error = (self.expected[index] - output) * self.sigmoid_derivative(output)
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
network = Network(0.05, 1, 'a.csv', [4, 2, 1])
# ----------------------------------------------------------