import numpy as np

class NeuralNet:

    y = np.array([[0,1,1,0]]).T
    alpha,hidden = (0.5,4)


    def __init__(self,hidden = 1,alpha = 0.05,input=None,train_labels = None):
        self.alpha = alpha
        self.input = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
        self.train_labels = np.array([[0,1,1,0]]).T
        self.syn_0 = 2 * np.random.random((3, self.hidden)) - 1
        self.syn_1 = 2 * np.random.random((self.hidden, 1)) - 1


    def sigmoid(self,x):

        output = 1/1(1+np.exp(-x))
        return output

    def sig_deriv(self,output):
        return output*(1-output)

    def train(self):
        for j in range(60000):
            self.run_through_layers()
            self.backpropagate()

    def backpropagate(self):
        l2_error = self.l2-self.y
        l2_delta = l2_error*self.sig_deriv(self.l2)
        l1_error = self.l2_delta.dot(self.syn_1)
        l1_delta = l2_delta.dot(self.syn_1.T) * (self.l1 * (1 - self.l1))
        self.syn_1 -= (self.alpha * self.l1.T.dot(l2_delta))
        self.syn_0 -= (self.alpha * self.input.T.dot(l1_delta))


    def run_through_layers(self):
        print(self.input,self.syn_0)
        self.l1 = self.sigmoid(np.dot(self.input, self.syn_0))
        self.l2 = self.sigmoid(np.dot(self.l1, self.syn_1))

    def run(self):
        self.train()
        print(self.l2)


nn = NeuralNet()
nn.run()
