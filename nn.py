import numpy as np

class NeuralNetwork:
    def __init__(self, num_layers, num_classes, layer_size, input_dim):
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.layer_size = layer_size
        self.input_dim = input_dim
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def initialize(self):
        #this function initialises the weight matrices. We denote them by theta
        self.theta = []
        self.delta = []
        self.activations = []
        self.error = []
        for i in range(self.num_layers):
            self.activations.append(np.zeros(shape=[self.layer_size[i], 1]))
            self.error.append(np.zeros(shape=[self.layer_size[i], 1]))
        # print("activations, ", self.activations)
        for i in range(1, self.num_layers):
            theta = np.random.random(size=[self.layer_size[i], self.layer_size[i - 1]]) #range of numbers: [2, 7]
            print("theta generated : ", theta)
            self.delta.append(np.zeros_like(theta))
            self.theta.append(theta)
        self.theta = np.array(self.theta)
        self.delta = np.array(self.delta)
        self.activations = np.array(self.activations)
        self.error = np.array(self.error)
        self.D = self.delta.copy()
        print("delta, ", self.delta)
    def forward_prop(self, x):
        #perform one pass of forwards propagation
        self.activations[0] = x
        for i in range(1, self.num_layers):
            self.activations[i] = self.sigmoid(np.matmul(self.theta[i - 1], self.activations[i - 1]))
            # print("z : for i", i, np.matmul(self.theta[i - 1], self.activations[i - 1]))
        # print("after forwardprop, ", self.activations)


    def backprop(self, y):
        self.error[self.num_layers - 1] = self.activations[self.num_layers - 1] - y
        for i in range(self.num_layers - 2, 0, -1):
            self.error[i] = np.multiply(np.matmul(np.transpose(self.theta[i]), self.error[i + 1]), np.multiply(self.activations[i], 1 - self.activations[i]))

        for i in range(self.num_layers - 1):
            self.delta[i] = self.delta[i] + np.matmul(self.error[i + 1], np.transpose(self.activations[i]))
        # print("error ^", i, self.error[i])
    def train(self, X, y, learning_rate, reg_lambda, epochs):
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.epochs = epochs
        for j in range(self.epochs):
            cost = 0
            for i in range(X.shape[0]):
                self.forward_prop(X[i])
                self.backprop(y[i])

                for k in range(self.num_layers - 1):
                    # if k != 0:
                    #we have bias unit only in the hidden layers
                    self.D[k][:, 1:] = (1 / X.shape[0]) * (self.delta[k][:, 1:] + self.reg_lambda * self.theta[k][:, 1:])
                    self.D[k][:, 0] = (1 / X.shape[0]) * (self.delta[k][:, 0])
                    # else:
                    #     self.D[k][:, :] = (1 / X.shape[0]) * (self.delta[k][:, :] + self.reg_lambda * self.theta[k][:, :])

                    #gradient descent step
                    self.theta[k] = self.theta[k] - self.learning_rate * self.D[k]
                    # print(k, self.D[k])
                cost += self.cost(y[i])
            cost /= X.shape[0]
            print("epoch %d cost %f" % (j, cost))
    def predict(self, X):
        self.forward_prop(X)
        return self.activations[-1]
    def cost(self, y):
        return np.sum((self.activations[self.num_layers - 1] - y) ** 2)