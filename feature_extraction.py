import numpy as np
import matplotlib.pyplot as plt
class feature_extract():
    def __init__(self, layers):
        self.layers = [np.zeros((size, 1)) for size in layers]
        self.parameters = {}
        for i in range(0, len(self.layers) - 1):
            prev_size = self.layers[i].shape[0]
            next_size = self.layers[i + 1].shape[0]

            self.parameters["weights_{0}".format(i + 1)] = np.random.randn(next_size, prev_size)
            self.parameters["bias_{0}".format(i + 1)] = np.zeros((next_size, 1))

        self.overflow = False

    def overflow_err(self,err,flag):
        self.overflow = True

    def sigmoid(self, input):
        np.seterr(over='call')
        np.seterrcall(self.overflow_err)
        val = 1 / (1 + np.exp(-input))
        if(self.overflow):
            val = np.zeros(input.shape)
            for i in range(input.shape[0]):
                for j in range(input.shape[1]):
                    if (input[i][j] > 3):
                        val[i][j] = 0
                    else:
                        val[i][j] = 1/(1 + np.exp(-input[i][j]))
            self.overflow = False

        return val

    def sigmoid_prime(self, input):
        return self.sigmoid(input) * (1 - self.sigmoid(input))

    def tanh_prime(self, input):
        return 1 - np.power((np.tanh(input)), 2)

    def forward_pass(self, input):
        states = {}
        if(self.layers[0].shape[0] != input.shape[0]):
            return 1

        self.layers[0] = input
        states["A_0"] = input
        for i in range(1, len(self.layers)):
            weights = self.parameters["weights_{0}".format(i)]
            bias = self.parameters["bias_{0}".format(i)]

            Z = np.dot(weights, self.layers[i - 1])
            A = self.sigmoid(Z)
            self.layers[i] = A

            states["Z_{0}".format(i)] = Z
            states["A_{0}".format(i)] = A

        return states

    def backprop(self, cache, Y):
        m = Y.shape[1]
        n = Y.shape[0]
        output = self.layers[-1]
        dA = -2/(m * n) * (Y - output)
        grads = {}
        for param in self.parameters.keys():
            grads[param] = np.zeros(self.parameters[param].shape)

        for i in range(len(self.layers) - 1):
            l = len(self.layers) - 1 - i
            weights = self.parameters["weights_{0}".format(l)]
            A_l = cache["A_{0}".format(l)]
            A_prev = cache["A_{0}".format(l - 1)]
            Z_l = cache["A_{0}".format(l)]

            dZ = dA * self.sigmoid_prime(Z_l)
            dW = np.dot(dZ, A_prev.T)
            db = dZ

            grads["weights_{0}".format(l)] = grads["weights_{0}".format(l)] + dW
            grads["bias_{0}".format(l)] = grads["bias_{0}".format(l)] + db

            dA = np.dot(weights.T, dZ)

        return grads

    def cost(self, cache, X):
        l = len(self.layers) - 1
        output = cache["A_{0}".format(l)]
        m = X.shape[1]
        n = X.shape[0]

        cost = np.sum(np.power((X - output), 2))/(m * n)
        return cost

    def update_parameters(self, grads, lrate):
        for key in self.parameters.keys():
            grad = grads[key]
            self.parameters[key] = self.parameters[key] - (lrate * grad)

    def train(self, X, iterations, lrate, fname):
        costs = []
        x_axis = [i + 1 for i in range(iterations)]
        for i in range(iterations):
            print("Iteration", i + 1)
            cache = self.forward_pass(X)
            cost = self.cost(cache, X)
            costs.append(cost)
            grads = self.backprop(cache, X)
            self.update_parameters(grads, lrate)

        plt.figure("Cost function")
        plt.plot(x_axis, costs)
        plt.savefig(fname)
        plt.show()
