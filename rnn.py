################################################################################

# This is the rnn file for the Speech-to-Speech synthesis project, for the
# Neural Network training partition of the project.
# Author: Raphael Nash
# Date created: 22/12/2020
# Date updated: 25/01/2021

################################################################################

#                              # IMPORTS #                                     #
import math
from nmatrices import *
from activations import *

################################################################################

#                              # CLASSES #                                     #

class RNN:
    def __init__(self):
        # input
        # hidden
        # output
        # weights
        # biases
        # timesteps
        # hidden_weights
        # hidden_biases
        # hidden_states (empty to begin with)
        # hidden_size

        pass

    def calc_change(self, n, observed, expected, u, v):
        # see DOC.md
        if (n != 0):  # avoid division by 0 error
            return (2/n) * (observed - expected) * (activationPrime(u)) * v
        else:
            return 0

    def calc_timesteps(self, t, w, i):
        if (t > 0):
            previous_state = hidden_states[t - 1]
            hw = self.hidden_weights

            current_state_pre_activation = matrix_multiply(hw, previous_state)
            u = current_state_pre_activation.getItem(i, 0)

            startval = activation_hidden_prime(u) * w
            startval *= calc_timesteps(t - 1, w, i)
            return startval
        else:
            return hidden_states[0].getItem(i, 0)


    def calc_change_time(self, expected, observed, t, i, w):
        # *t* - current timestep
        # *observed* - the observed output for element k of the output at
        # timestep t
        # *expected* - the expected output for element k of the output at
        # timestep t
        # *i* - index of the hidden state being observed
        if (n != 0):
            return (2/n) * (expected - observed) * calc_timesteps(t, w, i)
        else:
            return 0


    def backproptime(self, inputs, expectations, observations):
        # void

        # Create duplicate matrices containing the changes needed
        weights_changes = []
        biases_changes = []
        for i in range(len(self.weights)):
            currentweights = self.weights[i]
            width = currentweights.width
            height = currentweights.height
            newweights = Matrix(width, height, False)
            weights_changes.append(newweights)

        for i in range(len(biases)):
            currentbiases = self.biases[i]
            size = currentbiases.height
            newbiases = Matrix(1, size, False)
            biases_changes.append(newbiases)

        # First, adjust with respect to input and weights
        for t in range(self.timesteps):
            observed = observations[t]
            expected = expectations[t]
            layers = [observed, self.hidden_states[t], inputs[t]]
            newexpected = expected

            # Start backpropagation:
            for i in range(len(layers) - 1): # Iterate over layers
                currentweights = self.weights[len(self.weights) - (i + 1)]
                currentbiases = self.biases[len(self.biases) - (i + 1)]
                currentind = len(layers) - (i + 1)
                current_layer = layers[currentind]
                next_layer = copy_matrix(layers[currentind - 1])

                n = current_layer.height  # For use in gradient calculation
                current_state_pre_activation = matrix_multiply(currentweights, next_layer)

                for j in range(n):  # Iterate through output layer
                    observed_j = current_layer.getItem(j, 0)
                    expected_j = newexpected.getItem(j, 0)
                    bias = currentbiases.getItem(j, 0)
                    bias -= calc_change(n, observed_j, expected_j, u, 1)
                    biases_changes[i].setItem(j, 0, bias)
                    u = current_state_pre_activation.getItem(j, 0)
                    for k in range(next_layer.height):  # Iterate through input layer
                        weight = currentweights.getItem(j, k)
                        perceptron = next_layer.getItem(k, 0)

                        weight -=  calc_change(n, observed_j, expected_j, u, perceptron)
                        perceptron -= calc_change(n, observed_j, expected_j, u, weight)

                        weights_changes[i].setItem(j, k, weight)
                        next_layer.setItem(k, 0, perceptron)

                newexpected = matrix_copy(next_layer)
        for w in range(len(self.weights)):
            changes = self.weights_changes[w]
            weightset = matrix_add(self.weights[w], changes)
            self.weights[w] = weightset
        for b in range(len(self.biases)):
            changes = self.biases_changes[b]
            biasset = matrix_add(self.biases[b], changes)
            self.biases[b] = biasset

        # Second, adjust with respect to hidden state at each time step.
        pass

    def activation(self, value):
        return ReLU(value)

    def activation_prime(self, value):
        return ReLU_prime(value)

    def forward_pass(self, inputs, expectations):
        # This assumes one hidden-layer
        observations = []
        for t in range(timesteps):
            self.set_input(inputs[t])
            input2hidden = matrix_multiply(self.weights[0], self.input)
            hidden2hidden = matrix_multiply(self.hidden_weights, self.hidden)
            # self.hidden will either represent the last hidden state, or
            # it will be a zero vector.

            for i in range(self.hidden.height):
                input2hidden_i = input2hidden.getItem(i, 0)
                hidden2hidden_i = hidden2hidden.getItem(i, 0)
                value = activation(input2hidden_i + hidden2hidden_i)
                self.hidden.setItem(i, 0, value)

            hidden2output = matrix_multiply(self.weights[1], self.hidden)
            for i in range(self.output.height):
                self.output = activation(self.hidden2output.getItem(i, 0))
            observations.append(copy_matrix(self.output))
            this.hidden_states.append(copy_matrix(self.hidden))
        return observations

    def set_input(self, input):
        if (input.height == self.input.height):
            for i in range(input.height):
                val = input.getItem(i, 0)
                self.input.setItem(i, 0, val)
            return 0
        else:
            return 1

    def train(self, sample):
        inputs = [timestep[0] for timestep in sample]
        expectations = [timestep[1] for timestep in sample]
        observations = self.forward_pass(inputs, expectations)

        # Now we have a list of t inputs, t expectations, and t observations.
        # We can pass these into our backproptime method, and it'll
        # take it from there.

        self.backproptime(inputs, expectations, outputs)

    def train_dataset(self, dataset):
        # Split dataset into groups of t timesteps.
        t = self.timesteps
        newdataset = []
        i = 0
        while i < range(math.floor(len(dataset) / t)):
            newdataset.append(dataset[i * t:(i+1) * t])
        for datum in newdataset:
            self.train(datum)


class LSTM(RNN):
    def __init__(self):
        # forget_net
        # remember_net
        pass

    def activation_forget(self):
        # float
        pass

    def activation_remember(self):
        # float
        pass

    def __backproptime(self):
        # void
        pass

    def train(self):
        # void
        pass
