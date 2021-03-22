################################################################################

# This is the rnn file for the Speech-to-Speech synthesis project, for the
# Neural Network training partition of the project.
# Author: Raphael Nash
# Date created: 22/12/2020
# Date updated: 04/02/2021

################################################################################

#                              # IMPORTS #                                     #
import math
from nmatrices import *
from activations import *

################################################################################

#                              # CLASSES #                                     #

class RNN:
    def __init__(self, size_input, size_hidden, size_output, size_timesteps):
        # Initialise layers
        self.input = make_vector(size_input, False)
        self.hidden = make_vector(size_hidden, False)
        self.output = make_vector(size_output, False)

        # Initialise timesteps/memory
        self.timestep = 0
        self.timesteps = size_timesteps
        self.memory = [0 for i in range(size_timesteps)]

        # Initialise weights and biases
        size_layers = [size_input, size_hidden, size_output]
        self.weights = []
        self.biases = []
        for i in range(len(size_layers) - 1):
            w = Matrix(size_layers[i], size_layers[i + 1], True)
            self.weights.append(w)

            b = make_vector(1, size_layers[i + 1], True)
            self.biases.append(b)

        # Initialise hidden weights
        self.hidden_weights = make_identity(size_hidden)

    def calc_change(n, u, y, yhat, v):
        return (2/n) * (y - yhat) * (ReLU_prime(u)) * (v)

    def backprop(self, inputs, observations, expectations):
        weight_changes = []
        bias_changes = []
        for t in range(len(inputs)):
            input = inputs[t]
            output = observations[t]
            expected = expectations[t]
            hidden_state = self.memory[t]

            # First, perform backprop on each timeframe
            for w in self.weights:
                weight_changes.append(Matrix(w.width, w.height, False))
            for b in self.biases:
                bias_changes.append(Matrix(b.width, b.height, False))

            # Start iterating through each layer at timestep t
            layers = [input, hidden_state, output]
            for k in range(len(layers) - 1):
                i = (len(layers) - 1) - k
                weights = self.weights[i]
                biases = self.biases[i]
                current = layers[i]
                next = layers[i - 1]
                next_changes = make_vector(next.height, False)
                y_preact = matrix_multiply(weights, next)
                for r in range(weights.height):
                    y = current.getItem(r, 0)
                    expected_y = expected.getItem(r, 0)
                    u = y_preact.getItem(r, 0)
                    current_db = bias_changes[i].getItem(r, 0)

                    db = current_db - calc_change(current.height, u, y, expected_y, 1)
                    bias_changes[i].setItem(r, 0, db)

                    for c in range(weights.width):
                        w = weights.getItem(r, c)
                        x = next.getItem(c, 0)

                        current_dw = weights_changes[i].getItem(r, c)
                        dw = current_dw - calc_change(current.height, u, y, expected_y, x)
                        weights_changes[i].setItem(r, c, dw)

                        current_dx = next_changes.getItem(c, 0)
                        dx = current_dx - calc_change(current.height, u, y, expected_y, w)
                        next_changes.setItem(c, 0, dx)

                expected = matrix_add(next, next_changes)

        # Start iterating through timesteps

        # Start with final output; then loop through the timesteps
        current = observations[-1]
        expected = expectations[-1]
        weights = matrix_copy(self.weights[-1])

        hidden_weight_changes = Matrix(self.hidden_weights.width, self.hidden_weights.height, False)
        next = self.memory[-1]
        next_changes = make_vector(next.height, False)
        for i in range(weights.height):
            y = current.getItem(i, 0)
            expected_y = current.getItem(i, 0)
            for j in range(weights.width):
                w = weights.getItem(i, j)
                x = next.getItem(j, 0)
                current_dx = next_changes.getItem(j, 0)
                dx = current_dx - calc_change(current.height, u, y, expected_y, w)
                next_changes.setItem(j, 0, dx)

        expected = matrix_add(next, next_changes)

        # Now start iterating through each of the timesteps
        for i in range(self.timesteps - 1):
            t = self.timesteps - 1 - i

            current = self.memory[t]
            next = self.memory[t - 1]
            next_changes = make_vector(next.height, False)
            for r in range(current.height):
                y = current.getItem(r, 0)
                expected = expected.getItem(r, 0)
                for c in range(next.height):
                    x = next.getItem(c, 0)
                    w = self.hidden_weights.getItem(r, c)
                    current_dx = next_changes.getItem(c, 0)
                    dx = current_dx - calc_change(current.height, u, y, expected_y, w)
                    next_changes.setItem(c, 0, dx)

                    current_dw = hidden_weight_changes.getItem(r, c)
                    dw = current_dw - calc_change(current.height, u, y, expected_y, x)
                    hidden_weight_changes.setItem(r, c, dw)

            expected = matrix_add(next, next_changes)

        # Apply changes to each of the weights/biases

        for i in range(len(self.biases)):
            self.biases[i] = matrix_add(self.biases[i], bias_changes[i])
        for i in range(len(self.weights)):
            self.weights[i] = matrix_add(self.weights[i], self.weight_changes[i])

        self.hidden_weights = matrix_add(self.hidden_weights, hidden_weight_changes)

    def set_input(self, input):
        self.input = copy_matrix(input)

    def forward_pass(self, input):
        all_outputs = [0 for i in range(self.timesteps)]
        t = self.timestep;
        # Initialise a small array populated by references to each layer

        self.set_input(input)
        layers = [self.input, self.hidden, self.output]

        # From input and previous hidden state to current hidden
        sum_input = matrix_multiply(self.weights[0], self.input)
        sum_hidden = matrix_multiply(self.hidden_weights, self.hidden)
        preact_hidden = matrix_add(sum_input, sum_hidden)
        self.hidden = vector_activation(preact_hidden, self.activation)

        # From hidden to output
        preact_output = matrix_multiply(self.weights[1], self.hidden)
        self.output = vector_operation(preact_output, self.activation)

        # Update memory
        if (t == len(self.memory) - 1):
            self.memory.pop(0)
            self.memory.append(matrix_copy(self.hidden))
        else:
            self.memory[t] = matrix_copy(self.hidden)

        return matrix_copy(self.output)

    def train(self, sample):
        inputs = sample[0]
        expectations = sample[1]
        observations = []
        for t in range(self.timesteps):
            observed = self.forward_pass(inputs[t])
            observations.append(observed)
        # Perform backprop
        self.backprop(self, inputs, observations, expectations)

    def train_dataset(self, dataset):
        for sample in dataset:
            train(sample)

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
        observed = self.forward_pass(inputs[t])
        observations.append(observed)
        # Perform backprop
        self.backprop(self, inputs, observations, expectations)

    def train_dataset(self, dataset):
        for sample in dataset:
            train(sample)
