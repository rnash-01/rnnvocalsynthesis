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
    def __init__(self, size_input, size_hidden, size_output, size_timesteps):
        # Initialise layers
        self.input = make_vector(size_input, False)
        self.hidden = make_vector(size_hidden, False)
        self.last_hidden = make_vector(size_hidden, False)
        self.output = make_vector(size_output, False)

        # Initialise timesteps/memory
        self.memory = [0 for i in range(size_timesteps)]

        # Initialise weights and biases
        size_layers = [size_input, size_hidden, size_output]
        self.we            ights = []
        self.biases = []
        for i in range(len(size_layers) - 1):
            w = Matrix(size_layers[i], size_layers[i + 1], True)
            self.weights.append(w)

            b = make_vector(1, size_layers[i + 1], True)
            self.biases.append(b)

        # Initialise hidden weights
        self.hidden_weights = make_identity(hidden_size)

    def forward_pass(self, input):
        self.set_input(input)
        all_outputs = [0 for i in range(len(self.memory))]

        # Initialise a small array populated by references to each layer
        layers = [self.input, self.hidden, self.output]
        for t in range(len(self.memory)):
            self.hidden = copy_matrix(layers[1])
            self.output = copy_matrix(layers[2])

            # From input and previous hidden state to current hidden
            sum_input = matrix_multiply(self.weights[0], self.input)
            sum_hidden = matrix_multiply(self.hidden_weights, self.last_hidden)
            preact_hidden = matrix_add(sum_input, sum_hidden)
            self.hidden = vector_activation(preact_hidden, self.activation)

            # From hidden to output
            preact_output = matrix_multiply(self.weights[1], self.hidden)
            self.output = vector_operation(preact_output, self.activation)

            # Record hidden and output vectors. Copy them, do not reference
            # directly (don't use the reference - it will be the same
            # for all places in the list)
            self.memory[t] = matrix_copy(self.hidden)  # Copy items
            all_outputs[t] = matrix_copy(self.output)  # Copy items

        return all_outputs

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
