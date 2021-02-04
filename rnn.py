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
        self.output = make_vector(size_output, False)

        # Initialise timesteps/memory
        self.timestep = 0
        self.timesteps = size_timesteps
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

    def backprop(self, inputs, observations, expectations):
        for t in range(len(inputs)):
            input = inputs[t]
            output = observations[t]
            expected = expectations[t]
            hidden_state = self.memory[t]

            # First, perform backprop on each timeframe
            layers = [input, hidden_state, output]

        # Then, perform backprop on hidden layers

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
        for t in range(self.timesteps):################################################################################

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
        self.hidden_weights = make_identity(hidden_size)

    def backprop(self, inputs, observations, expectations):
        for t in range(len(inputs)):
            input = inputs[t]
            output = observations[t]
            expected = expectations[t]
            hidden_state = self.memory[t]

            # First, perform backprop on each timeframe
            weight_changes = []
            bias_changes = []
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
                for r in range(weights.height):
                    b = biases.getItem(r, 0)  # Bias
                    current_db = bias_changes[i].getItem(r, 0)
                    db = current_db - calc_change_bias('STUFF GOES HERE')
                    bias_changes[i].setItem(r, 0, db)

                    for c in range(weights.width):
                        w = weights.getItem(r, c)
                        current_dw = weights_changes[i].getItem(r, c)
                        dw = current_dw - calc_change_weight('STUFF GOES HERE')
                        weights_changes[i].setItem(r, c, dw)

                        x = next.getItem(c)
                        current_dx = next_changes.getItem(c, 0)
                        dx = current_dx - calc_change_x('STUFF GOES HERE')
                        next_changes.setItem(c, 0, dx)

                expected = matrix_add(next, next_changes)

            # Start iterating through timesteps

            # Start with final output; then loop through the timesteps
            final = observations[-1]
            expected = expectations[-1]
            weights = matrix_copy(self.weights[-1])
            next = self.memory[-1]

        # Then, perform backprop on hidden layers

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
        pass

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
        pass
