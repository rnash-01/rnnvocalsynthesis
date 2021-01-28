################################################################################

# This is the rnn file for the Speech-to-Speech synthesis project, for the
# Neural Network training partition of the project.
# Author: Raphael Nash
# Date created: 22/12/2020
# Date updated: 25/01/2021

################################################################################

#                              # IMPORTS #                                     #
import math
import nmatrices

################################################################################

#                              # CLASSES #                                     #

class RNN:
    def __init__(self):
        # input
        # hidden
        # output
        # weights
        # biases
        # hidden_weights
        # hidden_biases
        # hidden_states (empty to begin with)

        pass

    def calc_change(self, n, observed, expected, u, v):
        # *n* - the number of elements in the output
        # *observed* - the observed output for element k of the output at
        # timestep t
        # *expected* - the expected output for element k of the output at
        # timestep t
        # *u* - the sum of all weight multiplications, + the bias.
        # *v* - this will be:
        #       the input value, if differentiating with respect to a weight
        #       the weight, if differentiating with respect to an input value
        #       1, if differentiating with respect to a bias
        if (n != 0):  # avoid division by 0 error
            return (2/n) * (observed - expected) * (activationPrime(u)) * v
        else:
            return 0

    def calc_timesteps(self, t, w, i):
        if (t > 0):
            previous_state = hidden_states[t - 1]
            hw = self.hidden_weights
            current_state_pre_activation = matrix_multiply(hw, previous_state)
            u = current_state_pre_activation[i]

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
        return (2/n) * (expected - observed) * calc_timesteps(t, w, i)


    def backproptime(self):
        # void
        # First, adjust with respect to input and weights

        # Second, adjust with respect to hidden state at each time step.
        pass

    def activation(self, value):
        # float
        pass

    def forward_pass(self, input):
        # float[]
        pass

    def train(self, input):
        # void
        pass

    def train_dataset(self, dataset):
        pass

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
