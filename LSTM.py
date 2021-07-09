import numpy as np
import math

class LSTM:
    def __init__(self, input_size, output_size):
        # Basic layers
        self.input = np.zeros(input_size)
        self.hidden = np.zeros(output_size) # come back to this
        self.cstate = np.zeros(output_size)
        self.output = np.zeros(output_size)

        # Weight matrices
        self.weights_forget = np.random.uniform(-1, 1, (output_size, input_size + output_size))
        self.weights_add = np.random.uniform(-1, 1, (output_size, input_size + output_size))
        self.weights_out = np.random.uniform(-1, 1, (output_size, input_size + output_size))
        self.weights_cell = np.random.uniform(-1, 1, (output_size, output_size))

        # Bias vectors
        self.biases_forget = np.random.uniform(-1, 1, output_size)
        self.biases_add = np.random.uniform(-1, 1, output_size)
        self.biases_out = np.random.uniform(-1, 1, output_size)
        self.biases_cell = np.random.uniform(-1, 1, output_size)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward_pass(self, x):
        if (self.input.size != x.size):
            print("Input size must match that which has been specified for the LSTM")
            exit()

        self.input = x # CHECK NUMPY DOCS FOR "COPY" FUNCTION
        in_hid_concat = np.concatenate((self.input, self.hidden)) # CHECK NUMPY DOCS FOR "CONCAT" FUNCTION

        # Forget/add values. Change cstate after.
        forget = self.weights_forget.dot(in_hid_concat) + self.biases_forget
        forget_sigmoid = self.sigmoid(forget)
        add = self.weights_add.dot(in_hid_concat) + self.biases_add
        add_sigmoid = self.sigmoid(add)
        self.cstate = (self.cstate * forget) + (self.cstate + add)

        # Output filter
        cstate_out = self.weights_cell.dot(self.cstate) + self.biases_cell
        cstate_out_tanh = np.tanh(cstate_out)
        input_out = self.weights_out.dot(in_hid_concat) + self.biases_out
        input_out_sigmoid = self.sigmoid(input_out)
        output = cstate_out_tanh * input_out_sigmoid

        # Remember current state of the network
        # (if training, these will be pushed to some array outside of this
        # function)
        self.in_hid_concat = in_hid_concat
        self.forget = forget
        self.forget_sigmoid = forget_sigmoid
        self.add = add
        self.add_sigmoid = add_sigmoid
        self.cstate_out = cstate_out
        self.cstate_out_tanh = cstate_out_tanh
        self.input_out = input_out
        self.input_out_sigmoid = input_out_sigmoid
        self.output = output
