import numpy as np
import math

class LSTM:
    def __init__(self, input_size, output_size):
        # Basic layers
        self.input = np.zeros((input_size, 1))
        self.hidden = np.zeros((output_size, 1)) # come back to this
        self.cstate = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))

        # Weight matrices
        self.weights_forget = np.random.uniform(-1, 1, (output_size, input_size + output_size))
        self.weights_add = np.random.uniform(-1, 1, (output_size, input_size + output_size))
        self.weights_out = np.random.uniform(-1, 1, (output_size, input_size + output_size))
        self.weights_cell = np.random.uniform(-1, 1, (output_size, output_size))

        # Bias vectors
        self.biases_forget = np.random.uniform(-1, 1, (output_size, 1))
        self.biases_add = np.random.uniform(-1, 1, (output_size, 1))
        self.biases_out = np.random.uniform(-1, 1, (output_size, 1))
        self.biases_cell = np.random.uniform(-1, 1, (output_size, 1))

        # input/hidden/cstate/output sizes
        self.input_size = input_size
        self.internals_size = output_size

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh_prime(self, x):
        sech = 1 / (np.cosh(x))
        return np.square(sech)

    def forward_pass(self, x, y):
        if (self.input.size != x.size):
            print("Input size must match that which has been specified for the LSTM")
            exit()

        self.input = x # CHECK NUMPY DOCS FOR "COPY" FUNCTION
        in_hid_concat = np.append(self.input, self.hidden, axis=0) # CHECK NUMPY DOCS FOR "CONCAT" FUNCTION

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

        # Calculate error
        error = np.zeros(self.internals_size)
        # Remember current state of the network
        # (if training, these will be pushed to some array outside of this
        # function)
        self.in_hid_concat = in_hid_concat
        self.forget = forget
        self.add = add
        self.cstate_out = cstate_out
        self.input_out = input_out
        self.output = output
        self.error = error

    def calc_cost(self, Yhat, Y):
        loss = np.power((Y - Yhat), 2)
        sum = np.sum(loss)/m
        return np.squeeze(sum)

    def calc_gradients(self, states, forgets, adds, cstate_outs, filtered_outputs, predictions, X_train, Y_train):
        # Get everything for 'last' cell state and output
        count = outputs.shape[1]
        ht = outputs[-1]
        dhtm1 = (2/m) * (Y_train - predictions) #dh(t - 1) (but from the PREVIOUSLY evaluated cell)
        dht = dhtm1

        # cell state
        dtanhct = dht * self.sigmoid(filtered_outputs[:, -1]) * self.tanh_prime(cstate_outs[:, -1])
        dctm1 = np.dot(self.weights_cell.T, dtanhctm1)

        # Other bits
        concat_len = predictions.shape[0] + X_train.shape[0]
        dft = 1
        dat = 1
        dut = 1

        for i in range(count):
            # current input, hidden, and cell state
            xt = X_H_concat[0:concat_len/2, count - 1 - i]
            ht = outputs[:,count - 1 - i]
            in_hid_concat_t = np.insert(xt, ht, axis=0)
            ct = cstate_outs[:,count - 1 - i]

            # Get gradient with respect to current cellstate
            dct = dctm1
            dctm1 = self.sigmoid_prime(forgets[:, count - 1 - i])

            # Get gradient for current (and then previous) output
            dht = dhtm1
            dut = dht * self.tanh_prime(cstate_outs[count - 1 - i]) * self.sigmoid_prime(filtered_outputs[:,count - 1 - i])
            dhtm1 = np.dot(self.weights_forget.T, du)[[concat_len/2:concat_len, 1]]
            dtanhct = dht * self.sigmoid(filtered_outputs[:, count - 1- i]) * self.tanh_prime(cstate_outs[:, count - 1 - i])

            # Parameter gradients
            # Forget weights and biases
            dft = dct * states[:, count - 2 - i] * sigmoid_prime(forgets[:,count - 1 - i])
            dWft = np.dot(dft, in_hid_concat_t.T)
            dbft = dft

            # Add weights
            dat = dct * sigmoid_prime(adds[:,count - 1 - i])
            dWat = np.dot(dat, in_hid_concat_t.T)
            dbat = dat

            # Pre-output weights
            dWot = np.dot(dut, in_hid_concat_t.T)
            dbot = dut

            # Cell state weights
            dWct = np.dot(dtanhct, ct.T)
            dbct = dtanhct

        grads = {"dWft": dWft, "dWat": dWat, "dWot": dWot, "dWct": dWct, "dbft": dbft, "dbat": dbat, "dbot": dbot, "dbct": dbct}

        return grads

"""
    def train(self, x_data, y_data, time_size):
        i = 0
        while i < math.ceil(len(x_data) / time_size):
            # First, reset the cell state
            self.cstate = np.zeros(self.internals_size)

            # Now set up arrays
            concat_inputs = []
            forgets = []
            adds = []
            cstates = []
            cstate_outs = []
            filtered_outputs = []
            outputs = []
            errors = []
            cost = np.zeros((10, 1))
            costs = np.empty((10, 1), dtype=int)
            costs = np.append(costs, cost, axis=1)
            costs = np.append(costs, np.random.randint(0, 2, (10, 1)), axis=1)
            print(costs)

            # Complete forward pass for n=time_size time steps
            j = 0
            while (j < time_size) and ((i * time_size) + j) < len(x_data):
                x = x_data.T[(i * time_size) + j].T
                print(x)
                y = y_data.T[(i * time_size) + j].T

                self.forward_pass(x, y)
                if(j == 0):
                    concats_inputs = self.in_hid_concat
                    states = self.cstate
                    forgets = self.forget
                    adds = self.add
                    cstate_outs = self.cstate_out
                    filtered_outputs = self.input_out
                    outputs = self.output
                    errprs = self.error
                    cost = np.zeros((10, 1))
                else:
                    concat_inputs = np.append(concat_inputs, self.in_hid_concat, axis=1)
                    states = np.append(cstates, self.cstate, axis=1)
                    forgets = np.append(forgets, self.forget, axis=1)
                    adds = np.append(adds, self.add, axis=1)
                    cstate_outs = np.append(cstate_outs, self.cstate_out, axis=1)
                    filtered_outputs = np.append(filtered_outputs, self.input_out, axis=1)
                    outputs = np.append(outputs, self.output, axis=1)
                    errors = np.append(errors, self.error, axis=1)
                    cost = cost + self.error

                j += 1

            #self.calc_gradients(concat_inputs, states, forgets, adds, cstate_outs, filtered_outputs, outputs, cost)
            i += 1
    """

test = LSTM(10, 10)
x_data = np.random.uniform(-10, 10, (10, 50))
y_data = np.random.randint(0, 2, (10, 50))

test.train(x_data, y_data, 50)
