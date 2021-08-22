import numpy as np
import math
import matplotlib.pyplot as plt
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

        # Ensure that cstate is correctly sized
        # Forget/add values. Change cstate after.
        forget = self.weights_forget.dot(in_hid_concat) + self.biases_forget
        forget_sigmoid = self.sigmoid(forget)
        add = self.weights_add.dot(in_hid_concat) + self.biases_add
        add_sigmoid = self.sigmoid(add)
        self.cstate = (self.cstate * forget_sigmoid) + (self.cstate + add_sigmoid)

        # Output filter
        cstate_out = self.weights_cell.dot(self.cstate) + self.biases_cell
        cstate_out_tanh = np.tanh(cstate_out)
        input_out = self.weights_out.dot(in_hid_concat) + self.biases_out
        input_out_sigmoid = self.sigmoid(input_out)

        output = cstate_out_tanh * input_out_sigmoid

        # Calculate error
        error = np.zeros((self.internals_size, 1))
        # Remember current state of the network
        # (if training, these will be pushed to some array outside of this
        # function)
        self.hidden = output
        self.in_hid_concat = in_hid_concat
        self.forget = forget
        self.add = add
        self.cstate_out = cstate_out
        self.input_out = input_out
        self.output = output
        self.error = error

        return output, error

    def calc_cost(self, Yhat, Y):
        loss = np.power((Y - Yhat), 2)
        sum = np.sum(loss)/m
        return np.squeeze(sum)

    def calc_gradients(self, states, forgets, adds, cstate_outs, filtered_outputs, predictions, X_train, Y_train, hiddens):
        # Get everything for 'last' cell state and output
        count = predictions.shape[1]
        ht = predictions[:,-1].reshape(self.internals_size, 1)
        dhtm1 = (2/count) * (Y_train[:,count - 1].reshape(self.internals_size, 1) - ht) #dh(t - 1) (but from the PREVIOUSLY evaluated cell)
        dht = dhtm1

        # cell state
        dtanhct = dht * self.sigmoid(filtered_outputs[:, -1].reshape(self.internals_size, 1)) * self.tanh_prime(cstate_outs[:, -1].reshape(self.internals_size, 1))
        dctm1 = np.dot(self.weights_cell.T, dtanhct)

        # Other bits
        concat_len = predictions.shape[0] + X_train.shape[0]
        X_H_concat = np.append(X_train, hiddens, axis=0)
        dft = 1
        dat = 1
        dut = 1

        for i in range(count):
            # current input, hidden, cell state, transformed input, forget, add/remember
            xt = X_train[:, count - 1 - i].reshape(self.input_size, 1)
            ht = predictions[:,count - 1 - i].reshape(self.internals_size, 1)
            in_hid_concat_t = X_H_concat[:, count - 1 - i].reshape(self.input_size + self.internals_size, 1)

            ct = cstate_outs[:,count - 1 - i].reshape(self.internals_size, 1)
            if(i == count - 1):
                ctm1 = np.zeros((self.internals_size, 1))
            else:
                ctm1 = cstate_outs[:,count - 2 - i].reshape(self.internals_size, 1)

            ut = filtered_outputs[:,count - 1 - i].reshape(self.internals_size, 1)
            ft = forgets[:,count - 1 - i].reshape(self.internals_size, 1)
            at = adds[:,count - 1 - i].reshape(self.internals_size, 1)

            # Get gradient with respect to current cellstate
            dct = dctm1
            dctm1 = self.sigmoid_prime(ft)

            # Get gradient for current (and then previous) output
            dht = dhtm1
            dut = dht * self.tanh_prime(ct) * self.sigmoid_prime(ut)
            dhtm1 = np.dot(self.weights_forget.T, dut)[int(concat_len/2):concat_len, 0].reshape(self.internals_size, 1)
            dtanhct = dht * self.sigmoid(ut) * self.tanh_prime(ct)

            # Parameter gradients
            # Forget weights and biases
            dft = dct * ctm1 * self.sigmoid_prime(ft)
            dWft = np.dot(dft, in_hid_concat_t.T)
            dbft = dft

            # Add weights
            dat = dct * self.sigmoid_prime(at)
            dWat = np.dot(dat, in_hid_concat_t.T)
            dbat = dat

            # Pre-output weights
            dWot = np.dot(dut, in_hid_concat_t.T)
            dbot = dut

            # Cell state weights
            dWct = np.dot(dtanhct, ct.T)
            dbct = dtanhct

        grads = {"dWf": dWft, "dWa": dWat, "dWo": dWot, "dWc": dWct, "dbf": dbft, "dba": dbat, "dbo": dbot, "dbc": dbct}

        return grads

    def optimize_parameters(self, grads, learning_rate):
        # Get weights
        dWf = grads["dWf"]
        dWa = grads["dWa"]
        dWo = grads["dWo"]
        dWc = grads["dWc"]

        # Get biases
        dbf = grads["dbf"]
        dba = grads["dba"]
        dbo = grads["dbo"]
        dbc = grads["dbc"]

        # Update parameters
        self.weights_forget = self.weights_forget - learning_rate * dWf
        self.weights_add = self.weights_add - learning_rate * dWa
        self.weights_out = self.weights_out - learning_rate * dWo
        self.weights_cell = self.weights_cell - learning_rate * dWc

        self.biases_forget = self.biases_forget - learning_rate * dbf
        self.biases_add = self.biases_add - learning_rate * dba
        self.biases_out = self.biases_out - learning_rate * dbo
        self.biases_cell = self.biases_cell - learning_rate * dbc

    def train(self, x_data, y_data, time_size, learning_rate, iterations):
        iteration_axis = []
        cost_axis = []
        for iteration in range(iterations):
            iteration_axis.append(iteration)
            grads = {"dWf": 0, "dWa": 0, "dWo": 0, "dWc": 0, "dbf": 0, "dba": 0, "dbo": 0, "dbc": 0}
            i = 0
            while i < math.floor(len(x_data) / time_size):
                # First, reset the cell state
                self.cstate = np.zeros((self.internals_size, 1))

                # 'Declare' arrays
                inputs = []
                true_outs = []
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

                # Complete forward pass for n=time_size time steps
                j = 0
                while (j < time_size) and ((i * time_size) + j) < x_data.shape[1]:
                    x = x_data[:,(i * time_size) + j].reshape(10, 1)
                    y = y_data[:,(i * time_size) + j].reshape(10, 1)

                    self.forward_pass(x, y)
                    if(j == 0):
                        inputs = x
                        true_outs = y
                        hiddens = np.zeros((self.internals_size, 1))
                        concat_inputs = self.in_hid_concat
                        states = self.cstate
                        forgets = self.forget
                        adds = self.add
                        cstate_outs = self.cstate_out
                        filtered_outputs = self.input_out
                        outputs = self.output
                        errors = self.error
                        cost = np.zeros((10, 1))
                    else:
                        inputs = np.append(inputs, x, axis=1)
                        true_outs = np.append(true_outs, y, axis=1)
                        hiddens = np.append(hiddens, outputs[:, j - 1].reshape(self.internals_size, 1), axis=1)
                        concat_inputs = np.append(concat_inputs, self.in_hid_concat, axis=1)
                        states = np.append(states, self.cstate, axis=1)
                        forgets = np.append(forgets, self.forget, axis=1)
                        adds = np.append(adds, self.add, axis=1)
                        cstate_outs = np.append(cstate_outs, self.cstate_out, axis=1)
                        filtered_outputs = np.append(filtered_outputs, self.input_out, axis=1)
                        outputs = np.append(outputs, self.output, axis=1)
                        errors = np.append(errors, self.error, axis=1)
                        cost = cost + self.error


                    j += 1

                count = x_data.shape[1] / math.ceil(time_size)
                newgrads = self.calc_gradients(states, forgets, adds, cstate_outs, filtered_outputs, outputs, inputs, true_outs, hiddens)

                for key in grads.keys():
                    oldgrad = grads[key]
                    newgrad = newgrads[key]
                    grads[key] = oldgrad + newgrad / count

                i += 1

            cost_axis.append(np.abs(1/j * np.sum(costs, axis=1)))
            self.optimize_parameters(grads, learning_rate)

        plt.plot(iteration_axis, cost_axis)
        plt.xlabel("Iteration Number")
        plt.ylabel("Absolute loss")

        plt.savefig("cost.png")
        plt.show()

    def reset_state(self):
        self.input = np.zeros((input_size, 1))
        self.hidden = np.zeros((output_size, 1))
        self.cstate = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))

    def predict(self, x):
        predictions = 0

        if(x.shape[0] != self.input_size):
            print("Could not predict - size of given input ({0}) does not match network input size ({1})".format(x.shape[0], self.input_size))
            return 0
        else:
            for i in range(x.shape[1]):
                self.input = x[:,i].reshape(self.input_size, 1)
                in_hid_concat = np.append(self.input, self.hidden, axis=0)

                forget = self.sigmoid(self.weights_forget.dot(in_hid_concat))
                add = self.sigmoid(self.weights_add.dot(in_hid_concat))
                filtered_out = self.sigmoid(self.weights_out.dot(in_hid_concat))

                self.cstate = add + (self.cstate * forget)
                cell_selection = np.tanh(self.weights_cell.dot(self.cstate))

                self.output = filtered_out * cell_selection
                if(i == 0):
                    predictions = self.output
                else:
                    predictions = np.append(predictions, self.output, axis=1)

        return predictions
    
