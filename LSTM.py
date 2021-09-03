import numpy as np
#import matplotlib.pyplot as plt
class LSTM():

    def __init__(self, input_size, output_size, forget_layers, remember_layers, in_gate_layers, select_layers):
        # Properties
        self.input_size = input_size
        self.output_size = output_size

        # Layers
        self.input = np.zeros((input_size, 1))
        self.state = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))
        self.input_and_output = np.concatenate((self.output, self.input)) # *

        # * input and 'previous' output concatenated together.

        self.parameters = {}
        """
        # Weights
        self.parameters["weights_forget"] = np.random.randn(output_size, input_size + output_size)
        self.parameters["weights_in_gate"] =  np.random.randn(output_size, input_size + output_size)
        self.parameters["weights_remember"] = np.random.randn(output_size, input_size + output_size)
        self.parameters["weights_select"] = np.random.randn(output_size, input_size + output_size)
        # Biases
        self.parameters["biases_forget"] = np.zeros((output_size, 1))
        self.parameters["biases_in_gate"] = np.zeros((output_size, 1))
        self.parameters["biases_remember"] = np.zeros((output_size, 1))
        self.parameters["biases_select"] = np.zeros((output_size, 1))
        """


        # Additional state information
        self.forget = {}
        self.in_gate = {}
        self.select = {}
        self.remember = {}

        # previous_size - the size of the previous layer (first layer being the input)
        previous_size = self.input_and_output.shape[0]
        for i in range(len(forget_layers)):
            size = forget_layers[i]

            # Layers
            self.forget["{0}_Z".format(i)] = np.zeros((size, 1))
            self.forget["{0}_A".format(i)] = np.zeros((size, 1))

            # Parameters for layers
            self.parameters["weights_forget_{0}".format(i)] = np.random.randn(size, previous_size)
            self.parameters["biases_forget_{0}".format(i)] = np.zeros((size, 1))
            previous_size = size

        previous_size = self.input_and_output.shape[0]

        for i in range(len(select_layers)):
            size = select_layers[i]

            # Layers
            self.select["{0}_Z".format(i)] = np.zeros((size, 1))
            self.select["{0}_A".format(i)] = np.zeros((size, 1))

            # Parameters for layers
            self.parameters["weights_select_{0}".format(i)] = np.random.randn(size, previous_size)
            self.parameters["biases_select_{0}".format(i)] = np.zeros((size, 1))
            previous_size = size

        previous_size = self.input_and_output.shape[0]

        for i in range(len(remember_layers)):
            size = remember_layers[i]

            # Layers
            self.remember["{0}_Z".format(i)] = np.zeros((size, 1))
            self.remember["{0}_A".format(i)] = np.zeros((size, 1))

            # Parameters for layers
            self.parameters["weights_remember_{0}".format(i)] = np.random.randn(size, previous_size)
            self.parameters["biases_remember_{0}".format(i)] = np.zeros((size, 1))
            previous_size = size

        previous_size = self.input_and_output.shape[0]
        for i in range(len(in_gate_layers)):
            size = in_gate_layers[i]
            #Layers
            self.in_gate["{0}_Z".format(i)] = np.zeros((size, 1))
            self.in_gate["{0}_A".format(i)] = np.zeros((size, 1))

            # Parameters for layers
            self.parameters["weights_in_gate_{0}".format(i)] = np.random.randn(size, previous_size)
            self.parameters["biases_in_gate_{0}".format(i)] = np.zeros((size, 1))
            previous_size = size


        self.state_multiplied = np.zeros((output_size, 1)) # state * forget
        self.state_added = np.zeros((output_size, 1)) # state + remember
        self.state_tanh = np.zeros((output_size, 1)) #tanh(state)

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def sigmoid_prime(self, input):
        return self.sigmoid(input) * (1 - self.sigmoid(input))

    def tanh_prime(self, input):
        return 1 - np.power((np.tanh(input)), 2)

    def forward_pass(self, X, t):
        # Set up some arrays for the cache
        input_and_output = []
        forget = []
        in_gate = []
        remember = []
        select = []
        state_multiplied = []
        state_added = []
        state_tanh = []
        predictions = []

        cache = {}

        for time_step in range(t):
            x = X[time_step]
            self.input = x
            self.output = np.zeros((self.output_size, x.shape[1]))
            self.input_and_output = np.concatenate((self.output, self.input), axis=0)

            # Forget gate

            current_val = self.input_and_output
            self.forget["input"] = current_val
            for i in range(len(self.forget)//2):
                weights = self.parameters["weights_forget_{0}".format(i)]
                bias = self.parameters["biases_forget_{0}".format(i)]
                self.forget["{0}_Z".format(i)] = np.dot(weights, current_val) + bias
                self.forget["{0}_A".format(i)] = self.sigmoid(self.forget["{0}_Z".format(i)])
                current_val = self.forget["{0}_A".format(i)]

            self.state_multiplied = np.multiply(current_val, self.state)

            # Remember gate
            current_val = self.input_and_output
            self.remember["input"] = current_val
            for i in range(len(self.remember)//2):
                weights = self.parameters["weights_remember_{0}".format(i)]
                bias = self.parameters["biases_remember_{0}".format(i)]

                self.remember["{0}_Z".format(i)] = np.dot(weights, current_val) + bias
                self.remember["{0}_A".format(i)] = self.tanh(self.remember["{0}_Z".format(i)])
                current_val = self.remember["{0}_A".format(i)]

            # In gate
            remember_out = current_val

            current_val = self.input_and_output
            self.in_gate["input"] = current_val
            for i in range(len(self.in_gate)//2):
                weights = self.parameters["weights_in_gate_{0}".format(i)]
                bias = self.parameters["biases_in_gate_{0}".format(i)]

                self.in_gate["{0}_Z".format(i)] = np.dot(weights, current_val) + bias
                self.in_gate["{0}_A".format(i)] = self.sigmoid(self.in_gate["{0}_Z".format(i)])
                current_val = self.in_gate["{0}_A".format(i)]

            in_gate_out = current_val
            self.state_added = np.multiply(in_gate_out, remember_out) + self.state_multiplied

            self.state_tanh = np.tanh(self.state_added)

            # Select
            current_val = self.input_and_output
            self.select["input"] = current_val
            for i in range(len(self.select)//2):
                weights = self.parameters["weights_select_{0}".format(i)]
                bias = self.parameters["biases_select_{0}".format(i)]

                self.select["{0}_Z".format(i)] = np.dot(weights, current_val) + bias
                self.select["{0}_A".format(i)] = self.sigmoid(self.select["{0}_Z".format(i)])
                current_val = self.select["{0}_A".format(i)]


            self.output = current_val * self.state_tanh

            # Append to cache arrays
            input_and_output.append(self.input_and_output)
            forget.append(self.forget)
            in_gate.append(self.in_gate)
            remember.append(self.remember)
            select.append(self.select)
            state_multiplied.append(self.state_multiplied)
            state_added.append(self.state_added)
            state_tanh.append(self.state_tanh)
            predictions.append(self.output)

        cache["input_and_output"] = input_and_output
        cache["forget"] = forget
        cache["in_gate"] = in_gate
        cache["remember"] = remember
        cache["select"] = select
        cache["state_multiplied"] = state_multiplied
        cache["state_added"] = state_added
        cache["state_tanh"] = state_tanh
        cache["predictions"] = predictions

        return cache

    def calculate_gradients(self, Y, t, cache):
        # Get all cache elements
        input_and_output = cache["input_and_output"]
        forget = cache["forget"]
        in_gate = cache["in_gate"]
        remember = cache["remember"]
        select = cache["select"]
        state_multiplied = cache["state_multiplied"]
        state_added = cache["state_added"]
        state_tanh = cache["state_tanh"]
        predictions = cache["predictions"]


        # Set up gradient matrices
        gradients = {}
        for param_key in self.parameters:
            param = self.parameters[param_key]
            gradients[param_key] = np.zeros(param.shape)

        for i in range(t):
            time_step = t - 1 - i

            # Get important states
            prediction_t = predictions[time_step]
            forget_t = forget[time_step]
            select_t = select[time_step]
            remember_t = remember[time_step]
            in_gate_t = in_gate[time_step]
            input_and_output_t = input_and_output[time_step]
            state_current = state_added[time_step]

            if(time_step == 0):
                state_previous = np.zeros((self.output_size, true_t.shape[1]))
            else:
                state_previous = state_added[time_step - 1]

            # Calculate derivatives
            # First, sort out dh_t
            true_t = Y[time_step]

            # Initialise dc_t if not already done
            if(time_step == t - 1):
                dh_t = -2/(t * prediction_t.shape[0]) * (true_t - prediction_t)
                dc_t = dh_t * self.sigmoid(pre_select_t)

            # Calculate remember gate gradient
            dr = dc_t
            in_gate_layer = in_gate_t["A_{0}".format(len(in_gate_t)//2 - 1)]
            remember_layer = remember_t["Z_{0}".format(len(in_gate_t)//2 - 1)]
            dp_r = dr * in_gate_layer * self.tanh_prime(remember_layer)
            current_multiplier = dp_r

            for i in range(len(remember_t)//2):
                l = len(remember_t)//2 - 1 - i # current layer number
                Z = remember_t["Z_{0}".format(l)]
                if(l > 0):
                    A_prev = remember_t["A_{0}".format(l - 1)]
                else:
                    A_prev = input_and_output_t

                W = self.parameters["weights_remember_{0}".format(l)]
                B = self.parameters["biases_remember_{0}".format(l)]
                dW = np.dot(current_multiplier, A_prev.T)
                db = np.sum(current_multiplier, axis=1, keepdims=True)
                gradients["weights_remember_{0}".format(l)] += dW
                gradients["biases_remember_{0}".format(l)] += db

                current_multiplier = np.dot(W.T, Z)


            """
            dWr = dWr + np.dot(dp_r, input_and_output_t.T)
            dbr = dbr + np.sum(dp_r, axis=1, keepdims=True)
            """
            # Calculate input gate gradient
            dp_i = dr * np.tanh(pre_remember_t) * self.sigmoid_prime(pre_in_gate_t)
            dWi = dWi + np.dot(dp_i, input_and_output_t.T)
            dbi = dbi + np.sum(dp_i, axis=1, keepdims=True)

            # Calculate forget gate gradient
            df = dc_t * state_previous
            dp_f = df * self.sigmoid_prime(pre_forget_t)
            dWf = dWf + np.dot(dp_f, input_and_output_t.T)
            dbf = dbf + np.sum(dp_f, axis=1, keepdims=True)

            dp_s = dh_t * np.tanh(state_current) * self.sigmoid_prime(pre_select_t)
            dWs = dWs + np.dot(dp_s, input_and_output_t.T)
            dbs = dbs + np.sum(dp_s, axis=1, keepdims=True)

            # Update dc_t
            dc_t = dc_t * forget_t

            # Update dh_t
            dh_t = np.dot(self.parameters["weights_select"].T, dp_s)
            dh_t = dh_t[0:self.output_size, :]

        # Compile all gradients into one dictionary
        grads = {}

        grads["weights_forget"] = dWf
        grads["weights_in_gate"] = dWi
        grads["weights_remember"] = dWr
        grads["weights_select"] = dWs

        grads["biases_forget"] = dbf
        grads["biases_in_gate"] = dbi
        grads["biases_remember"] = dbr
        grads["biases_select"] = dbs

        return grads

    def optimise_parameters(self, grads, learning_rate):
        for key in self.parameters:
            self.parameters[key] = self.parameters[key] - learning_rate * grads[key]

    def segment_data(self, t, data):
        # Assumes that data is a matrix/2D array

        if (data.shape[1] % t != 0):
            padding = t - (data.shape[1] % t) # how many zeros to add
            data = np.concatenate((data, np.zeros((data.shape[0], padding))), axis=1)

        # Now set up 'mini batches' (inputs for each time step)
        new_data = []

        for time_step in range(t):
            batch = np.empty((data.shape[0], 0))
            for b in range(data.shape[1] - t + 1):
                new_vec = data[:, [time_step + b]]
                batch = np.append(batch, new_vec, axis=1)
            new_data.append(batch)

        return new_data

    def calculate_cost(self, cache, t, Y):
        predictions = cache["predictions"]
        losses = np.zeros((Y[0].shape))
        for time_step in range(t):
            output_t = predictions[time_step]
            true_t = Y[time_step]
            losses = losses + (1/t) * np.power((true_t - output_t), 2)
        cost = 1/(Y[0].shape[1] * Y[0].shape[0]) * np.sum(losses)
        return cost

    def train(self, X, Y, t, learning_rate, iterations):
        # General guide (note to self)

        # Assume we have a array of input vectors (or an input matrix)
        # Start by splitting the matrix t ways, where t is the number of
        # inputs, or in effect, the size of the time interval, that we perform
        # one forward pass through.

        # After splitting 't' ways, we ensure that the final time interval
        # is also of size 't' - it may be shorter as it is the final set of
        # frequencies in the batch. So pad with zeros if necessary.

        # Then we'll start creating matrices with the first vector of each
        # time interval, then the second, etc. This way, we keep the process
        # as vectorized as possible, and training is sped up.

        # We'll end up with a set of outputs for each time step, each output
        # corresponding to one of the forward passes at that time interval.
        # From this, we create a loss vector for each timestep. The loss vector
        # (of shape (1, n/t)). n is the size of the whole training batch, or
        # more specific to the project, the number of frequency samples we've
        # made over the audio.

        # We'll then create a single cost vector from each of the loss vectors.
        # Each element in the cost vector will correspond to one of the forward
        # passes.

        # During backpropagation, the vectorized form of the costs of each
        # forward pass will be used to calculate derivatives, of which the
        # average value will be calculated and used to update the current
        # parameters.
        # We will start with standard gradient descent for our optimisation
        # algorithm, but if it helps, could be useful to use momentum, RMSprop
        # or Adam.

        # Iterate as many times as specified

        # CODE

        # First, set up training data to be in correct format (e.g., exactly
        # divisible by 't')
        if(X.shape[1] != Y.shape[1]):
            print("Training input must have same number of training examples as training output")
            return 1

        new_X = self.segment_data(t, X)
        new_Y = self.segment_data(t, Y)

        # Now perform a forward pass each time through
        costs = []
        for i in range(iterations):
            cache = self.forward_pass(new_X, t)
            cost = self.calculate_cost(cache, t, new_Y)
            costs.append(cost)
            gradients = self.calculate_gradients(new_Y, t, cache)
            self.optimise_parameters(gradients, learning_rate)
            self.output = np.zeros((self.output_size, 1))
            self.state = np.zeros((self.output_size, 1))
            #print("Iteration {0} complete".format(i + 1))


        iteration_axis = np.arange(stop=iterations)
        plt.plot(iteration_axis, costs)
        plt.savefig("latest_cost_currentgrad.png")


    def predict(self, input):
        output = np.empty((input.shape[0], 0))
        for i in range(input.shape[1]):
            x = input[:,[i]]
            self.input = x
            self.output = np.zeros((self.output_size, x.shape[1]))
            self.input_and_output = np.concatenate((self.output, self.input), axis=0)

            # Forget gate
            self.pre_forget = np.dot(self.parameters["weights_forget"], self.input_and_output) + self.parameters["biases_forget"]
            self.forget = self.sigmoid(self.pre_forget)

            self.state_multiplied = np.multiply(self.forget, self.state)

            # Remember
            self.pre_in_gate = np.dot(self.parameters["weights_in_gate"], self.input_and_output) + self.parameters["biases_in_gate"]
            self.in_gate = self.sigmoid(self.pre_in_gate)
            self.pre_remember = np.dot(self.parameters["weights_remember"], self.input_and_output) + self.parameters["biases_remember"]
            self.remember = np.tanh(self.pre_remember)

            self.state_added = self.state_multiplied + self.in_gate

            # Select
            self.state_tanh = np.tanh(self.state_added)
            self.pre_select = np.dot(self.parameters["weights_select"], self.input_and_output) + self.parameters["biases_select"]
            self.select = self.sigmoid(self.pre_select)

            self.output = self.select * self.state_tanh
            output = np.append(output, self.output, axis=1)
        return output

    def save_parameters(self, file):
        f = open(file, "w")
        for param_key in self.parameters:
            param = self.parameters[param_key]
            for i in range(param.shape[0]):
                row = []
                for j in range(param.shape[1]):
                    row.append(str(param[i,j]))
                f.write(",".join(row) + "\n")
            f.write("_\n")
        f.close()

    def load_parameters(self, file):
        f = open(file, "r")
        for param_key in self.parameters:
            new_param = np.empty((0, 0))
            line = f.readline()
            i = 0
            while (line[0] != '_'):
                row = line.replace("\n","").split(",")
                row = [float(i) for i in row]
                row = np.array(row)
                if(i == 0):
                    new_param = np.empty((0, row.shape[0]))
                new_param = np.append(new_param, [row], axis=0)
                line = f.readline()
                i+=1
            self.parameters[param_key] = new_param

leng = 4
test = LSTM(leng, leng, [5, 5, leng], [5, 5, leng], [5, 5, leng], [5, 5, leng])
X = np.random.randint(0, 10, (leng, 50))
X = test.segment_data(5, X)
print(len(X), X[0].shape[1])
test.forward_pass(X, 5)
"""
X = np.identity(leng)
X_norm = np.linalg.norm(X)
Y = X
print(X)
Y_norm = np.linalg.norm(Y)

test.train(X, Y, 10, 1, 1000)
test.save_parameters("test_params.txt")
Y_test = test.predict(X)
print(Y_test)
"""
