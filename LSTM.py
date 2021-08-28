import numpy as np
import matplotlib.pyplot as plt
class LSTM():

    def __init__(self, input_size, output_size):
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
        # Weights

        self.parameters["weights_forget"] = np.random.randn(output_size, input_size + output_size)
        self.parameters["weights_in_gate"] = np.random.randn(output_size, input_size + output_size)
        self.parameters["weights_remember"] = np.random.randn(output_size, input_size + output_size)
        self.parameters["weights_select"] = np.random.randn(output_size, input_size + output_size)

        # Biases
        self.parameters["biases_forget"] = np.zeros((output_size, 1))
        self.parameters["biases_in_gate"] = np.zeros((output_size, 1))
        self.parameters["biases_remember"] = np.zeros((output_size, 1))
        self.parameters["biases_select"] = np.zeros((output_size, 1))


        # Additional state information
        self.pre_forget = np.zeros((output_size, 1)) # before sigmoid function
        self.forget = np.zeros((output_size, 1)) # after sigmoid function

        self.pre_in_gate = np.zeros((output_size, 1)) # before sigmoid
        self.in_gate = np.zeros((output_size, 1)) # after sigmoid

        self.pre_remember = np.zeros((output_size, 1)) # before sigmoid
        self.remember = np.zeros((output_size, 1)) # after sigmoid

        self.pre_select = np.zeros((output_size, 1)) # before sigmoid
        self.select = np.zeros((output_size, 1)) # after sigmoid

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
        pre_forget = []
        forget = []
        pre_in_gate = []
        in_gate = []
        pre_remember = []
        remember = []
        pre_select = []
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

            # Append to cache arrays
            input_and_output.append(self.input_and_output)
            pre_forget.append(self.pre_forget)
            forget.append(self.forget)
            pre_in_gate.append(self.pre_in_gate)
            in_gate.append(self.in_gate)
            pre_remember.append(self.pre_remember)
            remember.append(self.remember)
            pre_select.append(self.pre_select)
            select.append(self.select)
            state_multiplied.append(self.state_multiplied)
            state_added.append(self.state_added)
            state_tanh.append(self.state_tanh)
            predictions.append(self.output)

        cache["input_and_output"] = input_and_output
        cache["pre_forget"] = pre_forget
        cache["forget"] = forget
        cache["pre_in_gate"] = pre_in_gate
        cache["in_gate"] = in_gate
        cache["pre_remember"] = pre_remember
        cache["remember"] = remember
        cache["pre_select"] = pre_select
        cache["select"] = select
        cache["state_multiplied"] = state_multiplied
        cache["state_added"] = state_added
        cache["state_tanh"] = state_tanh
        cache["predictions"] = predictions

        return cache

    def calculate_gradients(self, Y, t, cache):
        # Get all cache elements
        input_and_output = cache["input_and_output"]
        pre_forget = cache["pre_forget"]
        forget = cache["forget"]
        pre_in_gate = cache["pre_in_gate"]
        in_gate = cache["in_gate"]
        pre_remember = cache["pre_remember"]
        remember = cache["remember"]
        pre_select = cache["pre_select"]
        select = cache["select"]
        state_multiplied = cache["state_multiplied"]
        state_added = cache["state_added"]
        state_tanh = cache["state_tanh"]
        predictions = cache["predictions"]


        # Get parameters
        weights_forget = self.parameters["weights_forget"]
        weights_in_gate = self.parameters["weights_in_gate"]
        weights_remember = self.parameters["weights_remember"]
        weights_select = self.parameters["weights_select"]

        biases_forget = self.parameters["biases_forget"]
        biases_in_gate = self.parameters["biases_in_gate"]
        biases_remember = self.parameters["biases_remember"]
        biases_select = self.parameters["biases_select"]

        # Set up gradient matrices
        dWf = np.zeros(weights_forget.shape)
        dWi = np.zeros(weights_in_gate.shape)
        dWr = np.zeros(weights_remember.shape)
        dWs = np.zeros(weights_remember.shape)

        dbf = np.zeros(biases_forget.shape)
        dbi = np.zeros(biases_in_gate.shape)
        dbr = np.zeros(biases_remember.shape)
        dbs = np.zeros(biases_select.shape)

        for i in range(t):
            time_step = t - 1 - i

            # Get important states
            prediction_t = predictions[time_step]
            pre_select_t = pre_select[time_step]
            pre_remember_t = pre_remember[time_step]
            pre_in_gate_t = pre_in_gate[time_step]
            pre_forget_t = pre_forget[time_step]
            forget_t = forget[time_step]
            input_and_output_t = input_and_output[time_step]
            state_current = state_added[time_step]

            if(time_step == 0):
                state_previous = np.zeros((self.output_size, true_t.shape[1]))
            else:
                state_previous = state_added[time_step - 1]

            # Calculate derivatives
            # First, sort out dh_t
            true_t = Y[time_step]
            dh_t = -2/(t * prediction_t.shape[0]) * (true_t - prediction_t)

            # Initialise dc_t if not already done
            if(time_step == t - 1):
                dc_t = dh_t * self.sigmoid(pre_select_t)

            # Calculate remember gate gradient
            dr = dc_t
            dp_r = dr * self.sigmoid(pre_select_t) * self.tanh_prime(pre_remember_t)
            dWr = dWr + np.dot(dp_r, input_and_output_t.T)
            dbr = dbr + np.sum(dp_r, axis=1, keepdims=True)

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
            #dh_t = dp_s * 

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
            for b in range(data.shape[1]//t):
                new_vec = data[:, [b * t + time_step]]
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
            print("Iteration {0} complete".format(i + 1))


        iteration_axis = np.arange(stop=iterations)
        plt.plot(iteration_axis, costs)
        plt.savefig("latest_cost.png")


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

test = LSTM(1, 1)
X = np.array([[1],[2],[3],[4]]).T
print(X)
Y = X + 1
print(Y)

test.train(X, Y, 2, 0.01, 10000)
print(test.predict(X))
