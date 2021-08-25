import numpy as np
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

        # Weights
        self.weights_forget = np.random.randn(output_size, input_size + output_size)
        self.weights_in_gate = np.random.randn(output_size, input_size + output_size)
        self.weights_remember = np.random.randn(output_size, input_size + output_size)
        self.weights_select = np.random.randn(output_size, input_size + output_size)

        # Biases
        self.biases_forget = np.zeros((output_size, 1))
        self.biases_in_gate = np.zeros((output_size, 1))
        self.biases_remember = np.zeros((output_size, 1))
        self.biases_select = np.zeros((output_size, 1))

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
            self.pre_forget = np.dot(self.weights_forget, self.input_and_output)
            self.forget = self.sigmoid(self.pre_forget)

        return cache


    def segment_data(self, t, data):
        # Assumes that data is a matrix/2D array

        if (data.shape[1] % t != 0):
            padding = t - (X.shape[1] % t) # how many zeros to add
            data = np.concatenate((data, np.zeros((data.shape[0], padding))), axis=1)

        # Now set up 'mini batches' (inputs for each time step)
        new_data = []
        for b in range(X.shape[1]//t):
            batch = np.empty((data.shape[0], 0))
            for timestep in range(t):
                new_vec = data[:, [b * t + timestep]]
                batch = np.append(batch, new_vec, axis=1)
            new_data.append(batch)
        return new_data

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
        for i in range(iterations):
            cache = self.forward_pass(new_X, t)

        # All states,

test = LSTM(10, 5)
X = np.random.randint(0, 10, (10, 6))
Y = np.random.randint(0, 10, (5, 6))
print(X)
print(Y)
test.train(X, Y, 2, 1, 1)
