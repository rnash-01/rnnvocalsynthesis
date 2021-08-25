import numpy as np
class LSTM():

    def __init__(self, input_size, output_size):
        # Properties
        self.input_size = input_size
        self.output_size = output_size

        # Layers
        self.input = np.zeros((input_size, 1))
        self.state = np.zeros((output_size, 1))
        self.input_and_state = np.concatenate((self.input, self.state))

        # Weights
        self.weights_forget = np.random.randn(output_size, input_size + output_size)
        self.weights_remember = np.random.randn(output_size, input_size + output_size)
        self.weights_select = np.random.randn(output_size, input_size + output_size)

        # Biases
        self.biases_forget = np.zeros((output_size, 1))
        self.biases_remember = np.zeros((output_size, 1))
        self.biases_select = np.zeros((output_size, 1))

        # Additional state information

    # Training functions

    def train(self):
        # Assume we have a array of input vectors (or an input matrix)
        # Start by splitting the matrix t ways, where t is the number of
        # inputs, or in effect, the size of the time interval, that we perform
        # one forward pass through.

        # After splitting 't' ways, we ensure that the final time interval
        # is also of size 't' - it may be shorter as it is the final set of
        # frequencies in the batch. So pad with zeros if necessary

        # Then we'll start creating matrices with the first vector of each
        # time interval, then the second, etc. This way, we keep the process
        # as vectorized as possible, and training is sped up.

        # We'll end up with a set of outputs for each time step, each output
        # corresponding to one of the forward passes at that time interval.
        # From this, we create a loss vector for each timestep. The loss vector
        # (of shape (1, n/t)). n is the size of the whole training batch, or
        # more specific to the project, the number of frequency samples we've
        # made over the audio
