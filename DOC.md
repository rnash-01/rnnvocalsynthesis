# Documentation for project
## rnn.py
### `class RNN()`

#### `RNN.__init__(self, size_input, size_hidden, size_output, size_timesteps)`

The constructor of the RNN class. It implements the architecture of an RNN.
The following attributes are initialised:

- `input` (*Matrix*) - this is the input layer of the RNN.

  It is defined as a Matrix, but takes the abstract form of a vector. It is kept as a
  Matrix to be used in Matrix calculations. The same reasoning applies to `hidden`, `output` and to the elements of `biases` and `memory`.
- `hidden` (*Matrix*) - this is the hidden layer of the RNN.

  Defined as a Matrix, but with 1 column to imitate a vector for Matrix calculations
- `output` (*Matrix*)- this is the output layer of the RNN.

  Defined as a Matrix, but with 1 column to imitate a vector for Matrix calculations
- `weights` (*Matrix[]*) - this is an array that comprises the two weight matrices,
  going from input to hidden, and going from hidden to output, respectively.
- `biases` (*Matrix[]*) - this is an array that comprises the two bias vectors, applied
  to hidden and to output respectively.

  Each element is defined as a Matrix, but with 1 column to imitate a vector for Matrix calculations
- `memory` (*Matrix[]*) - this is an array that contains the past *size_timesteps*
  timesteps. This will be reset at each forward pass for training purposes, but
  may be kept and just updated in a normal context.

  Each element is defined as a Matrix, but with 1 column to imitate a vector for Matrix calculations
- `hidden_weights` (*Matrix*) - a single square Matrix of size *size_hidden*.

  This will contain all of the weights that are applied to a hidden layer at timestep *t* to calculate the state of the hidden layer at timestep *t + 1*.

#### `RNN.forward_pass(self, input)`

This method performs a forward pass through the network, across however
many timesteps there are.

- `input` (*Matrix[]*) - the input samples to the network

  The input is formatted as a list of Matrix objects that each represent a timestep
  in the network. The maximum length of this input is the length of the
  `memory` list attribute of the RNN instance.
