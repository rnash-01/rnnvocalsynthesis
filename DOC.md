# Documentation for project
## rnn.py
### `class RNN()`

#### `RNN.calc_change(self, n, observed, expected, u, v)`

Calculates the required change in a weight/perceptron/bias between one layer
and the one before it.

#### `RNN.forward_pass(self)`

Uses current configuration of the input vector to perform a forward pass through
one timestep. The state of each layer is preserved until the next forward pass,
or until another method resets their values.

#### `RNN.set_input(self, input)`

Assigns a new value to the input vector of the RNN.

Parameters:
- `input` - an instance of the Matrix class, acting as a vector - this is the
vector that will be assigned to the input

#### `RNN.train_dataset(self, dataset)`
Takes in training data, splits it into groups of **t** timeframes,
and calls `RNN.train` on each sample (*each sample represents the* ***t*** *timeframes*)

Parameters:
- `dataset` - a 2-dimensional array, consisting of:
  1. Every recorded group of frequencies throughout the audio recording
  2. the input vector (your voice), and the expected output vector (their voice)

Return value: `void`

#### `RNN.train(self, sample)`

Changes weights based on one sample. A sample is a 2-dimensional array, consisting of:
1. Dimension 1: the collected frequencies at each timestep in the sample
2. Dimension 2: the input vector, and the expected output vector

Parameters
i.e. [[input_1, expected_1], [input_2, expected_2], ... , [input_t, expected_t]]
