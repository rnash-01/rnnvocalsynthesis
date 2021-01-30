# Documentation for project
## rnn.py
### ```class RNN()```

#### ```RNN.calc_change(self, n, observed, expected, u, v)```

Calculates the required change in a weight/perceptron/bias between one layer
and the one before it.

#### ```RNN.train(self, sample)```

Changes weights based on one sample. A sample is a 2-dimensional array, consisting of:
1. Dimension 1: the collected frequencies at each timestep in the sample
2. Dimension 2: the input vector, and the expected output vector

i.e. [[input_1, expected_1], [input_2, expected_2], ... , [input_t, expected_t]]
