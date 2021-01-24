import math
import random

# Define classes for each type of structure in the network:
# - layer
# - weight (matrix)

class Matrix:
    def __init__(self, width, height):  # Construct matrix
        self.w = width
        self.h = height
        m = [[round(random.uniform(0,1), 2) for i in range(width)] for j in range(height)]
        self.matrix = m
    def showItems(self, places):  # For testing purposes
        for row in self.matrix:
            ps = ""
            for col in row:
                ps += str(round(col, places)) + "\t"
            print(ps + "\n")
    def setItem(self, row, col, value):
        self.matrix[row][col] = value

    def getItem(self, row, col):
        return self.matrix[row][col]

class Vector(Matrix):
    def __init__(self, size):
        self.elements = [0.0 for i in range(size)]
        self.size = size
    def getElement(self, index):
        try:
            return self.neurons[index]
        except (IndexError):
            return 0
    def setElement(self, index, value):
        try:
            self.elements[index] = value
        except (IndexError):
            pass
    def reset(self):
        self.__init__(self.size)
    def showElements(self):
        print(self.elements)

class MyRNN:
    # self.input
    # self.hidden
    # self.output
    # self.weights
    # self.biases
    # self.hidden_weights
    # self.hidden_biases
    # self.activation
    # self.hiddenactivation
    def __init__(self, input, hidden, output, timesteps):
        # Create layers
        self.hidden = []
        self.input = Matrix(1, input)
        for i in range(len(hidden)):
            l = Matrix(1, hidden[i])
            self.hidden.append(l)
        self.output = Matrix(1, output)

        # Initialise weights
        self.weights = []
        self.biases = []
        tw = 0  # Temporary width (holds width value)
        th = 0  # Temporary height (holds height value)

        # input to first hidden:
        tw = self.input.h
        th = self.hidden[0].h
        self.weights.append(Matrix(tw, th))

        # hidden layers to hidden layers:
        for i in range(len(self.hidden) - 1):
            tw = self.hidden[i].h
            th = self.hidden[i + 1].h
            self.weights.append(Matrix(tw, th))

        # final hidden layer to output
        final = len(self.hidden) - 1
        tw = self.hidden[final].h
        th = self.output.h
        self.weights.append(Matrix(tw, th))

        # Initialise biases
        # hidden layers
        for i in range(len(self.hidden)):
            self.biases.append(Matrix(1, self.hidden[i].h))

        # output layer
        self.biases.append(Matrix(1, self.output.h))

        # Hidden State
        self.hidden_state = Matrix(1, self.output.h)
        # Initialise hidden weights
        self.hidden_weights = Matrix(self.output.h, self.hidden[0].h)

    # Assign activation function

    def activation(self, val):
        return ReLU(val)  # CHANGE AS NEEDED

    def updateLayer(self, l1, l2, w):
        current = l1
        currentweight = w
        result = mat_multiply(currentweight, current)
        print("WEIGHTS WIDTH: {0} | WEIGHTS HEIGHT: {1}".format(currentweight.w, currentweight.h))
        for j in range(result.h):
            val = self.activation(result.getItem(j, 0))
            l2.setItem(j, 0, val)

    def run(self, inp):  # inp As List - input.
        pass

    def calcGradient(self):
        pass

    def backProp(self):
        pass
    def train(self):

    def showEverything(self):
        print("############ LAYERS #############")
        print("##### INPUT #####")
        self.input.showItems(4)

        print("#### HIDDEN #####")
        for h in self.hidden:
            print("#####\n")
            h.showItems(4)

        print("#### OUTPUT #####")
        self.output.showItems(4)

        print("############ WEIGHTS ############")
        for w in self.weights:
            print("###############################")
            w.showItems(4)

        print("############ BIASES #############")
        for b in self.biases:
            print("#########")
            b.showItems(4)

        print("######### HIDDEN STATE ##########")
        self.hidden_state.showItems(4)

# Define linear algebra functions
def mat_multiply(m1, m2):
    width = m2.w
    height = m1.h
    if m1.w == m2.h:  # m1.width = m2.h for mutliplication to work
        shared = m1.w
        m3 = Matrix(width, height)
        for i in range(height):
            for j in range(width):
                el = 0
                for k in range(shared):
                    temp1 = m1.getItem(i, k)
                    temp2 = m2.getItem(k, j)
                    result = temp1 * temp2
                    el += result
                m3.setItem(i, j, el)
        return m3
    else:
        return 0

# Define potential non-linearity functions (sigmoid and ReLU functions)
def ReLU(val):
    if (val >= 0):
        return val
    else:
        return 0

def sigmoid(val):
    return 1/(1 + math.exp(-val))
