
# ReLU function

def ReLU(val):
    if (val >= 0):
        return val
    else:
        return 0

# derivative of ReLU function

def ReLU_prime(val):
    if (val >= 0):
        return 1
    else:
        return 0

# Sigmoid function
def sigmoid(val):
    return 1/(1 + math.pow(math.e, -val))

# derivative of sigmoid function
def sigmoid_prime(val):
    return sigmoid(val) * (1 - sigmoid(val))
