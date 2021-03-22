################################################################################

# This is the data conversion file for the Speech-to-Speech synthesis project,
# for the Neural Network training partition of the project.
# Author: Raphael Nash
# Date created: 09/02/2021
# Date updated: 09/02/2021

################################################################################

#                              # IMPORTS #                                 #

from scipy.fft import fft
import math

################################################################################

#                             # FUNCTIONS #                                #
def hexStringToArr(hexString):
    return [c for c in hexString]


def hexToBinary(hex):
    binarr = []
    hexMap = {'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15}
    for i in range(len(hex)):
        if hex[i] in hexMap:
            val = hexMap[hex[i]]
        else:
            val = int(hex[i])
        j = 3
        while j >= 0:
            if (val >= math.pow(2, j)):
                val -= math.pow(2, j)
                binarr.append(1)
            else:
                binarr.append(0)
            j -= 1
    return binarr

def binaryToDenary(bin, signed, endianness):

    # Encapsulate code in if statement to ensure bin array has elements
    if (len(bin) > 0):

        # Reverse if binary string is big endian
        # Big endian: endianness = 1
        # Little endian: endianness = 0
        if endianness == 1:
            bin = bin[::-1]

        # Go through binary array
        i = 0
        sum = 0
        while i < len(bin) - 1:
            sum += bin[i] * math.pow(2, i)
            i += 1

        # Check if signed/modify accordingly
        if signed == 1:
            sum -= bin[i] * math.pow(2, i)
        else:
            sum += bin[i] * math.pow(2, i)

    else:
        sum = 0
    return sum
