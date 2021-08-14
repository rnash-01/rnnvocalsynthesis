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
import numpy as np

################################################################################

#                             # FUNCTIONS #                                #
def hexStringToArr(hexString):
    return [c for c in hexString]


def hexToBinary(hex, endianness):
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
    # Big endian: endianness = 0
    # Little endian: endianness = 1
    if(endianness == 1):
        binarr = binarr[::-1]
    return binarr

def binaryToDenary(bin, signed):
    # Encapsulate code in if statement to ensure bin array has elements
    if (len(bin) > 0):
        # Reverse if binary string is big endian

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

def denaryToBinary(n, endianness, bytes):
    binarr = []
    n = int(n)
    absn = abs(n)
    if (n >= 0):
        for i in range(bytes * 8, 0, -1):
            if absn >= np.power(2, i - 1):
                absn -= np.power(2, i - 1)
                binarr.append(1)
            else:
                binarr.append(0)
    elif n < 0:
        binarr.append(1)
        runsum = -np.power(2, bytes * 8 - 1)
        for i in range(bytes * 8 - 1, 0, -1):
            if (n >= runsum + np.power(2, i - 1)):
                binarr.append(1)
                runsum += np.power(2, i - 1)
            else:
                binarr.append(0)

    if (endianness == 1):
        binarr = binarr[::-1]
        new_bin = []
        for i in range(len(binarr)//8):
            bin_i = binarr[i*8:(i+1)*8]
            new_bin += bin_i[::-1]

        binarr = new_bin

    return binarr

def binaryToHex(bin, p):
    hexstring = ""
    if (len(bin)/8 != len(bin)//8):
        return 0
    else:
        vals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        for i in range(len(bin)//8):
            for k in range(2):
                ind = 0
                for j in range(4):
                    ind += np.power(2, 3-j) * bin[i * 8 + (k * 4) + j]
                hexstring += vals[ind]
    if(p):
        print(hexstring)

    bytearr = bytearray.fromhex(hexstring)
    return bytearr
