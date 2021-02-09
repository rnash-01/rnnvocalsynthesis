############################# IMPORTS ##########################################

# to test, drag file into test folder so that python can detect it in runtime
from test_functions import *
from dataconversion import *

############################ NECESSARY GLOBALS #################################


############################## TESTING BEGINS HERE #############################

def t_hexStringToArr():
    reset()
    print("======== hexToBinary() =======")
    print("Test 1: normal hex string")
    it()
    try:
        passed = True
        hexString = 'ff002305'
        hexArr = hexStringToArr(hexString)
        for c in range(len(hexString)):
            if hexString[c] != hexArr[c]:
                passed = False
        if passed:
            print("Passed")
            ins()

    except Exception as e:
        print("Failed (error)")
        printerror(e)
    finishtest()

def t_hexToBinary():
    reset()
    print("======== hexToBinary() =======")
    print("Test 1: normal hex array")
    it()
    hexDigits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E']
    hex = ['f', 'f', '2', '6', 'e', 'a', '7', '0']
    binary = hexToBinary(hex)
    expected = [1,1,1,1, 1,1,1,1, 0,0,1,0, 0,1,1,0, 1,1,1,0, 1,0,1,0, 0,1,1,1, 0,0,0,0]
    err = False

    for i in range(len(expected)):
        if (expected[i] != binary[i]):
            err = True

    if err:
        print("Failed")
    else:
        print("Passed")
        ins()

    return score

def t_binaryToDenary():
    reset()
    print("======== binaryToDenary() =======")
    bintest = [1,1,0,1]

    print("Test 1: Signed, little")
    it()
    expected = -5
    if(binaryToDenary(bintest, 1, 0) == expected):
        print("Passed")
        ins()
    else:
        print("Failed")

    print("Test 2: Unsigned, little")
    it()
    expected = 11
    if(binaryToDenary(bintest, 0, 0) == expected):
        print("Passed")
        ins()
    else:
        print("Failed")

    print("Test 3: Signed, big")
    it()
    expected = -3
    if(binaryToDenary(bintest, 1, 1) == expected):
        print("Passed")
        ins()
    else:
        print("Failed")

    print("Test 4: Unsigned, big")
    it()
    expected = 13
    if(binaryToDenary(bintest, 0, 1) == expected):
        print("Passed")
        ins()
    else:
        print("Failed")

    finishtest()
    return score

def main():
    t_hexStringToArr()
    t_hexToBinary()
    t_binaryToDenary()
    print("Total: {0}/{1}".format(getgs(), getgt()))

main()
