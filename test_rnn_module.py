import rnn
import random
import sys
import traceback

states = ["failed", "passed"]
def printErr(error):
    print("################# AN ERROR OCCURRED #################\n")
    print(error)
    print("################# END OF ERROR MESSAGE ################")
def printScore(score, total):
    print("Score: {0}/{1}".format(score, total))
def test_matrix(width, height, loud):
    score = 0
    total = 0
    p = 0

    # Test construction with parameters
    total += 1
    testm = 0
    try:
        testm = rnn.Matrix(width, height)
        if loud:
            testm.showItems(4)
        score += 1
        p = True
    except:
        p = 0
        e = sys.exc_info()[0]
        print("Error: {0}".format(e))
    if loud:
        print("Test construction with parameters: {0}".format(states[p]))

    # Test set item:
    r = round(random.random() * height) - 1  # row
    c = round(random.random() * width) - 1  # column
    val = round(random.random() * 1000, 2)  # val
    total += 1
    if (testm != 0):
        try:
            item = testm.setItem(r, c, val)
            score += 1
            p = True
        except:
            p = 0
            e = sys.exc_info()[0]
            print("Error: {0}".format(e))
    else:
        if loud:
            print("Could not execute 2nd test, as 1st failed")
        p = 0
    if loud:
        print("Test setter (exception checking): {0}".format(states[p]))

    # test get item:
    total += 1
    if (testm != 0):
        try:
            item = testm.getItem(r, c)
            if item == val:
                score += 1
                p = 1
            else:
                print(item, val)
                p = 0
        except:
            p = 0
            e = sys.exc_info()[0]
            if loud:
                print("Error: {0}".format(e))
    else:
        if loud:
            print("Could not execute 3rd test, as 1st failed")
    if loud:
        print("Test getter: {0}".format(states[p]))
        print("Score: {0}/{1}".format(score, total))
    return [score, total]

def test_multiply(loud):
    print("############### TEST MULTIPLY #################")
    score = 0
    total = 0
    p = 0

    # Test for invalid matrices (m1.width != m2.height)
    total += 1
    try:
        m1 = rnn.Matrix(10, 15)
        m2 = rnn.Matrix(13, 20)
        m3 = rnn.mat_multiply(m1, m2)
        if m3 == 0:
            score += 1
            p = 1
        else:
            p = 0
    except:
        p = 0
        e = sys.exc_info()[0]
        if loud:
            print("Error: {0}".format(e))
    if loud:
        print("Test for invalid matrices: {0}".format(states[p]))

    # Test for valid matrices:
    total += 1
    try:
        m1 = rnn.Matrix(10, 15)
        m2 = rnn.Matrix(6, 10)
        m3 = rnn.mat_multiply(m1, m2)
        if (m3 != 0):
            p = 1
            score += 1
            print("======== MATRIX 1: ========")
            m1.showItems(2)
            print("======== MATRIX 2: ========")
            m2.showItems(2)
            print("======== MATRIX 3: ========")
            m3.showItems(2)
        else:
            p = 0
    except:
        p = 0
        e = sys.exc_info()[0]
        if loud:
            print("Error: {0}".format(e))

    if loud:
        print("Valid matrices (exceptions): {0}".format(states[p]))

    print("Score: {0}/{1}".format(score, total))

def test_square_matrix(size, loud):
    print("######### TEST SQUARE MATRIX ##########")
    score = 0
    total = 0
    p = 0

    # Test initialisation
    total += 1
    m = rnn.SquareMatrix(10)
    try:
        m.showItems(2)
        score += 1
    except:
        e = sys.exc_info()[0]
        print("Error: {0}".format(e))
    print("Score: {0}/{1}".format(score, total))

    return score

def test_rnn(loud):
    print("######### TEST RNN ##########")
    score = 0
    total = 0
    p = 0

    # Test initialisation
    total += 1
    try:
        test = rnn.MyRNN(10, [10, 8], 5, 4)
        p = 1
        score += 1
        test.showEverything()
    except:
        e = traceback.format_exc()
        if loud:
            printErr(e)
        p = 0
    if loud:
        print("Testing initialisation: {0}".format(states[p]))

    print("Score: {0}/{1}".format(score, total))

def test_rnn_feed(loud):
    score = 0
    total = 0
    p = 0

    # Test for exceptions
    print("Testing with valid input")
    total += 1
    testRNN = rnn.MyRNN(8, [5, 5], 4, 4)
    try:
        output = testRNN.run([4,3,2.5,3,7.7,5.1,0.5,2.8])
        if loud:
            output.showItems(3)
        score += 1

    except:
        if loud:
            e = traceback.format_exc()
            printErr(e)

    # test with erroneous input
    print("Testing with erroneous input")
    total += 1
    try:
        testRNN = rnn.MyRNN(8, [5, 5], 4, 4)
        output = testRNN.run([2])
        if (output == 0):
            score += 1
    except:
        e = traceback.format_exc()
        if loud:
            printErr(e)
    printScore(score, total)
#test_matrix(10, 10, True)
#test_square_matrix(10, True)
#test_multiply(True)
test_rnn(True)
#test_rnn_feed(True)
