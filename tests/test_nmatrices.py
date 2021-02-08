############################# IMPORTS ##########################################

# to test, drag file into test folder so that python can detect it in runtime
from test_functions import *
from nmatrices import *

############################ NECESSARY GLOBALS #################################


############################## TESTING BEGINS HERE #############################

def arbitrary_op(val):
    return 100 * val

def randmatrix(maxwidth, maxheight):
    width = random.randint(1, maxwidth)
    height = random.randint(1, maxheight)
    m = Matrix(width, height, True)
    return m

def t_matrix_init():
    print("\n===== Matrix.__init__() =====\n")
    reset()
    print("Test 1: init matrix with correct params:")
    it()
    try:
        m = Matrix(10, 15, True)
        ins()
        print("Passed")

    except Exception as e:
        print("Failed")
        printerror(e)

    print("Test 2: init matrix with incorrect params:")
    it()
    try:
        m = Matrix(-56, 3.6, True)
        ins()
        print("Passed")
    except Exception as e:
        print("Failed")
        printerror(e)

    finishtest()

def t_matrix_showItems():
    print("\n===== Matrix.showItems() =====\n")
    reset()
    print("Test 1: generic matrix")
    it()
    try:
        m = randmatrix(12, 12)
        m.showItems()
        ins()
        print("No runtime errors - passed if print looks okay")

    except Exception as e:
        print("Failed")
        printerror(e)

    finishtest()

def t_matrix_setItem():
    print("\n===== Matrix.setItem() =====\n")
    reset()
    m = randmatrix(12, 12)
    print("Test 1: set with valid indices (all indices)")
    it()
    accuracy = 0
    iteration = 0
    try:
        for i in range(m.height):
            for j in range(m.width):
                val = round(random.random(), 4)
                m.setItem(i, j, val)
        ins()
        print("Passed")

    except Exception as e:
        print("Failed")
        printerror(e)

    print("Test 2: set with invalid indices (same matrix)")
    it()
    try:
        row = round((-1 + (random.random() * 2)) * m.height, 2)
        col = round((-1 + (random.random() * 2)) * m.width, 2)
        val = round(random.random(), 4)
        m.setItem(col, row, val)
        ins()
        print("Passed")

    except Exception as e:
        print("Failed")
        printerror(e)
    finishtest()


def t_matrix_getItem():
    print("\n===== Matrix.getItem() =====\n")
    reset()
    m = randmatrix(12, 12)
    print("Test 1: Test with valid indices")
    it()
    try:
        for i in range(m.height):
            for j in range(m.width):
                val = m.getItem(i, j)
        ins()
        print("Passed")

    except Exception as e:
        print("Failed")
        printerror(e)

    print("Test 2: Test with invalid indices (same matrix)")
    it()
    try:
        row = round((-1 + (random.random() * 2)) * m.height, 2)
        col = round((-1 + (random.random() * 2)) * m.width, 2)

        val = m.getItem(row, col)
        ins()
        print("Passed")

    except Exception as e:
        print("Failed")
        printerror(e)

    finishtest()

def t_matrix_multiply():
    print("\n===== matrix_multiply() =====\n")
    reset()
    m1height = random.randint(1, 5)
    m2width = random.randint(1, 5)
    shared = random.randint(1,5)

    print("Test 1: Valid params (m1.width = m2.height)")
    it()
    try:
        m1 = Matrix(shared, m1height, True)
        m2 = Matrix(m2width, shared, True)
        m3 = matrix_multiply(m1, m2)
        ins()
        print("Passed")
    except Exception as e:
        print("Failed")
        printerror(e)

    print("Test 2: Invalid params (m1.width != m2.height1)")
    it()
    m1width = random.randint(1, 5)
    m2height = m1width  # This is so that it will enter the while loop
    while m2height == m1width:  # Ensure that m1width != m2height
        m2height = random.randint(1, 5)

    try:
        m1 = Matrix(m1width, m1height, True)
        m2 = Matrix(m2width, m2height, True)
        m3 = matrix_multiply(m1, m2)
        if m3 == 0:
            ins()
            print("Passed")
        else:
            print("Failed; did not return 0 for erroneous data")

    except Exception as e:
        print("Failed")
        printerror(e)

    finishtest()
    # Note: line 155; in lieu of a do/while feature in Python, I just ensured
    # that the while condition was true to begin with, so that I could
    # properly ensure that the two are different in the end

def t_matrix_add():
    print("\n===== matrix_add() =====\n")
    print("Test:")
    it()
    width = random.randint(1, 10)
    height = random.randint(1, 10)
    try:
        m1 = Matrix(width, height, True)
        m1.showItems()
        print("ADDING TO")
        m2 = Matrix(width, height, True)
        m2.showItems()
        m3 = matrix_add(m1, m2)
        print("=")
        m3.showItems()

        print("Passed")
        ins()
    except Exception as e:
        print("Failed")
        printerror(e)


def t_make_vector():
    print("\n===== make_vector() =====\n")

    print("Test 1: With rdm=True")

    it()
    size = random.randint(1, 10)
    try:
        v = make_vector(size, True)
        v.showItems()
        ins()
        print("Passed")
    except Exception as e:
        print("Failed")
        printerror(e)

    print("Test 2: With rdm=False")
    it()
    try:
        v = make_vector(size, False)
        v.showItems()
        zerofound = False
        for i in range(size):
            if (v.getItem(i, 0) != 0):
                print(v.getItem(i, 0))
                zerofound = True
        if zerofound:
            print("Failed; non-zero vector")
        else:
            ins()
            print("Passed")
    except Exception as e:
        print("Failed")
        printerror(e)

    finishtest()

def t_vector_operation():
    print("\n===== t_vector_operation() =====\n")

    print("Test: using arbitrary_op (f(n) = 100n)")
    it()
    v = make_vector(15, True)
    try:
        print("First vector:")
        v.showItems()
        newv = vector_operation(v, arbitrary_op)
        print("Second vector: ")
        newv.showItems()
        ins()
        print("Passed")
    except Exception as e:
        print("Failed")
        printerror(e)

def main():
    t_matrix_init()
    t_matrix_showItems()
    t_matrix_setItem()
    t_matrix_getItem()
    t_matrix_multiply()
    t_matrix_add()
    t_make_vector()
    t_vector_operation()

    gs = getgs()
    gt = getgt()
    if gt != 0:
        ratio = round((gs/gt) * 100, 2)
    else:
        ratio = 0
    print("TOTAL SCORE: {0}/{1} ({2} %)".format(gs, gt, ratio))

main()
