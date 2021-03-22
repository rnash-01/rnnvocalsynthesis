######################### TESTING: COMMON FUNCTIONS ############################


# This is a script to grap functions from.
# It cannot be imported, because import doesn't include the global variables I
# need, nor does it reference the global variables I create when I call the
# imported functions. They must be explicitly defined in each test.
# Think of this script as a repository for those functions.

# ^ if this is wrong, feel free to correct!!! I still make dumb mistakes!


# total = local total for each unit test (defined globally, reset each time)
# score = local score for each unit test (defined globally, reset each time)
# gt = global total for all tests (defined globally)
# gs = global score for all tests (defined globally)

import traceback

total = 0
score = 0
gt = 0
gs = 0

def it():  # Increment total
    global total
    global gt
    total += 1
    gt += 1

def ins():  # Increment score
    global score
    global gs
    score += 1
    gs += 1

def reset():
    global total
    global score
    total = 0
    score = 0

def printscore():
    print("Test score: {0}/{1}".format(score, total))

def finishtest():
    printscore()
    #print(gt, gs)

def startTest(fname):
    reset()
    print("========================================")
    print("Testing function: " + fname + "()")
    print("========================================")


def printerror(e):
    print("================= ERROR ================")
    track = traceback.format_exc()
    print(track)
    print(e)
    print("========================================")

def getgt():
    global gt
    return gt
def getgs():
    global gs
    return gs
