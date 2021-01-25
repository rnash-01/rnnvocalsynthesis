################################################################################
#                  This is a test script for nmatrices.py                      #
# Author: Raphael Nash                                                         #
# Date created: 25/01/2021                                                     #
# Date updated: 25/01/2021                                                     #
################################################################################

#                             # IMPORTS #                                      #
import nmatrices
import math

################################################################################
#                              # GLOBAL #                                      #

total = 0
current = 0

################################################################################

#                             # UNIT TESTS #                                   #
def resetstats():
    total = 0
    current = 0

def it():  # it = increment_total
    total += 1

def as(score):  # as = add score
    current += score

def matrix_init():
    resetstats()

    print("Test 1: instantiate matrix class with correct params")
    it()
    try:
        m = Matrix(10, 5)
        print("Status: passed")
        current += score
    except:
        print("Status: failed")

    print("Test 2: instantiate matrix class with incorrect params")
    it()
    try:
        m = Matrix(-5, -8.6)
        print("Status: passed")
        current += score
    except:
        print("Status: failed")
