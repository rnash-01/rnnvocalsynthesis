################################################################################

# This is the matrices file for the Speech-to-Speech synthesis project,
# for the Neural Network training partition of the project.
# Author: Raphael Nash
# Date created: 22/12/2020
# Date updated: 25/01/2021

################################################################################

#                             # IMPORTS #                                      #
import math
import random

################################################################################

#                              # CLASSES #                                     #

class Matrix:
    def __init__(self, width, height):
        if (width > 0 and height > 0):
            self.width = width
            self.height = height
            self.matrix = [[random.random() for j in range(width)] for i in range(height)]
        else:
            self.width = 0
            self.height = 0
            self.matrix = 0


    def getwidth(self):
        return self.width

    def getheight(self):
        return self.height

    def showItems(self):
        if (self.width > 0 and self.height > 0):
            for i in range(self.height):
                rowstr = ""
                for j in range(self.width):
                    v = str(round(self.matrix[i][j], 2))
                rowstr += v + "\t"
                print(rowstr + "\n")

        else:
            print("Cannot show matrix items: improper attributes")

    def setItem(self, row, col, val):
        try:
            self.matrix[row][col] = val
            return 1
        except:
            return 0

    def getItem(self, row, col):
        try:
            return self.matrix[row][col]
        except IndexError:
            return "row or column exceeds range"
        except:
            return "unhandled exception"

################################################################################

#                            # ROUTINES #                                      #

def matrix_multiply(m1, m2):
    if (m1.width != m2.height):
        return 0
    else:
        shared = m1.getwidth()  # m1.width = m2.height; matrix multiplication
        newwidth = m2.getwidth()  # width of resultant matrix
        newheight = m1.getheight()  # height of resultant matrix

        try:
            m3 = Matrix(width, height)
            for i in range(height):
                for j in range(width):
                    sum = 0
                    for k in range(shared):  # multiply m1 row on m2 col
                        val1 = m1.getItem(i, k)  # row vector; current item
                        val2 = m2.getItem(k, j)  # column vector; current item
                        sum += val1 * val2  # mutliply the two, add to sum
                    m3.setItem(i, j, sum)
            return m3
        except ArithmeticError:
            print("arithmetic error (matrix_multiply)")
            return 1
        except IndexError:
            print("index error - access to height/width improper? (matrix_multiply)")
            return 2
        except:
            return 0

def make_vector(size, rdm):
    v = Matrix(1, size)
    if rdm == False:
        for i in range(size):
            v.setItem(i, 0, 0)

    return v
################################################################################
