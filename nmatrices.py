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
    def __init__(self, width, height, rnd):
        if (width > 0 and height > 0):
            self.width = width
            self.height = height
            if rnd:
                self.matrix = [[random.random() for j in range(width)] for i in range(height)]
            else:
                self.matrix = [[0 for j in range(width)] for i in range(height)]
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
            m3 = Matrix(width, height, False)
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

def matrix_add(m1, m2):
    if (m1.getwidth() == m2.getwidth() && m1.getheight() == m2.getheight()):
        width = m1.getwidth()
        height = m1.getheight()
        m3 = Matrix(width, height, False1)
        for i in range(height):
            for j in range(width):
                val1 = m1.getItem(i, j)
                val2 = m2.getItem(i, j)
                m3.setItem(i, j, val1 + val2)

        return m3
    else:
        return 1



def make_vector(size, rdm):
    if rdm == False:
        v = Matrix(1, size, False)
    else:
        v = Matrix(1, size, True)
    return v

def copy_matrix(m):
    width = m.getwidth()
    height = m.getheight()
    newmatrix = Matrix(width, height, False)
    for i in range(height):
        for j in range(width):
            val = m.getItem(i, j)
            newmatrix.setItem(i, j, val)
    return newmatrix
################################################################################
