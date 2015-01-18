__author__ = 'geoffrey'

import numpy
import pylab

array_x = numpy.loadtxt('hw1x.dat', float)
vector_y = numpy.loadtxt('hw1y.dat', float)

array_ones = numpy.ones((array_x[:, 0].size, 1))

array_x = numpy.append(array_x, array_ones, axis=1)

matrix_x = numpy.matrix(array_x)

print matrix_x
print type(matrix_x)


pylab.scatter(array_x[:, 1], vector_y)
pylab.show()