__author__ = 'geoffrey'

import numpy
import pylab

my_array_x = numpy.loadtxt('hw2x.dat', float)
my_array_y = numpy.loadtxt('hw2y.dat', float)


print my_array_x[:, 0]
print my_array_y.size


pylab.scatter(my_array_x[:, 1], my_array_y)
pylab.show()