__author__ = 'geoffrey'

import numpy
import pylab
from sklearn import linear_model

def PolyRegress():
    return null


def LinearRegressionWithNormalization(x_matrix, y_vector):
    regr = linear_model.LinearRegression()

    regr.fit(x_matrix, y_vector)

    print regr.coef_

    pylab.scatter(x_matrix[:, 0], x_matrix[:, 1], c=y_vector)
    pylab.show()

    max_col0 = max(x_matrix[:, 0])
    max_col1 = max(x_matrix[:, 1])

    normalized = numpy.c_[x_matrix[:, 0] / max_col0, x_matrix[:, 1] / max_col1]


    pylab.scatter(normalized[:, 0], normalized[:, 1], c=y_vector)
    pylab.show()


    #pylab.scatter()

#load the data files
my_array_x = numpy.loadtxt('hw2x.dat', float)
my_array_y = numpy.loadtxt('hw2y.dat', float)

#convert numpy.darray into a matrix
my_matrix_x = numpy.matrix(my_array_x)
#add one to the last column of the x matrix
my_matrix_x = numpy.c_[my_matrix_x, numpy.ones(my_matrix_x[:, 0].size)]

LinearRegressionWithNormalization(my_matrix_x, my_array_y)
