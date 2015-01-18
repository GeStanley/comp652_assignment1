__author__ = 'geoffrey'

import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

array_x = numpy.loadtxt('hw1x.dat', float)
vector_y = numpy.loadtxt('hw1y.dat', float)

array_ones = numpy.ones((array_x[:, 0].size, 1))

array_x = numpy.append(array_x, array_ones, axis=1)

#matrix_x = numpy.matrix(array_x)

# print array_x[:, 0].shape
# print type(array_x[:, 0])
# print vector_y.shape
# print type(vector_y)

poly = PolynomialFeatures(degree=2)

model = Pipeline([('poly', poly),
                ('linear', LinearRegression())])




model.fit(array_x[:, 1, numpy.newaxis], vector_y[:, numpy.newaxis])


# Plot outputs
plt.scatter(array_x[:, 1], vector_y[:, numpy.newaxis],  color='black')

print array_x[:, 1, numpy.newaxis]
print model.predict(array_x[:, 1, numpy.newaxis])

sorted = numpy.sort(array_x[:, 1, numpy.newaxis], axis=0)
print sorted

plt.plot(sorted, model.predict(sorted), color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

plt.show()