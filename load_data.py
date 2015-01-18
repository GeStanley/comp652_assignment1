__author__ = 'geoffrey'

import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def LinearRegression(feature_array, target_array):

    feature_array_transposed = feature_array.transpose()

    dot_product = numpy.dot(feature_array_transposed, feature_array)

    inverse = numpy.linalg.inv(dot_product)

    dot_product = numpy.dot(feature_array_transposed, target_array)

    weights = numpy.dot(inverse, dot_product)

    return weights

def FeatureNormalization(feature_array, target_array):

    array_shape = feature_array.shape

    weights = numpy.zeros((2, array_shape[1]))

    weights[0] = LinearRegression(feature_array, target_array)

    max_index = feature_array.argmax(axis=0)
    normalized_feature_array = numpy.empty((feature_array.shape[0], 0), dtype=float)

    for i in range(0, array_shape[1]):

        normalized_vector = feature_array[:, i] / feature_array[max_index[i], i]

        normalized_feature_array = numpy.append(normalized_feature_array, normalized_vector[:, numpy.newaxis], axis=1)

    weights[1] = LinearRegression(normalized_feature_array, target_array)

    return weights

def PolyRegress():

    return 0

def FiveFoldCrossValidation():

    return 0




array_x = numpy.loadtxt('hw1x.dat', float)
vector_y = numpy.loadtxt('hw1y.dat', float)

array_ones = numpy.ones((array_x[:, 0].size, 1))

array_x = numpy.append(array_x, array_ones, axis=1)

print FeatureNormalization(array_x, vector_y)
#
# #matrix_x = numpy.matrix(array_x)
#
# # print array_x[:, 0].shape
# # print type(array_x[:, 0])
# # print vector_y.shape
# # print type(vector_y)
#
# # poly = PolynomialFeatures(degree=2)
# #
# #
# # model = Pipeline([('poly', poly),
# #                 ('linear', regression)])
#
# plt.figure(figsize=(14, 4))
#
# for i in range(0, 3):
#     ax = plt.subplot(1, 3, i+1)
#     plt.setp(ax)
#
#     regression = LinearRegression()
#
#     regression.fit(array_x[:, i, numpy.newaxis], vector_y[:, numpy.newaxis])
#
#     # Plot outputs
#     plt.scatter(array_x[:, i], vector_y[:, numpy.newaxis],  color='black')
#
#     print array_x[:, i]
#     print vector_y[:, numpy.newaxis]
#
#     sorted_x = numpy.sort(array_x[:, i, numpy.newaxis], axis=0)
#
#     plt.plot(sorted_x, regression.predict(sorted_x), color='blue', linewidth=3)
#
#     min_x = min(sorted_x) - 1
#     max_x = max(sorted_x) + 1
#
#     plt.axis([min_x, max_x, -20, 140])
#     plt.title("Feature %d" % i)
#
# # plt.xticks(())
# # plt.yticks(())
#
# plt.show()