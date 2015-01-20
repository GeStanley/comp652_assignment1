__author__ = 'geoffrey'

import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def LinearRegression(feature_array, target_array):

    #perform the linear regression
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

    normalized_feature_array = BuildNormalizedArray(feature_array)

    weights[1] = LinearRegression(normalized_feature_array, target_array)

    return weights

def BuildNormalizedArray(original_array):

    absolute_array = numpy.absolute(original_array)
    max_index = absolute_array.argmax(axis=0)

    normalized_array = numpy.empty((original_array.shape[0], 0), dtype=float)

    for i in range(0, original_array.shape[1]):

        normalized_vector = original_array[:, i] / absolute_array[max_index[i], i]

        normalized_array = numpy.append(normalized_array, normalized_vector[:, numpy.newaxis], axis=1)

    return normalized_array


def PolyRegress(feature_array, target_array, degree):

    return 0


def BuildPolynomialArray(original_array, degree):

    polynomial_array = numpy.empty((original_array.shape[0], 0), dtype=float)

    for d in range(degree, 0, -1):

        degree_array = numpy.empty((original_array.shape[0], 0), dtype=float)

        for feature in range(0, original_array.shape[1]-1):

            exponents = original_array[:, feature] ** d

            degree_array = numpy.append(degree_array, exponents[:, numpy.newaxis], axis=1)

        polynomial_array = numpy.append(polynomial_array, degree_array, axis=1)

    #add a noise column of ones to the feature matrix
    array_ones = numpy.ones((polynomial_array[:, 0].size, 1))
    polynomial_array = numpy.append(polynomial_array, array_ones, axis=1)

    return polynomial_array

def CalculateError(feature_array, target_array, weights):

    error = 0

    for row in range(0, feature_array.shape[0]):

        estimate = numpy.sum(feature_array[row] * weights, axis=1)

        error += (estimate - target_array[row]) ** 2


    return error

def FiveFoldCrossValidation(feature_array, target_array):

    complete_array = numpy.append(feature_array, target_array[:, numpy.newaxis], axis=1)

    five_folds = numpy.array_split(complete_array, 5)

    #go through the 5 training cases
    for i in range(0, 5):

        training_data = numpy.empty((0, complete_array.shape[1]), dtype=float)

        #concatenate the 4 folds that are being used for training
        for j in range(0, 5):
            if i != j:
                training_data = numpy.append(training_data, five_folds[j], axis=0)

                training_features = training_data[:, 0:training_data.shape[1]-1]
                training_target = training_data[:, training_data.shape[1]-1]


        weights = LinearRegression(training_features, training_target[:, numpy.newaxis])

        training_error = CalculateError(training_features, training_target, weights.transpose())



        testing_features = five_folds[i][:, 0:five_folds[i].shape[1]-1]
        testing_target = five_folds[i][:, five_folds[i].shape[1]-1]

        testing_error = CalculateError(testing_features, testing_target, weights.transpose())


        print 'Fold #%s results are:' % (i+1)

        print 'Training error is : %s ' % training_error
        print 'Testing error is : %s' % testing_error




    return 0




array_x = numpy.loadtxt('hw1x.dat', float)
vector_y = numpy.loadtxt('hw1y.dat', float)

#add a noise column of ones to the feature matrix
array_ones = numpy.ones((array_x[:, 0].size, 1))
array_x = numpy.append(array_x, array_ones, axis=1)

FiveFoldCrossValidation(array_x, vector_y)

print FeatureNormalization(array_x, vector_y)

#matrix_x = numpy.matrix(array_x)

# print array_x[:, 0].shape
# print type(array_x[:, 0])
# print vector_y.shape
# print type(vector_y)

# poly = PolynomialFeatures(degree=2)
#
#
# model = Pipeline([('poly', poly),
#                 ('linear', regression)])
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
