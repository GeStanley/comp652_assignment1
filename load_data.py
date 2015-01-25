__author__ = 'geoffrey'

import numpy
from math import sqrt
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

def LinearRegressionWithL2Regularization(feature_array, target_array, lamb):

    feature_array_transposed = feature_array.transpose()

    dot_product = numpy.dot(feature_array_transposed, feature_array)


    identity = numpy.identity(feature_array.shape[1], dtype=float)

    lamb_result = numpy.dot(lamb, identity)


    inverse = numpy.linalg.inv(dot_product + lamb_result)


    right_part = numpy.dot(feature_array_transposed, target_array)


    return numpy.dot(inverse, right_part)

# def FeatureNormalization(feature_array, target_array):
#
#     array_shape = feature_array.shape
#
#     weights = numpy.zeros((2, array_shape[1]))
#
#     weights[0] = LinearRegression(feature_array, target_array)
#
#     normalized_feature_array = BuildNormalizedArray(feature_array)
#
#     weights[1] = LinearRegression(normalized_feature_array, target_array)
#
#     return weights

def BuildNormalizedArray(original_array):

    absolute_array = numpy.absolute(original_array)
    max_index = absolute_array.argmax(axis=0)

    normalized_array = numpy.empty((original_array.shape[0], 0), dtype=float)

    for i in range(0, original_array.shape[1]):

        normalized_vector = original_array[:, i] / absolute_array[max_index[i], i]

        normalized_array = numpy.append(normalized_array, normalized_vector[:, numpy.newaxis], axis=1)

    return normalized_array

def PolyRegress(original_array, degree):

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

        target = numpy.sum(target_array[row])

        error += (estimate - target) ** 2


    return error / target_array.shape[0]

def FixingFiveFold():
    return 0

def FiveFoldCrossValidation(feature_array, target_array, lamb = None):

    complete_array = numpy.append(feature_array, target_array[:, numpy.newaxis], axis=1)

    five_folds = numpy.array_split(complete_array, 5)

    training_errors = {}
    testing_errors = {}
    all_weights = {}

    #go through the 5 training cases
    for i in range(0, 5):

        training_data = numpy.empty((0, complete_array.shape[1]), dtype=float)

        #concatenate the 4 folds that are being used for training
        for j in range(0, 5):
            if i != j:
                training_data = numpy.append(training_data, five_folds[j], axis=0)

                training_features = training_data[:, 0:training_data.shape[1]-1]
                training_target = training_data[:, training_data.shape[1]-1]


        if lamb==None:
            weights = LinearRegression(training_features, training_target[:, numpy.newaxis])
        else:
            weights = LinearRegressionWithL2Regularization(training_features, training_target[:, numpy.newaxis], lamb)

        training_error = CalculateError(training_features, training_target, weights.transpose())

        testing_features = five_folds[i][:, 0:five_folds[i].shape[1]-1]
        testing_target = five_folds[i][:, five_folds[i].shape[1]-1]

        testing_error = CalculateError(testing_features, testing_target, weights.transpose())

        training_errors[i+1] = training_error
        testing_errors[i+1] = testing_error
        all_weights[i+1] = weights.transpose()

        print 'Fold #%s results are:' % (i+1)

        print 'Training error is : %s ' % training_error
        print 'Testing error is : %s' % testing_error


    key = min(testing_errors, key=testing_errors.get)

    print 'Optimal run is: %s' % key
    print 'Optimal weight vector is: %s' % (all_weights[key])

    training_list = []
    testing_list = []

    for key in training_errors.keys():
        training_list.append(training_errors[key])
        testing_list.append(testing_errors[key])


    result = {'train_avg': mean(training_list),
              'train_std': stddev(training_list),
              'test_avg': mean(testing_list),
              'test_std': stddev(testing_list),
              'weights': all_weights[key]}

    return result


def GradientDescent(feature_array, target_array, max_iterations, alpha, delta, lamb):

    theta = numpy.ones(feature_array.shape[1])

    x_transpose = feature_array.transpose()

    m = feature_array.shape[0]

    for iter in range(0, max_iterations):

        for i in range(0, m):

            x = feature_array[i]
            y = target_array[i]

            hypothesis = numpy.dot(theta, x)

            if abs(y - hypothesis) <= delta:
                loss = hypothesis - target_array

                J = numpy.dot(x_transpose, loss)

                gradient = J + lamb/2 * numpy.dot(theta.transpose(), theta)

            else:
                loss = hypothesis - target_array

                J  = delta * loss

                gradient = J + lamb/2 * numpy.dot(theta.transpose(), theta)

            theta = theta - alpha * gradient


    theta = theta + lamb/2 * numpy.dot(x_transpose, feature_array)

    return theta

def mean(lst):
    """calculates mean"""
    return float(sum(lst) / len(lst))

def stddev(lst):
    """returns the standard deviation of lst"""
    mn = mean(lst)
    variance = sum([(e-mn)**2 for e in lst])
    return sqrt(variance)


if __name__ == '__main__':

    #############
    #question 1 a
    #############
    print "QUESTION 1 A"
    array_x = numpy.loadtxt('hw1x.dat', float)
    vector_y = numpy.loadtxt('hw1y.dat', float)

    #add a noise column of ones to the feature matrix
    array_ones = numpy.ones((array_x[:, 0].size, 1))
    array_x = numpy.append(array_x, array_ones, axis=1)

    #############
    #question 1 b
    #############
    print
    print
    print "QUESTION 1 B"
    array_x_norm  = BuildNormalizedArray(array_x)

    array_x_d2 = PolyRegress(array_x, 2)
    array_x_d2_norm = BuildNormalizedArray(array_x_d2)

    array_x_d3 = PolyRegress(array_x, 3)
    array_x_d3_norm = BuildNormalizedArray(array_x_d3)

    array_x_d4 = PolyRegress(array_x, 4)
    array_x_d4_norm = BuildNormalizedArray(array_x_d4)

    array_x_d5 = PolyRegress(array_x, 5)
    array_x_d5_norm = BuildNormalizedArray(array_x_d5)

    print LinearRegression(array_x, vector_y)
    print LinearRegression(array_x_norm, vector_y)

    print LinearRegression(array_x_d2, vector_y)
    print LinearRegression(array_x_d2_norm, vector_y)

    print LinearRegression(array_x_d3, vector_y)
    print LinearRegression(array_x_d3_norm, vector_y)

    print LinearRegression(array_x_d4, vector_y)
    print LinearRegression(array_x_d4_norm, vector_y)

    print LinearRegression(array_x_d5, vector_y)
    print LinearRegression(array_x_d5_norm, vector_y)

    #############
    #question 1 d
    #############
    print
    print
    print "QUESTION 1 D"
    FiveFoldCrossValidation(array_x, vector_y)

    #############
    #question 1 e
    #############
    print
    print
    print "QUESTION 1 E"

    stats = {}

    for d in range(0, 5):
        print
        print 'Running regression on polynomials of degree : %s' % str(d+1)
        poly_array = PolyRegress(array_x, d+1)
        poly_norm = BuildNormalizedArray(poly_array)

        stats[d] = FiveFoldCrossValidation(poly_array, vector_y)

    print
    print

    print 'degree   terror  tstd    verror  vstd'
    for key in stats.keys():
        print '%s   %9.5f  %9.5f  %9.5f  %9.5f  ' % (key+1,
                                         stats[key]['train_avg'],
                                         stats[key]['train_std'],
                                         stats[key]['test_avg'],
                                         stats[key]['test_std'])

    #############
    #question 1 f
    #############
    print
    print
    print "QUESTION 1 F"

    terror = []
    verror = []
    lambdas = []
    weights = []

    for i in numpy.arange(1, 500, 1):

        print i
        errors = FiveFoldCrossValidation(array_x_d4_norm, vector_y, i)
        lambdas.append(i)
        terror.append(errors['train_avg'])
        verror.append(errors['test_avg'])
        weights.append(errors['weights'])


    print errors

    print lambdas
    print terror
    print verror

    one_over_lambdas = [float(1.0/l) for l in lambdas]

    print one_over_lambdas

    plt.plot(one_over_lambdas, terror)
    plt.plot(one_over_lambdas, verror)

    plt.ylabel('mean squared error')
    plt.xlabel('1/$\lambda$')

    plt.show()
    for degree in range(0, weights[0].shape[0]):
        weight_list = []
        for element in weights:
            weight_list.append(element[degree])

        plt.plot(lambdas, weight_list)


    plt.ylabel('mean squared error')
    plt.xlabel('$\lambda$')

    plt.show()

    #############
    #question 2 c
    #############
    print "QUESTION 2 C"



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
