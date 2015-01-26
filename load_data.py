__author__ = 'geoffrey'

import numpy
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.utils import shuffle


def LinearRegression(feature_array, target_array):

    # perform the linear regression
    feature_array_transposed = feature_array.transpose()

    dot_product = numpy.dot(feature_array_transposed, feature_array)

    inverse = numpy.linalg.inv(dot_product)

    dot_product = numpy.dot(feature_array_transposed, target_array)

    return numpy.dot(inverse, dot_product)

def LinearRegressionWithL2Regularization(feature_array, target_array, lamb):

    feature_array_transposed = feature_array.transpose()

    dot_product = numpy.dot(feature_array_transposed, feature_array)

    identity = numpy.identity(feature_array.shape[1], dtype=float)

    lamb_result = numpy.dot(lamb, identity)

    inverse = numpy.linalg.inv(dot_product + lamb_result)

    right_part = numpy.dot(feature_array_transposed, target_array)

    return numpy.dot(inverse, right_part)

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

    # add a noise column of ones to the feature matrix
    array_noise = numpy.ones((polynomial_array[:, 0].size, 1))
    polynomial_array = numpy.append(polynomial_array, array_noise, axis=1)

    return polynomial_array

def CalculateError(feature_array, target_array, weights):

    error = 0

    for row in range(0, feature_array.shape[0]):

        estimate = numpy.sum(feature_array[row] * weights, axis=1)

        target = numpy.sum(target_array[row])

        error += (estimate - target) ** 2


    return error / target_array.shape[0]

def FiveFoldCrossValidation(feature_array, target_array, function='linear', degree=None, delta=None, lamb=None):

    complete_array = numpy.append(BuildNormalizedArray(feature_array), target_array[:, numpy.newaxis], axis=1)

    complete_array = shuffle(complete_array, random_state=125)

    five_folds = numpy.array_split(complete_array, 5)

    returned_statistics = {}

    # go through the 5 training cases
    for i in range(0, 5):

        training_data = numpy.empty((0, complete_array.shape[1]), dtype=float)
        validation_data = numpy.empty((0, complete_array.shape[1]), dtype=float)
        testing_data = numpy.empty((0, complete_array.shape[1]), dtype=float)

        for j in range(0, 5):
            if j == i:
                testing_data = numpy.append(testing_data, five_folds[j], axis=0)
            elif j == (i+1) % 5:
                validation_data = numpy.append(validation_data, five_folds[j], axis=0)
            else:
                training_data = numpy.append(training_data, five_folds[j], axis=0)

        # split the training data into features and targets
        training_features = training_data[:, 0:training_data.shape[1]-1]
        training_target = training_data[:, training_data.shape[1]-1]

        # split the validation data into features and targets
        validation_features = validation_data[:, 0:validation_data.shape[1]-1]
        validation_target = validation_data[:, validation_data.shape[1]-1]

        # split the testing data into features and targets
        testing_features = testing_data[:, 0:testing_data.shape[1]-1]
        testing_target = testing_data[:, testing_data.shape[1]-1]

        validation_errors = {}
        hypothesis_weights = {}

        # find the hypothesis class with the lowest error
        if degree is None:
            start = 0
            end = 5
        else:
            start = degree-1
            end = degree


        for d in range(start, end):

            poly_training_features = PolyRegress(training_features, d+1)
            poly_validation_features = PolyRegress(validation_features, d+1)

            # perform regression using the training set
            if function=='linear':
                weights = LinearRegression(poly_training_features,
                                           training_target[:, numpy.newaxis])
            elif function=='regularize':
                weights = LinearRegressionWithL2Regularization(poly_training_features,
                                                               training_target[:, numpy.newaxis],
                                                               lamb)
            elif function=='gradient':
                weights = GradientDescent(poly_training_features,
                                          training_target[:, numpy.newaxis],
                                          100,
                                          delta)

            validation_error = CalculateError(poly_validation_features, validation_target, weights.transpose())

            validation_errors[d] = validation_error

            hypothesis_weights[d] = weights.transpose()


        min_key = min(validation_errors, key=validation_errors.get)

        optimal_degree = min_key + 1
        optimal_weights = hypothesis_weights[min_key]

        poly_training_features = PolyRegress(training_features, optimal_degree)
        poly_validation_features = PolyRegress(validation_features, optimal_degree)
        poly_testing_features = PolyRegress(testing_features, optimal_degree)

        training_error = CalculateError(numpy.append(poly_training_features, poly_validation_features, axis=0),
                                        numpy.append(training_target, validation_target, axis=0),
                                        optimal_weights)

        testing_error = CalculateError(poly_testing_features, testing_target, optimal_weights)

        fold_data = {'optimal_degree': optimal_degree,
                     'training_error': training_error,
                     'testing_error': testing_error,
                     'optimal_weights': optimal_weights}

        returned_statistics[i] = fold_data

    return returned_statistics

def GradientDescent(feature_array, target_array, max_iterations, delta, alpha=0.01):


    theta = numpy.ones(feature_array.shape[1])

    x_transpose = feature_array.transpose()

    m = feature_array.shape[0]

    for iteration in range(0, max_iterations):



        for i in range(0, m):
            gradient = 0
            x = feature_array[i]
            y = target_array[i]

            hypothesis = numpy.dot(theta, x)
            loss = y - hypothesis

            print loss

            if abs(loss) <= delta:

                J = -x * loss

                gradient = -J

            else:

                if loss < 0:
                    gradient = delta * x * loss
                else:
                    gradient = -delta * x * loss

            print gradient
            print alpha
            print theta
            theta = theta - alpha * gradient
            print theta

    #theta = theta + numpy.dot(x_transpose, feature_array)


    return theta[:, numpy.newaxis]

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
    # question 1 a
    #############
    array_x = numpy.loadtxt('hw1x.dat', float)
    vector_y = numpy.loadtxt('hw1y.dat', float)

    # add a noise column of ones to the feature matrix
    array_ones = numpy.ones((array_x[:, 0].size, 1))
    array_x = numpy.append(array_x, array_ones, axis=1)


    #############
    # question 1 e
    #############
    # print
    # print
    # print "QUESTION 1 E"
    # stats = FiveFoldCrossValidation(array_x, vector_y)
    #
    # for key in stats.keys():
    #     print stats[key]


    #############
    # question 1 f
    #############
    print
    print
    print "QUESTION 1 F"

    train_error = []
    test_error = []
    lambdas = []
    weights = []
    #
    # for i in numpy.arange(1, 100, 1):
    #
    #     errors = FiveFoldCrossValidation(array_x, vector_y, function='regularize', degree=4, lamb=i)
    #     lambdas.append(i)
    #     train_error.append(errors[0]['training_error'])
    #     test_error.append(errors[0]['testing_error'])
    #     weights.append(errors[0]['optimal_weights'])
    #
    #
    # one_over_lambdas = [float(1.0/l) for l in lambdas]
    #
    # line_train = plt.plot(one_over_lambdas, train_error, label="Training Error")
    # line_test = plt.plot(one_over_lambdas, test_error, label="Testing Error")
    #
    # plt.legend(loc=0, borderaxespad=0.)
    # plt.ylabel('mean squared error')
    # plt.xlabel('1/$\lambda$')
    #
    # plt.show()
    #
    # for degree in range(0, weights[0].shape[0]):
    #     weight_list = []
    #     for element in weights:
    #
    #         weight_list.append(element[degree])
    #
    #     plt.plot(lambdas, weight_list)
    #
    #
    # plt.ylabel('mean squared error')
    # plt.xlabel('$\lambda$')
    #
    # plt.show()

    #############
    # question 2 c
    #############
    print "QUESTION 2 C"

    weights = GradientDescent(BuildNormalizedArray(PolyRegress(array_x, 4)),
                          vector_y,
                          max_iterations=100,
                          delta = 10)

    print weights

    train_error = []
    test_error = []
    deltas = []
    weights = []

    for i in numpy.arange(1, 100, 1):

        errors = FiveFoldCrossValidation(array_x, vector_y, function='gradient', degree=4, delta=i)
        deltas.append(i)
        train_error.append(errors[0]['training_error'])
        test_error.append(errors[0]['testing_error'])
        weights.append(errors[0]['optimal_weights'])



    line_train = plt.plot(deltas, train_error, label="Training Error")
    line_test = plt.plot(deltas, test_error, label="Testing Error")

    plt.legend(loc=0, borderaxespad=0.)
    plt.ylabel('mean squared error')
    plt.xlabel('1/$\lambda$')

    plt.show()