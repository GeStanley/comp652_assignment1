__author__ = 'geoffrey'

import unittest
import numpy
import load_data

class TestAssignment1(unittest.TestCase):

    def test_linear_regression(self):

        dataset = numpy.array([[0.75, 0.86, 1],
                               [0.01, 0.09, 1],
                               [0.73, -0.85, 1],
                               [0.76, 0.87, 1],
                               [0.19, -0.44, 1],
                               [0.18, -0.43, 1],
                               [1.22, -1.10, 1],
                               [0.16, 0.40, 1],
                               [0.93, -0.96, 1],
                               [0.03, 0.17, 1]])


        target = numpy.array([2.49, 0.83, -0.25, 3.10, 0.87, 0.02, -0.12, 1.81, -0.83, 0.43])


        result = load_data.LinearRegression(dataset, target)

        expected_result = numpy.array([0.67, 1.74, 0.73])

        self.assertTrue(numpy.allclose(result, expected_result, rtol=1e-01, atol=1e-01))

    def test_build_normalized_array(self):

        dataset = numpy.array([[2, -10, 1],
                                [4, 6, 1],
                                [10, 8, 1],
                                [-8, 7, 1],
                                [8, -5, 1]], dtype=float)

        result = load_data.BuildNormalizedArray(dataset)

        expected_result = numpy.array([[0.2, -1, 1],
                                [0.4, 0.6, 1],
                                [1, 0.8, 1],
                                [-0.8, 0.7, 1],
                                [0.8, -0.5, 1]])

        self.assertTrue(numpy.allclose(result, expected_result, rtol=1e-01, atol=1e-01))

    def test_build_polynomial_array(self):

        dataset = numpy.array([[0.86, 1],
                               [0.09, 1],
                               [-0.85, 1],
                               [0.87, 1],
                               [-0.44, 1],
                               [-0.43, 1],
                               [-1.10, 1],
                               [0.40, 1],
                               [-0.96, 1],
                               [0.17, 1]])


        result = load_data.PolyRegress(dataset, 2)

        expected_result = numpy.array([[0.7396, 0.86, 1],
                               [0.0081, 0.09, 1],
                               [0.7225, -0.85, 1],
                               [0.7569, 0.87, 1],
                               [0.1936, -0.44, 1],
                               [0.1849, -0.43, 1],
                               [1.21, -1.10, 1],
                               [0.16, 0.40, 1],
                               [0.9216, -0.96, 1],
                               [0.0289, 0.17, 1]])


        self.assertTrue(numpy.allclose(result, expected_result, rtol=1e-03, atol=1e-03))

        dataset = numpy.array([[3, 0.86, 1],
                       [2, 0.09, 1],
                       [4, -0.85, 1]])


        result = load_data.PolyRegress(dataset, 2)

        expected_result = numpy.array([[9, 0.75, 3, 0.86, 1],
                               [4, 0.01, 2, 0.09, 1],
                               [16, 0.73, 4, -0.85, 1]])


        self.assertTrue(numpy.allclose(result, expected_result, rtol=1e-01, atol=1e-01))

    def test_calculate_error(self):
        features = numpy.array([[1, 2, 3]])
        targets = numpy.array([[22]])
        weights = numpy.array([[2, 3, 4]])

        error = load_data.CalculateError(features, targets, weights)

        expected_error = 4

        self.assertEqual(error, expected_error)

        features = numpy.array([[1, 2, 3], [2, 2, 2]])
        targets = numpy.array([[22], [20]])
        weights = numpy.array([[2, 3, 4]])

        error = load_data.CalculateError(features, targets, weights)

        expected_error = 8

        self.assertEqual(error, expected_error)

suite = unittest.TestLoader().loadTestsFromTestCase(TestAssignment1)
unittest.TextTestRunner(verbosity=2).run(suite)