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

        dataset = numpy.array([[2, -10],
                                [4, 6],
                                [10, 8],
                                [-8, 7],
                                [8, -5]], dtype=float)

        result = load_data.BuildNormalizedArray(dataset)

        expected_result = numpy.array([[0.2, -1],
                                [0.4, 0.6],
                                [1, 0.8],
                                [-0.8, 0.7],
                                [0.8, -0.5]])

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


        result = load_data.BuildPolynomialArray(dataset, 2)

        expected_result = numpy.array([[0.75, 0.86, 1],
                               [0.01, 0.09, 1],
                               [0.73, -0.85, 1],
                               [0.76, 0.87, 1],
                               [0.19, -0.44, 1],
                               [0.18, -0.43, 1],
                               [1.22, -1.10, 1],
                               [0.16, 0.40, 1],
                               [0.93, -0.96, 1],
                               [0.03, 0.17, 1]])

        self.assertTrue(numpy.allclose(result, expected_result, rtol=1e-01, atol=1e-01))


suite = unittest.TestLoader().loadTestsFromTestCase(TestAssignment1)
unittest.TextTestRunner(verbosity=2).run(suite)