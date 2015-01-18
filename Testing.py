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


        print dataset.shape
        print dataset

        target = numpy.array([2.49, 0.83, -0.25, 3.10, 0.87, 0.02, -0.12, 1.81, -0.83, 0.43])


        result = load_data.LinearRegression(dataset, target)

        expected_result = numpy.array([0.67, 1.74, 0.73])

        self.assertTrue(numpy.allclose(result, expected_result, rtol=1e-01, atol=1e-01))


suite = unittest.TestLoader().loadTestsFromTestCase(TestAssignment1)
unittest.TextTestRunner(verbosity=2).run(suite)