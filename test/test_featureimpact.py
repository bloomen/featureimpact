#!/usr/bin/env python
"""
Unit tests for module featureimpact
"""
import unittest
import numpy
import pandas
from numpy.testing import assert_array_almost_equal
from featureimpact import FeatureImpact, FeatureImpactError, \
                          averaged_impact
numpy.random.seed(42)


class Test(unittest.TestCase):

    def test_samples_property(self):
        fi = FeatureImpact()
        self.assertEqual(None, fi.samples)
        fi.samples = []
        assert_array_almost_equal(pandas.DataFrame([]), fi.samples, 6)
        fi.samples = [[1, 2], [3, 4]]
        assert_array_almost_equal(pandas.DataFrame([[1, 2], [3, 4]]), fi.samples, 6)

    def test_select_samples(self):
        fi = FeatureImpact()
        self.assertRaises(FeatureImpactError, fi.select_samples,
                          X=[], count=0)
        X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        fi.select_samples(X, count=3)
        exp = [[1, 5, 3],
               [4, 8, 6],
               [7, 2, 9]]
        assert_array_almost_equal(pandas.DataFrame(exp), fi.samples, 6)

    def test_compute_impact_zero_prediction(self):
        class M:
            def predict(self, _):
                return numpy.array([0., 0., 0.])
        fi = FeatureImpact()
        X = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        impact = fi.compute_impact(M(), X)
        exp = numpy.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype=float)
        assert_array_almost_equal(exp, impact, 6)

    def test_compute_impact_real_prediction(self):
        class M:
            def __init__(self):
                self._i = 0
            def predict(self, X):
                if self._i >= X.shape[1]:
                    self._i = 0
                y = numpy.array(X.iloc[:, self._i])
                self._i += 1
                return y
        fi = FeatureImpact()
        X = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        impact = fi.compute_impact(M(), X)
        exp = numpy.array([[0, 0.816497, 0],
                           [0, 0, 0.816497],
                           [0.816497, 0, 0]], dtype=float)
        assert_array_almost_equal(exp, impact, 6)

    def test_averaged_impact(self):
        impact = []
        self.assertTrue((numpy.array([]) == averaged_impact(impact)).all())
        impact = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        impave = averaged_impact(impact, normalize=False)
        self.assertTrue((numpy.array([4, 5, 6]) == impave).all())
        impavenorm = averaged_impact(impact)
        exp = numpy.array([0.26666667, 1./3., 0.4])
        assert_array_almost_equal(exp, impavenorm, 6)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
