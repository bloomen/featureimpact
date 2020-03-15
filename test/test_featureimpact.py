#!/usr/bin/env python
"""
Unit tests for module featureimpact
"""
import unittest
import numpy
from numpy.testing import assert_array_almost_equal
from featureimpact import FeatureImpact, FeatureImpactError, \
                          make_averaged_impact


class Test(unittest.TestCase):

    def test_quantiles_property(self):
        fi = FeatureImpact()
        self.assertEqual(None, fi.quantiles)
        fi.quantiles = []
        self.assertTrue((numpy.array([]) == fi.quantiles).all())
        fi.quantiles = [1, 2, 3]
        self.assertTrue((numpy.array([1, 2, 3]) == fi.quantiles).all())

    def test_make_quantiles(self):
        fi = FeatureImpact()
        self.assertRaises(FeatureImpactError, fi.make_quantiles,
                          X=[], n_quantiles=2)
        self.assertRaises(FeatureImpactError, fi.make_quantiles,
                          X=[[1, 2]], n_quantiles=3)
        X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        fi.make_quantiles(X, n_quantiles=3)
        quants = fi.quantiles
        exp = numpy.transpose(X)
        assert_array_almost_equal(exp, quants, 2)

    def test_compute_impact_zero_prediction(self):
        class M:
            def predict(self, X):
                if len(X) == 3:
                    return [0., 0., 0.]
                else:
                    return 0.
        fi = FeatureImpact()
        X = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        fi.quantiles = X.transpose()
        impact = fi.compute_impact(M(), X, normalize=False)
        exp = numpy.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype=float)
        self.assertTrue((exp == impact).all())

    def test_compute_impact_real_prediction(self):
        class M:
            def predict(self, X):
                if len(X) == 3:
                    return [1., 2., 3.]
                else:
                    return 1.
        fi = FeatureImpact()
        X = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        fi.quantiles = X.transpose()
        impact = fi.compute_impact(M(), X, normalize=False)
        exp = numpy.array([[0., 0., 0.],
                           [1./3., 1./3., 1./3.],
                           [2./3., 2./3., 2./3.]], dtype=float)
        assert_array_almost_equal(exp, impact, 6)

    def test_compute_impact_real_prediction_normalize(self):
        class M:
            def predict(self, X):
                if len(X) == 3:
                    return [1., 2., 3.]
                else:
                    return 1.
        fi = FeatureImpact()
        X = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        fi.quantiles = X.transpose()
        impact = fi.compute_impact(M(), X, normalize=True)
        exp = numpy.array([[0., 0., 0.],
                           [1./3., 1./3., 1./3.],
                           [1./3., 1./3., 1./3.]], dtype=float)
        assert_array_almost_equal(exp, impact, 6)

    def test_compute_impact_real_prediction_with_single_output(self):
        class M:
            def predict(self, X):
                if len(X) == 3:
                    return [1., 2., 3.]
                else:
                    return 1
        fi = FeatureImpact()
        X = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        fi.quantiles = X.transpose()
        impact = fi.compute_impact(M(), X, normalize=False)
        exp = numpy.array([[0., 0., 0.],
                           [1./3., 1./3., 1./3.],
                           [2./3., 2./3., 2./3.]], dtype=float)
        assert_array_almost_equal(exp, impact, 6)

    def test__get_impact(self):
        class M:
            def predict(self, X):
                assert(len(X) == 1)
                return 0.
        fi = FeatureImpact()
        X = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        fi.quantiles = X.transpose()
        y = numpy.array([1, 2, 3], dtype=float)
        X_star = numpy.zeros((X.shape[1],), dtype=float)
        impact = fi._get_impact(M(), 'predict', X_star, X, y, event=1, feature=0)
        self.assertEqual(2./3., impact)

    def test_make_averaged_impact(self):
        impact = []
        self.assertTrue((numpy.array([]) == make_averaged_impact(impact)).all())
        impact = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        impave = make_averaged_impact(impact)
        self.assertTrue((numpy.array([4, 5, 6]) == impave).all())
        impavenorm = make_averaged_impact(impact, normalize=True)
        exp = numpy.array([0.26666667, 1./3., 0.4])
        assert_array_almost_equal(exp, impavenorm, 6)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
