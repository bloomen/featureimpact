#!/usr/bin/env python
"""
Performance tests for module featureimpact
"""
from __future__ import print_function
import unittest
import numpy
from featureimpact import FeatureImpact, make_averaged_impact
import time
numpy.random.seed(1)


class Model(object):

    def __init__(self, y):
        self._y = y

    def predict(self, X):
        return self._y


class Timer(object):

    def __init__(self):
        self._start = time.time()

    def get(self):
        return time.time() - self._start

    def reset(self):
        self._start = time.time()

    def out(self):
        print("[%ss]" % self, end=' ')

    def __repr__(self):
        return "%d" % self.get()


def get_features(n_samples, n_features):
    X = []
    for _ in range(n_samples):
        X.append(numpy.random.rand(n_features))
    return numpy.array(X, dtype=float)


class Test(unittest.TestCase):

    def test_compute_impact_with_defaults(self):
        n_samples = 1000
        n_features = 100
        X = get_features(n_samples, n_features)
        fi = FeatureImpact()
        fi.make_quantiles(X)
        timer = Timer()
        imp = fi.compute_impact(Model(numpy.random.rand(n_samples)), X)
        make_averaged_impact(imp)
        timer.out()

    def test_compute_impact_with_normalize(self):
        n_samples = 1000
        n_features = 100
        X = get_features(n_samples, n_features)
        fi = FeatureImpact()
        fi.make_quantiles(X)
        timer = Timer()
        imp = fi.compute_impact(Model(numpy.random.rand(n_samples)), X, True)
        make_averaged_impact(imp, True)
        timer.out()


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
