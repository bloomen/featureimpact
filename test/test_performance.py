#!/usr/bin/env python
"""
Performance tests for module featureimpact
"""
from __future__ import print_function
import unittest
import numpy
from numpy.testing import assert_array_almost_equal
from featureimpact import FeatureImpact, FeatureImpactError, \
                          make_averaged_impact
import time
numpy.random.seed(1)


class Model(object):

    def __init__(self, size):
        self._size = size

    def predict(self, X):
        return numpy.random.rand(self._size)


class Timer(object):

    def __init__(self):
        self._start = time.time()

    def get(self):
        return time.time() - self._start

    def reset(self):
        self._start = time.time()

    def __repr__(self):
        return "%d" % self.get()


def get_features(n_samples, n_features):
    X = []
    for _ in range(n_samples):
        X.append(numpy.random.rand(n_features))
    return X


def print_sec(value):
    print("[%ss]" % value, end=' ')


class Test(unittest.TestCase):

    def test_compute_impact_with_defaults(self):
        n_samples = 100
        n_features = 100
        X = get_features(n_samples, n_features)
        fi = FeatureImpact()
        fi.make_quantiles(X)
        timer = Timer()
        fi.compute_impact(Model(n_samples), X)
        print_sec(timer)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
