#!/usr/bin/env python
"""
Performance tests for module featureimpact
"""
from __future__ import print_function
import unittest
import numpy
from featureimpact import FeatureImpact
import time
numpy.random.seed(1)


class Model(object):

    def __init__(self, y):
        self._y = y

    def predict(self, _):
        return self._y


class Timer(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self._start = time.time()

    def __repr__(self):
        return "{}s".format(round(time.time() - self._start, 2))


def get_features(n_samples, n_features):
    X = []
    for _ in range(n_samples):
        X.append(numpy.random.rand(n_features))
    return numpy.array(X, dtype=float)


class Test(unittest.TestCase):

    def test(self):
        n_samples = 100000
        n_features = 100
        X = get_features(n_samples, n_features)
        y = numpy.random.rand(n_samples)
        fi = FeatureImpact()
        timer = Timer()
        fi.make_quantiles(X)
        print('')
        print("make_quantiles: {}".format(timer))
        timer.reset()
        imp = fi.compute_impact(Model(y), X)
        print("compute_impact: {}".format(timer))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
