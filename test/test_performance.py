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
        if len(X) > 1:
            return self._y
        else:
            return numpy.random.random()


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

    def _run(self, normalize=False):
        n_samples = 1000
        n_features = 100
        X = get_features(n_samples, n_features)
        fi = FeatureImpact()
        timer = Timer()
        fi.make_quantiles(X)
        print('')
        print("make_quantiles: {}".format(timer))
        timer.reset()
        imp = fi.compute_impact(Model(numpy.random.rand(n_samples)), X, normalize)
        print("compute_impact: {}".format(timer))
        timer.reset()
        make_averaged_impact(imp, normalize)
        print("make_averaged_impact: {}".format(timer))

    def test_with_defaults(self):
        self._run()

    def test_with_normalize(self):
        self._run(True)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
