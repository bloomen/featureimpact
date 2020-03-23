"""
Compute the statistical impact of features given a trained estimator
"""
from scipy.stats.mstats import mquantiles
import numpy
import pandas


def make_averaged_impact(impact, normalize=False):
    """
    Computes the averaged impact across all events for each feature

    :param impact: Array-like object of shape [n_samples, n_features].
        This should be the return value of FeatureImpact.compute_impact()
    :param normalize: Whether to normalize the averaged impacts such that
        that the impacts sum up to one

    :returns: The averaged impact as an numpy.ndarray of shape [n_features]
    """
    impact = pandas.DataFrame(impact)
    average = numpy.zeros((impact.shape[1],), dtype=float)
    for i, col in enumerate(impact):
        average[i] = impact[col].mean()
    if normalize:
        average /= average.sum()
    return average


class FeatureImpact(object):
    """
    Compute the statistical impact of features given a trained estimator
    """
    def __init__(self):
        self._quantiles = None

    @property
    def quantiles(self):
        """
        The quantiles corresponding to the features

        :returns: numpy.ndarray of shape [n_features, n_quantiles]
        """
        return self._quantiles

    @quantiles.setter
    def quantiles(self, value):
        """
        The quantiles corresponding to the features

        :param value: Array-like object of shape [n_features, n_quantiles]
        """
        self._quantiles = numpy.asarray(value, dtype=float)

    def make_quantiles(self, X, n_quantiles=10):
        """
        Generates the quantiles for each feature in X. The quantiles for one
        feature are computed such that the area between quantiles is the
        same throughout.

        :param X: Features. Array-like object of shape [n_samples, n_features]
        :param n_quantiles: The number of quantiles to compute
        """
        if n_quantiles < 3:
            raise FeatureImpactError("n_quantiles must be at least three.")
        if len(X) < 3:
            raise FeatureImpactError("X must carry at least three events")
        X = pandas.DataFrame(X)
        arange = numpy.arange(0.0001, 0.9999, 0.999 / (n_quantiles - 1))
        quantiles = [mquantiles(X[col], arange) for col in X]
        self._quantiles = numpy.array(quantiles, dtype=float)

    def compute_impact(self, estimator, X, normalize=False, method='predict'):
        """
        Computes the statistical impact of each feature based on the mean
        variation of the difference between quantile and original predictions.
        The impact is always >= 0.

        :param estimator: A trained estimator implementing the given predict `method`.
            It is assumed that the predict method does not change its input.
        :param X: Features. Array-like object of shape [n_samples, n_features]
        :param normalize: Whether to normalize the impact per event such that
            the sum of all impacts per event is one.
        :param method: The predict method to call on `estimator`.

        :returns: Impact. numpy.ndarray of shape [n_samples, n_features]
        """
        if self._quantiles is None:
            raise FeatureImpactError("make_quantiles() must be called first "
                                     "or the quantiles explicitly assigned")
        if not hasattr(estimator, method):
            raise FeatureImpactError("estimator does not implement {}()".format(method))
        X_ref = pandas.DataFrame(X)
        y_ref = getattr(estimator, method)(X_ref)
        X_star = pandas.DataFrame(numpy.zeros((1, X_ref.shape[1]), dtype=float), columns=X_ref.columns) # caching
        result = numpy.zeros((X_ref.shape[0], X_ref.shape[1]), dtype=float)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = self._get_impact(estimator, method, X_star, X_ref, y_ref, i, j)
        if normalize:
            factors = result.sum(axis=1)[:, numpy.newaxis]
            factors[factors == 0] = 1
            result /= factors
        return result

    def _get_impact(self, est, method, X_star, X, y, event, feature):
        X_star.values[0, :] = numpy.array(X.values[event, :], dtype=float)
        impact = 0.
        for quantile in self._quantiles[feature]:
            X_star.values[0, feature] = quantile
            y_star = getattr(est, method)(X_star)
            impact += numpy.abs(y_star - y[event]).sum()
        return impact / (len(self._quantiles[feature]) * len(y))


class FeatureImpactError(Exception):
    pass
