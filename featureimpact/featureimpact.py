"""
Compute the statistical impact of features given a scikit-learn estimator
"""
from scipy.stats.mstats import mquantiles
import numpy


def make_averaged_impact(impact, normalize=False):
    """
    Computes the averaged impact across all events for each feature

    :param impact: Array-like object of shape [n_samples, n_features].
        This should be the return value of FeatureImpact.compute_impact()
    :param normalize: Whether to normalize the averaged impacts such that
        that the impacts sum up to one

    :returns: The averaged impact as an array-like object of shape [n_features]
    """
    imparray = numpy.asarray(impact).transpose()
    average = numpy.zeros((imparray.shape[0],), dtype=float)
    for i, series in enumerate(imparray):
        average[i] = series.mean()
    if normalize:
        average /= average.sum()
    return average


class FeatureImpact(object):
    """
    Compute the statistical impact of features given a scikit-learn estimator
    """
    def __init__(self):
        self._quantiles = None

    @property
    def quantiles(self):
        """
        The quantiles corresponding to the features

        :returns: Array-like object of shape [n_features, n_quantiles]
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
        Xarray = numpy.asarray(X, dtype=float)
        self._quantiles = self._get_quantiles(Xarray, n_quantiles)

    def compute_impact(self, estimator, X, normalize=False):
        """
        Computes the statistical impact of each feature based on the mean
        variation of the difference between quantile and original predictions.
        The impact is always >= 0. The impact is reliable for regressors
        and binary classifiers.

        :param estimator: A scikit-learn estimator implementing predict()
            which must return a single value per event. It is assumed that
            predict() does not change its input.
        :param X: Features. Array-like object of shape [n_samples, n_features]
        :param normalize: Whether to normalize the impact per event such that
            the sum of all impacts per event is one

        :returns: Impact. Array-like object of shape [n_samples, n_features]
        """
        if self._quantiles is None:
            raise FeatureImpactError("make_quantiles() must be called first "
                                     "or the quantiles explicitly assigned")
        if not hasattr(estimator, 'predict'):
            raise FeatureImpactError("estimator does not implement predict()")
        X_ref = numpy.asarray(X, dtype=float)
        y_ref = numpy.asarray(estimator.predict(X_ref), dtype=float)
        functor = lambda i, j: self._get_impact(estimator, X_ref, y_ref, i, j)
        result = numpy.zeros((X_ref.shape[0], X_ref.shape[1]), dtype=float)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = functor(i, j)
        if normalize:
            result /= result.sum(axis=1)[:, numpy.newaxis]
        return result

    def _get_impact(self, est, X, y, event, feature):
        X_star = numpy.array(X[event, :])
        impact = 0.
        for quantile in self._quantiles[feature]:
            X_star[feature] = quantile
            y_star = numpy.asarray(est.predict([X_star]), dtype=float)
            impact += numpy.abs(y_star - y[event]).sum()
        return impact / (len(self._quantiles[feature]) * len(y))

    @staticmethod
    def _get_quantiles(X, n_quantiles):
        arange = numpy.arange(0.0001, 0.9999, 0.999 / (n_quantiles - 1))
        quantiles = [mquantiles(x, arange) for x in X.transpose()]
        return numpy.array(quantiles)


class FeatureImpactError(Exception):
    pass
