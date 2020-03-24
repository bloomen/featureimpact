"""
Compute the statistical impact of features given a trained estimator
"""
from scipy.stats.mstats import mquantiles
import numpy
import pandas


def averaged_impact(impact, normalize=True):
    """
    Computes the averaged impact across all quantiles for each feature

    :param impact: Array-like object of shape [n_quantiles, n_features].
        This should be the return value of FeatureImpact.compute_impact()
    :param normalize: Whether to normalize the averaged impacts such that
        that the impacts sum up to one

    :returns: The averaged impact as a pandas.Series of shape [n_features]
    """
    impact = pandas.DataFrame(impact)
    average = pandas.Series(index=impact.columns)
    for col in impact:
        average[col] = impact[col].mean()
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

        :returns: pandas.DataFrame of shape [n_quantiles, n_features] or None
        """
        return self._quantiles

    @quantiles.setter
    def quantiles(self, value):
        """
        The quantiles corresponding to the features

        :param value: Array-like object of shape [n_quantiles, n_features]
        """
        self._quantiles = pandas.DataFrame(value)

    def make_quantiles(self, X, n_quantiles=9):
        """
        Generates the quantiles for each feature in X. The quantiles for one
        feature are computed such that the area between quantiles is the
        same throughout. The default quantiles are computed at the following
        probablities: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

        :param X: Features. Array-like object of shape [n_samples, n_features]
        :param n_quantiles: The number of quantiles to compute
        """
        if n_quantiles < 1:
            raise FeatureImpactError("n_quantiles must be at least one.")
        X = pandas.DataFrame(X)
        probs = numpy.linspace(0.0, 1.0, n_quantiles + 2)[1:-1]
        self._quantiles = pandas.DataFrame()
        for col in X:
            self._quantiles[col] = mquantiles(X[col], probs)

    def compute_impact(self, estimator, X, method='predict'):
        """
        Computes the statistical impact of each feature based on the mean
        variation of the difference between perturbed and original predictions.
        The impact is always >= 0.

        :param estimator: A trained estimator implementing the given predict `method`.
            It is assumed that the predict method does not change its input.
        :param X: Features. Array-like object of shape [n_samples, n_features]
        :param method: The predict method to call on `estimator`.

        :returns: Impact as a pandas.DataFrame of shape [n_quantiles, n_features]
        """
        if self._quantiles is None:
            raise FeatureImpactError("make_quantiles() must be called first "
                                     "or the quantiles explicitly assigned")
        if not hasattr(estimator, method):
            raise FeatureImpactError("estimator does not implement {}()".format(method))
        X = pandas.DataFrame(X)
        y = getattr(estimator, method)(X)
        impact = pandas.DataFrame()
        for feature in X:
            X_star = pandas.DataFrame(X, copy=True)
            x_std = X[feature].std()
            imp = []
            for quantile in self._quantiles[feature]:
                X_star[feature] = quantile
                y_star = getattr(estimator, method)(X_star)
                imp.append(numpy.std(y - y_star) / x_std)
            impact[feature] = imp
        return impact


class FeatureImpactError(Exception):
    pass
