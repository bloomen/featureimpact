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
    average = pandas.Series(index=impact.columns, dtype=float)
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
        self._quantiles = pandas.DataFrame(value, dtype=float)

    def make_quantiles(self, X, n_quantiles=9):
        """
        Generates the quantiles for each feature in X. The quantiles for one
        feature are computed such that the area between quantiles is the
        same throughout. The default quantiles are computed at the following
        probablities: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

        The actual quantiles being used are the values that are closest to the
        computed quantiles. This ensures only values are used that are actually
        part of the features, particularly important for distributions with
        multiple peaks (e.g. categorical features).

        :param X: Features. Array-like object of shape [n_samples, n_features]
        :param n_quantiles: The number of quantiles to compute
        """
        if n_quantiles < 1:
            raise FeatureImpactError("n_quantiles must be at least one.")
        X = pandas.DataFrame(X)
        probs = numpy.linspace(0.0, 1.0, n_quantiles + 2)[1:-1]
        self._quantiles = pandas.DataFrame(dtype=float)
        for col in X:
            feature = X[col].dropna().values
            values = []
            for quantile in mquantiles(feature, probs):
                closest = numpy.abs(feature - quantile).argmin()
                values.append(feature[closest])
            self._quantiles[col] = values

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
        impact = pandas.DataFrame(dtype=float)
        for feature in X:
            orig_feat = pandas.Series(X[feature], copy=True)
            x_std = orig_feat.std(skipna=True)
            if x_std > 0.0:
                imp = []
                for quantile, count in self._quantiles[feature].value_counts().iteritems():
                    X[feature] = quantile
                    y_star = getattr(estimator, method)(X)
                    diff_std = pandas.Series(y - y_star).std(skipna=True)
                    res = diff_std / x_std if diff_std > 0.0 else 0.0
                    imp.extend([res] * count)
            else:
                imp = [0.0] * self._quantiles.shape[0]
            impact[feature] = imp
            X[feature] = orig_feat
        return impact


class FeatureImpactError(Exception):
    pass
