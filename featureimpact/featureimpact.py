"""
Compute the statistical impact of features given a trained estimator
"""
import numpy
import pandas


def averaged_impact(impact, normalize=True):
    """
    Computes the averaged impact across all samples for each feature

    :param impact: Array-like object of shape [n_samples, n_features].
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
        self._samples = None

    @property
    def samples(self):
        """
        The samples corresponding to the features

        :returns: pandas.DataFrame of shape [n_samples, n_features] or None
        """
        return self._samples

    @samples.setter
    def samples(self, value):
        """
        The samples corresponding to the features

        :param value: Array-like object of shape [n_samples, n_features] or None
        """
        self._samples = pandas.DataFrame(value) if value is not None else value

    def select_samples(self, X, count=100, rng=numpy.random):
        """
        Selects `count` samples randomly from each feature in X without replacement.

        :param X: Features. Array-like object of shape [n_obs, n_features]
        :param count: The number of samples to select
        :param rng: The random number generator to use
        """
        if count < 1:
            raise FeatureImpactError("count must be at least one.")
        X = pandas.DataFrame(X)
        count = count if count <= X.shape[0] else X.shape[0]
        self._samples = pandas.DataFrame()
        for col in X:
            self._samples[col] = rng.choice(X[col], count, replace=False)

    def compute_impact(self, estimator, X, method='predict'):
        """
        Computes the statistical impact of each feature based on the mean
        variation of the difference between perturbed and original predictions.
        The impact is always >= 0.

        If samples were not selected then all samples in X are used to
        generate the perturbed predictions.

        :param estimator: A trained estimator implementing the given predict `method`.
            It is assumed that the predict method does not change its input.
        :param X: Features. Array-like object of shape [n_obs, n_features]
        :param method: The predict method to call on `estimator`.

        :returns: Impact as a pandas.DataFrame of shape [n_samples, n_features]
        """
        if not hasattr(estimator, method):
            raise FeatureImpactError("estimator does not implement {}()".format(method))
        X = pandas.DataFrame(X)
        y = getattr(estimator, method)(X)
        samples = self._samples if self._samples is not None else X
        impact = pandas.DataFrame()
        for feature in X:
            X_star = pandas.DataFrame(X, copy=True)
            x_std = X[feature].std()
            imp = []
            for sample in samples[feature]:
                X_star[feature] = sample
                y_star = getattr(estimator, method)(X_star)
                imp.append(numpy.std(y - y_star) / x_std)
            impact[feature] = imp
        return impact


class FeatureImpactError(Exception):
    pass
