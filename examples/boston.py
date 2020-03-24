#!/usr/bin/env python
from featureimpact import FeatureImpact, averaged_impact
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy

# Load data
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(numpy.float)

# Training
linreg = LinearRegression()
linreg.fit(X, y)

forest = RandomForestRegressor(n_estimators=100)
forest.fit(X, y)

svr = SVR(gamma='scale')
svr.fit(X, y)

# Get linreg and forest coefficients
coefs_linreg = numpy.abs(linreg.coef_)
coefs_forest = forest.feature_importances_

# Computing the impact
fi = FeatureImpact()
fi.make_quantiles(X)
impact_linreg = fi.compute_impact(linreg, X)
impact_forest = averaged_impact(fi.compute_impact(forest, X))
impact_svr = averaged_impact(fi.compute_impact(svr, X))

print("Impact vs LinearRegression coeffs:")
for i, imp in enumerate(impact_linreg):
    print(i, impact_linreg[imp].mean(), coefs_linreg[i])

print("Impact vs RandomForestRegressor coeffs:")
for i, imp in enumerate(impact_forest):
    print(i, imp, coefs_forest[i])

print("Impact on SVR:")
for i, imp in enumerate(impact_svr):
    print(i, imp)
