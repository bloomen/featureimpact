#!/usr/bin/env python
from featureimpact import FeatureImpact, averaged_impact
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy
import pandas

# Generate some data
X = []
y = []
for i in range(200):
    x1 = numpy.sin(i)
    x2 = 0.5 * numpy.sin(i - numpy.pi / 2.)
    X.append([x1, x2])
    y.append(x1 + x2)
X = numpy.array(X, dtype=float)
y = numpy.array(y, dtype=float)

X = StandardScaler().fit_transform(X)
X = pandas.DataFrame(X, columns=["x1", "x2"])

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
