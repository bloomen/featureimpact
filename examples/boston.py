#!/usr/bin/env python
from featureimpact import FeatureImpact, make_averaged_impact
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
import numpy
import pylab

# Load data
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(numpy.float)

StandardScaler(copy=False).fit_transform(X)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# Training
model = LinearRegression()
model.fit(X_train, y_train)

# LinReg's coefficients
coefs = numpy.abs(model.coef_)
coefs /= coefs.sum()

# Computing the impact on the testing period
featimp = FeatureImpact()
featimp.make_quantiles(X_train)
impact = featimp.compute_impact(model, X_test)

# Comparing averaged impact to LinReg's coefficients
ave_imp = make_averaged_impact(impact, normalize=True)
for i, imp in enumerate(ave_imp):
    print i, imp, coefs[i]

# Plotting
pylab.figure()
pylab.xlabel('Events')
pylab.ylabel('Impact [Target Unit] ')
labels = []
for i, series in enumerate(impact.transpose()):
    width = i % 2 + 1
    pylab.plot(series, linewidth=width)
    labels.append("%d" % i)
pylab.legend(labels, title="Features", bbox_to_anchor=(1.01, 1),
             loc=2, borderaxespad=0.)

try:
    pylab.show()
except KeyboardInterrupt:
    pass
