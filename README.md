**featureimpact** let's you compute the statistical impact of features given
a trained estimator. The computation is based on the mean variation
of the difference between quantile and original predictions. The estimator must
predict purely numerical values. All features must also consist of purely
numerical values.

Example:
```python
from featureimpact import FeatureImpact
fi = FeatureImpact()
fi.make_quantiles(X_train)
impact = fi.compute_impact(model, X_test)
```

Note: In order to run the examples you'll need scikit-learn and matplotlib
installed in addition to this package and its regular dependencies.
