# featureimpact

[![Actions](https://github.com/bloomen/featureimpact/actions/workflows/featureimpact-tests.yml/badge.svg?branch=master)](https://github.com/bloomen/featureimpact/actions/workflows/featureimpact-tests.yml?query=branch%3Amaster)

featureimpact let's you compute the statistical impact of features given
a trained estimator. The computation is based on the mean variation
of the difference between perturbed and original predictions. The estimator must
predict purely numerical values. All features must also consist of purely
numerical values.

Example:
```python
from featureimpact import FeatureImpact
fi = FeatureImpact()
fi.make_quantiles(X_train)
impact = fi.compute_impact(model, X_test)
```

Note: In order to run the examples you'll need scikit-learn
installed in addition to this package and its regular dependencies.

The algorithm is described here:
https://bloomen.github.io/pub/featureimpact.pdf
