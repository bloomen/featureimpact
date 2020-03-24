**featureimpact** let's you compute the statistical impact of features given
a trained estimator. The computation is based on the mean variation
of the difference between perturbed and original predictions. The estimator must
predict purely numerical values. All features must also consist of purely
numerical values.

Example:
```
from featureimpact import FeatureImpact
fi = FeatureImpact()
fi.select_samples(X_train)
impact = fi.compute_impact(model, X_test)
```

Note: In order to run the examples you'll need scikit-learn
installed in addition to this package and its regular dependencies.

The impact estimation of this package follows the approach in Section 3.9.2 of
```
Blume, C., 2012: Statistical Learning To Model Stratospheric Variability. Doctoral thesis,
Institute for Meteorology, Freie Universität Berlin. https://refubium.fu-berlin.de/handle/fub188/13901
```
and extends it to compute the impact over a range of perturbations.
