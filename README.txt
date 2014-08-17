This package let's you compute the statistical impact of features given
a scikit-learn estimator. The computation is based on the mean variation 
of the difference between quantile and original predictions. The impact
is reliable for regressors and binary classifiers.

Currently, all features must consist of pure-numerical, non-categorical values.

featureimpact is being developed by Christian Blume. Contact Christian at
chr.blume@gmail.com for any questions or comments.

Note: In order to run the examples you'll need scikit-learn and matplotlib
installed in addition to this package and its regular dependencies.
