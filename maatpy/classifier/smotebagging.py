import numbers
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from maatpy.samplers.oversampling import SMOTE
from maatpy.pipeline import Pipeline


class SMOTEBagging(BaggingClassifier):
    """
    Implementation of SMOTEBagging.

    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 k_neighbors=5,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 ratio='auto',
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        """

        :param base_estimator: object, optional (default=DecisionTreeClassifier)
               The base estimator from which the boosted ensemble is built.
               Support for sample weighting is required, as well as proper 'classes_' and 'n_classes_' attributes.
        :param n_estimators: int, optional (default=50)
               The maximum number of estimators at which boosting is terminated.
               In case of perfect fit, the learning procedure is stopped early.
        :param k_neighbors: int, optional (default=5)
               Number of nearest neighbors.
        :param max_samples: int or float, optional (default=1.0)
               The number of samples to draw from X to train each base estimator.
               - If int, then draw `max_samples` samples.
               - If float, then draw `max_samples * X.shape[0]` samples.
        :param max_features: int or float, optional (default=1.0)
               The number of features to draw from X to train each base estimator.
               - If int, then draw `max_features` features.
               - If float, then draw `max_features * X.shape[1]` features.
        :param bootstrap: boolean, optional (default=True)
               Whether samples are drawn with replacement.
        :param bootstrap_features: boolean, optional (default=False)
               Whether features are drawn with replacement.
        :param oob_score: bool, optional (default=False)
               Whether to use out-of-bag samples to estimate the generalization error
        :param warm_start: bool, optional (default=False)
               When set to True, reuse the solution of the previous call to fit and add more estimators to
               the ensemble, otherwise, just fit a whole new ensemble.
        :param ratio: str, dict, or callable, optional (default='auto')
               Ratio to use for resampling the data set.
               - If "str", has to be one of: (i) 'minority': resample the minority class;
                 (ii) 'majority': resample the majority class,
                 (iii) 'not minority': resample all classes apart of the minority class,
                 (iv) 'all': resample all classes, and (v) 'auto': correspond to 'all' with for over-sampling
                 methods and 'not_minority' for under-sampling methods. The classes targeted will be over-sampled or
                 under-sampled to achieve an equal number of sample with the majority or minority class.
               - If "dict`", the keys correspond to the targeted classes. The values correspond to the desired number
                 of samples.
               - If callable, function taking "y" and returns a "dict". The keys correspond to the targeted classes.
                 The values correspond to the desired number of samples.
        :param n_jobs: int, optional (default=1)
               The number of jobs to run in parallel for both `fit` and `predict`.
               If -1, then the number of jobs is set to the number of cores.
        :param random_state: int, RandomState instance or None, optional (default=None)
               If int, random_state is the seed used by the random number generator; If RandomState instance,
               random_state is the random number generator; If None, the random number generator is the RandomState
               instance used by 'np.random'.
        :param verbose: int, optional (default=0)
               Controls the verbosity of the building process.
        """
        super(SMOTEBagging, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.k_neighbors = k_neighbors
        self.ratio = ratio

    def _validate_estimator(self, default=BaggingClassifier()):
        """
        Check the estimator and the n_estimator attribute, set the
        'base_estimator_' attribute.

        :param default: classifier object used if base_estimator=None
        :return:
        """
        """"""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError("n_estimators must be an integer, "
                             "got {0}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            base_estimator = clone(self.base_estimator)
        else:
            base_estimator = clone(default)

        if isinstance(self.ratio, dict):
            raise ValueError("'dict' type cannot be accepted for ratio in this class; "
                             "use alternative options")

        self.base_estimator_ = Pipeline(
            [('sampler', SMOTE(ratio=self.ratio, k_neighbors=self.k_neighbors,
                               random_state=self.random_state)),
             ('classifier', base_estimator)])

    def fit(self, X, y):
        """
        Find the classes statistics before performing sampling

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the data which have to be sampled.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :return: object; Return self
        """

        return self._fit(X, y, self.max_samples, sample_weight=None)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import cohen_kappa_score

    X, y = make_classification(n_samples=1000, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=2, n_clusters_per_class=1,
                               weights=[0.90, 0.1],
                               class_sep=0.8, random_state=39)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=39)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    clf = SMOTEBagging(random_state=39)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    kappa = cohen_kappa_score(y_test, y_pred)
    print(kappa)