import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from sklearn.utils import check_array
from sklearn.utils import compute_class_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import column_or_1d
from sklearn.base import is_regressor
from sklearn.ensemble.forest import BaseForest
from sklearn.tree.tree import BaseDecisionTree


class AdaCost(AdaBoostClassifier):

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm=None,
                 class_weight='balanced',
                 random_state=None):
        """

        :rtype: object
        """
        super(AdaBoostClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.algorithm = algorithm
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))
        y = self._validate_targets(y)

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if self.algorithm is None:
            self.algorithm = "adacost"

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self

    def _boost(self, iboost, X, y, sample_weight, random_state):

        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

            # assign class weight to each sample index
            costs = np.copy(y).astype(float)
            for n in range(self.n_classes_):
                costs[y == n] = self.class_weight_[n]
            # Normalize costs
            self.costs_ = costs / costs.sum(dtype=np.float64)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            if self.algorithm == "adacost":
                costs = self.costs_
                costs[y == y_predict] = np.array(list(map(lambda x: -0.5 * x + 0.5, self.costs_[y == y_predict])))
                costs[y != y_predict] = np.array(list(map(lambda x: 0.5 * x + 0.5, self.costs_[y != y_predict])))
                # Only boost positive weights
                sample_weight *= np.exp(costs * estimator_weight * incorrect *
                                        ((sample_weight > 0) |
                                         (estimator_weight < 0)))
            elif self.algorithm == "adac1":
                sample_weight *= np.exp(self.costs_ * estimator_weight * incorrect *
                                        ((sample_weight > 0) |
                                         (estimator_weight < 0)))
            elif self.algorithm == "adac2":
                sample_weight *= self.costs_ * np.exp(estimator_weight * incorrect)
            elif self.algorithm == "adac3":
                sample_weight *= self.costs_ * np.exp(self.costs_ * estimator_weight * incorrect)
            else:
                raise ValueError("algorithm %s is not supported" % self.algorithm)

            # re-normalise sample_weight
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            print(iboost, len(np.where(incorrect==True)[0]), estimator_weight, estimator_error)

        return sample_weight, estimator_weight, estimator_error

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        self.class_weight_ = compute_class_weight(self.class_weight, cls, y_)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d"
                % len(cls))

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order='C')


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import cohen_kappa_score

    X, y = make_classification(n_samples=1000, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=3, n_clusters_per_class=1,
                               weights=[0.5, 0.4, 0.1],
                               class_sep=0.8, random_state=39)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=39)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    clf = AdaCost(random_state=39, algorithm="adacost")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    kappa = cohen_kappa_score(y_test, y_pred)
    print(kappa)
