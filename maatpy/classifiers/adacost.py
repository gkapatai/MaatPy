# coding: utf-8
import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.forest import BaseForest
from sklearn.tree.tree import BaseDecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import (check_random_state,
                           check_X_y,
                           check_array,
                           compute_class_weight,
                           column_or_1d)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

__all__ = ['AdaCost']


class AdaCost(AdaBoostClassifier):
    """
    Implementation of the cost sensitive variants of AdaBoost; Adacost and AdaC1-3

    Reference: Nikolaou et al Mach Learn (2016) 104:359–384.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm=None,
                 class_weight='balanced',
                 random_state=None):
        """

        :param base_estimator: object, optional (default=DecisionTreeClassifier)
               The base estimator from which the boosted ensemble is built.
               Support for sample weighting is required, as well as proper 'classes_' and 'n_classes_' attributes.
        :param n_estimators: int, optional (default=50)
               The maximum number of estimators at which boosting is terminated.
               In case of perfect fit, the learning procedure is stopped early.
        :param learning_rate: float, optional (default=1.)
               Learning rate shrinks the contribution of each classifier by "learning_rate".
               There is a trade-off between "learning_rate" and "n_estimators".
        :param algorithm: algorithm: {'adacost', 'adac1', 'adac2', 'adac3'}, optional (default='adacost')
        :param class_weight: dict, list of dicts, “balanced” or None, default=None
               Weights associated with classes in the form {class_label: weight}. If not given, all classes are
               supposed to have weight one. For multi-output problems, a list of dicts can be provided in the
               same order as the columns of y.
        :param random_state: int, RandomState instance or None, optional (default=None)
               If int, random_state is the seed used by the random number generator; If RandomState instance,
               random_state is the random number generator; If None, the random number generator is the RandomState
               instance used by 'np.random'.
        """
        super(AdaBoostClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.algorithm = algorithm
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        """
        Build a boosted classifier from the training set (X, y).

        :param X:{array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the training data.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :param sample_weight: array-like of shape = [n_samples], optional
               Sample weights. If None, the sample weights are initialized to the class weights
        :return: object; Return self
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
            # Initialize weights to class weights
            # assign class weight to each sample index
            sample_weight = np.copy(y).astype(float)
            for n in range(len(self.classes)):
                sample_weight[y == n] = self.class_weight_[n]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        # Normalize existing weights
        #sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

        # Check that the sample weights sum is positive
        if sample_weight.sum() <= 0:
            raise ValueError(
                "Attempting to fit with a non-positive weighted number of samples.")

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

        return self

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """
        Implement a single boost.

        Perform a single boost according to the algorithm selected and return the updated
        sample weights.

        :param iboost: int
               The index of the current boost iteration
        :param X:{array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the training data.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :param sample_weight: array-like of shape = [n_samples], optional
               Sample weights. If None, the sample weights are initialized to the class weights
        :param random_state: int, RandomState instance or None, optional (default=None)
               If int, random_state is the seed used by the random number generator; If RandomState instance,
               random_state is the random number generator; If None, the random number generator is the RandomState
               instance used by 'np.random'.
        :return: sample_weight {array-like of shape = [n_samples]}, estimator_weight {float}, estimator_error {float}
                Returns updates values for sample weights, estimator weight and estimator error
        """

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
            self.cost_ = costs
        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        if self.algorithm == "adacost":
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
        elif self.algorithm in ['adac1', 'adac2']:
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight*self.cost_, axis=0))
        elif self.algorithm == "adac3":
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight*np.power(self.cost_, 2), axis=0))
        else:
            raise ValueError("Algorithms 'adacost', 'adac1', 'adac2' and 'adac3' are accepted;"\
                             " got {0}".format(self.algorithm))
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

        # Boost weight based on algorithm (Nikolaou et al Mach Learn (2016) 104:359–384)
        if self.algorithm == "adacost" or self.algorithm == "adac2":
            estimator_weight = self.learning_rate * 0.5 * (
                np.log((1. - estimator_error) / estimator_error))
        elif self.algorithm == "adac1":
            estimator_weight = self.learning_rate * 0.5 * (
                np.log((1 + (1. - estimator_error) - estimator_error) /
                       (1 - (1. - estimator_error) + estimator_error)))
        elif self.algorithm == "adac3":
            estimator_weight = self.learning_rate * 0.5 * (
                np.log((np.sum(sample_weight*self.cost_) + (1 - estimator_error) - estimator_error) /
                       (np.sum(sample_weight*self.cost_) - (1. - estimator_error) + estimator_error)))
        # Only boost the weights if it will fit again
        if iboost < self.n_estimators - 1:
            if self.algorithm == "adacost":
                beta = np.copy(self.cost_).astype(float)
                beta[y == y_predict] = np.array(list(map(lambda x: -0.5 * x + 0.5, self.cost_[y == y_predict])))
                beta[y != y_predict] = np.array(list(map(lambda x: 0.5 * x + 0.5, self.cost_[y != y_predict])))
                # Only boost positive weights
                sample_weight *= np.exp(beta * estimator_weight * incorrect *
                                        ((sample_weight > 0) | (estimator_weight < 0)))
            elif self.algorithm == "adac1":
                sample_weight *= np.exp(self.cost_ * estimator_weight * incorrect *
                                        ((sample_weight > 0) | (estimator_weight < 0)))
            elif self.algorithm == "adac2":
                sample_weight *= self.cost_ * np.exp(estimator_weight * incorrect *
                                                     ((sample_weight > 0) | (estimator_weight < 0)))
            elif self.algorithm == "adac3":
                sample_weight *= self.cost_ * np.exp(self.cost_ * estimator_weight * incorrect *
                                                     ((sample_weight > 0) | (estimator_weight < 0)))
            else:
                raise ValueError("algorithm %s is not supported" % self.algorithm)

        return sample_weight, estimator_weight, estimator_error

    def _validate_estimator(self):
        """
        Check the estimator and set the base_estimator_ attribute.
        """
        super(AdaBoostClassifier, self)._validate_estimator(
            default=DecisionTreeClassifier(max_depth=1, class_weight=self.class_weight_))

    def _validate_targets(self, y):
        """
        Validation of y and class_weight.

        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :return: validated y {array-like, shape (n_samples,)}
        """
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        class_weight_ = compute_class_weight(self.class_weight, cls, y_)
        self.class_weight_ = {i: class_weight_[i] for i in range(len(class_weight_))}
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d"
                % len(cls))

        self.classes = cls

        return np.asarray(y, dtype=np.float64, order='C')

    def predict(self, X):
        """
        Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the input samples.
        :return: y_predicted; predicted labels for X
        """
        pred = self.decision_function(X)
        # >>> removed binary special case
        # if self.n_classes_ == 2:
        #    return self.classes_.take(pred == 0, axis=0)
        # <<<

        return self.classes.take(np.argmax(pred, axis=1), axis=0)

    def decision_function(self, X):
        """
        Compute the decision function of X
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the input samples.
        :return: score : array, shape = [n_samples, k]
                 The decision function of the input samples. The order of outputs is the same of
                 that of the 'classes_' attribute.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        classes = self.classes_[:, np.newaxis]
        pred = sum((estimator.predict(X) == classes).T * w
                   for estimator, w in zip(self.estimators_,
                                           self.estimator_weights_))

        pred /= self.estimator_weights_.sum()
        # >>> removed binary special case
        # if n_classes == 2:
        #    pred[:, 0] *= -1
        #    return pred.sum(axis=1)
        # <<<
        return pred
