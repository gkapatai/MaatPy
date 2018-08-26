import numbers
import numpy as np
from collections import Counter
from sklearn.base import (clone,
                          is_regressor)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.forest import BaseForest
from sklearn.preprocessing import normalize
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import (check_random_state,
                           check_X_y,
                           check_array,
                           safe_indexing)
from imblearn.utils import check_neighbors_object
from imblearn.over_sampling import SMOTE

__all__ = ['SMOTEBoost']

MAX_INT = np.iinfo(np.int32).max


class SMOTEBoost(AdaBoostClassifier):
    """
    Implementation of SMOTEBoost.

    SMOTEBoost introduces data sampling into the AdaBoost algorithm by oversampling the minority class
    using SMOTE on each boosting iteration [1]. This implementation inherits methods from the scikit-learn
    AdaBoostClassifier class and some code from the SMOTEBoost class from
    https://github.com/dialnd/imbalanced-algorithms/blob/master/smote.py.
    Adjusted to work with the imblearn.over-sampling.SMOTE class

    References
    ----------
    .. [1] N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer.
           "SMOTEBoost: Improving Prediction of the Minority Class in Boosting."
           European Conference on Principles of Data Mining and Knowledge Discovery (PKDD), 2003.
    """

    def __init__(self,
                 k_neighbors=5,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 ratio="auto",
                 algorithm='SAMME.R',
                 random_state=None,
                 n_jobs=1):
        """

        :param k_neighbors: int, optional (default=5)
               Number of nearest neighbors.
        :param base_estimator:object, optional (default=DecisionTreeClassifier)
               The base estimator from which the boosted ensemble is built.
               Support for sample weighting is required, as well as proper 'classes_' and 'n_classes_' attributes.
        :param n_estimators: int, optional (default=50)
               The maximum number of estimators at which boosting is terminated.
               In case of perfect fit, the learning procedure is stopped early.
        :param learning_rate: float, optional (default=1.)
               Learning rate shrinks the contribution of each classifier by "learning_rate".
               There is a trade-off between "learning_rate" and "n_estimators".
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
        :param algorithm: {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
               If 'SAMME.R' then use the SAMME.R real boosting algorithm. The "base_estimator" must support
               calculation of class probabilities.
               If 'SAMME' then use the SAMME discrete boosting algorithm.
               The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with
               fewer boosting iterations.
        :param random_state: int, RandomState instance or None, optional (default=None)
               If int, random_state is the seed used by the random number generator; If RandomState instance,
               random_state is the random number generator; If None, the random number generator is the RandomState
               instance used by 'np.random'.
        :param n_jobs: int, optional (default=1)
               The number of jobs to run in parallel for both `fit` and `predict`.
               If -1, then the number of jobs is set to the number of cores.
        """
        super(AdaBoostClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.algorithm = algorithm
        self.k_neighbors = k_neighbors
        self.ratio = ratio
        self.n_jobs=n_jobs

    def _validate_estimator(self, default=AdaBoostClassifier()):
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

        if isinstance(self.ratio, dict) and self.ratio != {}:
            raise ValueError("'dict' type cannot be accepted for ratio in this class; "
                             "use alternative options")

        self.nn_k_ = check_neighbors_object('k_neighbors',
                                            self.k_neighbors,
                                            additional_neighbor=1)
        self.nn_k_.set_params(**{'n_jobs': self.n_jobs})

        self.smote = SMOTE(ratio=self.ratio, k_neighbors=self.k_neighbors,
                           random_state=self.random_state)

        self.base_estimator_ = base_estimator

    def fit(self, X, y, sample_weight=None):
        """
        Find the classes statistics before performing sampling

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the data which have to be sampled.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :return: object; Return self
        """
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

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

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # SMOTE step

            target_stats = Counter(y)
            min_class = min(target_stats, key=target_stats.get)
            n_sample_majority = max(target_stats.values())
            n_samples = n_sample_majority - target_stats[min_class]
            target_class_indices = np.flatnonzero(y == min_class)
            X_class = safe_indexing(X, target_class_indices)
            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]

            X_new, y_new = self.smote._make_samples(X_class, min_class, X_class,
                                              nns, n_samples, 1.0)

            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.empty(X_new.shape[0], dtype=np.float64)
            sample_weight_syn[:] = 1. / X.shape[0]

            # Combine the original and synthetic samples.
            X = np.vstack((X, X_new))
            y = np.append(y, y_new)

            # Combine the weights.
            sample_weight = \
                np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))

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
