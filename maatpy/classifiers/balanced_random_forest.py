# coding: utf-8
import numbers
import warnings

import numpy as np
from joblib import (Parallel,
                    delayed)
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.forest import _generate_sample_indices
from sklearn.exceptions import DataConversionWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import (check_random_state,
                           check_array,
                           compute_sample_weight)

from imblearn.under_sampling import RandomUnderSampler

MAX_INT = np.iinfo(np.int32).max


class BalancedRandomForestClassifier(RandomForestClassifier):
    """
    Implementation of the Balanced Random Forest

    Reference: Chen et al, “Using Random Forest to Learn Imbalanced Data,” UC Berkeley (tech-report), 2004.
    """

    def __init__(self,
                 n_estimators=10,
                 bootstrap=True,
                 oob_score=False,
                 max_depth=None,
                 criterion="gini",
                 max_features="auto",
                 ratio="auto",
                 replacement=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        """

        :param n_estimators: int, optional (default=50)
               The maximum number of estimators at which boosting is terminated.
               In case of perfect fit, the learning procedure is stopped early.
        :param bootstrap: boolean, optional (default=True)
               Whether samples are drawn with replacement.
        :param oob_score: bool, optional (default=False)
               Whether to use out-of-bag samples to estimate the generalization error
        :param max_depth : integer or None, optional (default=None)
               The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure
               or until all leaves contain less than min_samples_split samples.
        :param criterion : string, optional (default="gini")
               The function to measure the quality of a split. Supported criteria are "gini" for the 
               Gini impurity and "entropy" for the information gain.
               Note: this parameter is tree-specific.
        :param max_features : int, float, string or None, optional (default="auto")
               The number of features to consider when looking for the best split:
               - If int, then consider `max_features` features at each split.
               - If float, then `max_features` is a percentage and `int(max_features * n_features)`
               features are considered at each split.
               - If "auto", then `max_features=sqrt(n_features)`.
               - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
               - If "log2", then `max_features=log2(n_features)`.
               - If None, then `max_features=n_features`.
               Note: the search for a split does not stop until at least one valid partition of the node
               samples is found, even if it requires to effectively inspect more than ``max_features`` 
               features.
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
        :param replacement: boolean, optional (default=False)
               Whether the sample is with or without replacement
        :param n_jobs: int, optional (default=1)
               The number of jobs to run in parallel for both `fit` and `predict`.
               If -1, then the number of jobs is set to the number of cores.
        :param random_state: int, RandomState instance or None, optional (default=None)
               If int, random_state is the seed used by the random number generator; If RandomState instance,
               random_state is the random number generator; If None, the random number generator is the RandomState
               instance used by 'np.random'.
        :param verbose: int, optional (default=0)
               Controls the verbosity of the building process.
        :param warm_start: bool, optional (default=False)
               When set to True, reuse the solution of the previous call to fit and add more estimators to
               the ensemble, otherwise, just fit a whole new ensemble.
        :param class_weight: dict, list of dicts, "balanced", "balanced_subsample" or None, optional (default=None)
               Weights associated with classes in the form ``{class_label: weight}``.
               If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts
               can be provided in the same order as the columns of y.
        """
        super(BalancedRandomForestClassifier, self).__init__(
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            oob_score=oob_score,
            max_depth=max_depth,
            criterion=criterion,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.ratio = ratio
        self.replacement = replacement
        self.base_estimator = DecisionTreeClassifier(random_state=random_state)

    def _validate_estimator(self, default=DecisionTreeClassifier()):
        """
        Check the estimator and the n_estimator attribute, set the 'base_estimator_' attribute.

        :param default: classifier object used if base_estimator=None
        :return:
        """

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
        if isinstance(self.class_weight, int):
            raise ValueError("'class_weight' can accept dict, list of dicts, 'balanced', 'balanced_subsample' or None;"\
                             " got {0} instead".format(self.class_weight))

        self.rus = RandomUnderSampler(ratio=self.ratio, replacement=self.replacement,
                                      return_indices=True, random_state=self.random_state)
        self.base_estimator_ = base_estimator

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Copied from sklearn.ensemble.BaseForest.fit() to use the edited _parallel_build_function

        :param X:{array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the training data.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :param sample_weight: array-like of shape = [n_samples], optional
               Sample weights. If None, the sample weights are equally weighted. Splits that would create child
               nodes with net zero or negative weight are ignored while searching for a split in each node. In
               the case of classification, splits are also ignored if they would result in any single class carrying
               a negative weight in either child node.
        :return: object; Return self
        """
        # Check parameters
        self._validate_estimator()
        # Validate or convert input data
        X = check_array(X, accept_sparse="csc", dtype=np.float32)
        y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()
        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != np.float64 or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.float64)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warnings.warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = []
            for i in range(n_more_estimators):
                tree = self._make_estimator(append=False,
                                            random_state=random_state)
                trees.append(tree)

            # Parallel loop: we use the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading always more efficient than multiprocessing in
            # that case.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             backend="threading")(
                delayed(self._parallel_build_trees)(
                    t, self, X, y, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def _parallel_build_trees(self, tree, forest, X, y, sample_weight, tree_idx, n_trees,
                              verbose=0, class_weight=None):
        """
        Private function used to fit a single tree in parallel.

        Copied from sklearn.ensemble.forest and converted to a class function to perform undersampling prior to
        fitting the single tree

        :param tree: base_estimator {default=DecisionTreeClassifier()}
        :param forest: self {BalancedRandomForestClassifier object}
        :param X:{array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the training data.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :param sample_weight: array-like of shape = [n_samples], optional
               Sample weights.
        :param tree_idx: index for specific tree
        :param n_trees: total number of trees
        :param verbose: int, optional (default=0)
               Controls the verbosity of the building process.
        :param class_weight: dict, list of dicts, "balanced", "balanced_subsample" or None, optional (default=None)
               Weights associated with classes in the form ``{class_label: weight}``.
               If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts
               can be provided in the same order as the columns of y.
        :return: fitted tree
        """
        if verbose > 1:
            print("building tree %d of %d" % (tree_idx + 1, n_trees))

        X_res, y_res, indices = self.rus.fit_sample(X, y)
        if forest.bootstrap:
            n_samples = X_res.shape[0]
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
            else:
                curr_sample_weight = sample_weight[indices]

            indices = _generate_sample_indices(tree.random_state, n_samples)
            sample_counts = np.bincount(indices, minlength=n_samples)
            curr_sample_weight *= sample_counts

            if class_weight == 'subsample':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    curr_sample_weight *= compute_sample_weight('auto', y, indices)
            elif class_weight == 'balanced_subsample':
                curr_sample_weight *= compute_sample_weight('balanced', y, indices)

            tree.fit(X_res, y_res, sample_weight=curr_sample_weight, check_input=False)
        else:
            tree.fit(X_res, y_res, sample_weight=sample_weight, check_input=False)

        return tree
