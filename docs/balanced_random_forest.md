Help on module balanced_random_forest:

NAME
    balanced_random_forest - # coding: utf-8

CLASSES
    sklearn.ensemble.forest.RandomForestClassifier(sklearn.ensemble.forest.ForestClassifier)
        BalancedRandomForestClassifier
    
    class BalancedRandomForestClassifier(sklearn.ensemble.forest.RandomForestClassifier)
     |  BalancedRandomForestClassifier(n_estimators=10, bootstrap=True, oob_score=False, max_depth=None, criterion='gini', max_features='auto', ratio='auto', replacement=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
     |  
     |  Implementation of the Balanced Random Forest
     |  
     |  Reference: Chen et al, “Using Random Forest to Learn Imbalanced Data,” UC Berkeley (tech-report), 2004.
     |  
     |  Method resolution order:
     |      BalancedRandomForestClassifier
     |      sklearn.ensemble.forest.RandomForestClassifier
     |      sklearn.ensemble.forest.ForestClassifier
     |      abc.NewBase
     |      sklearn.ensemble.forest.BaseForest
     |      abc.NewBase
     |      sklearn.ensemble.base.BaseEnsemble
     |      abc.NewBase
     |      sklearn.base.BaseEstimator
     |      sklearn.base.MetaEstimatorMixin
     |      sklearn.base.ClassifierMixin
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, n_estimators=10, bootstrap=True, oob_score=False, max_depth=None, criterion='gini', max_features='auto', ratio='auto', replacement=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
     |      :param n_estimators: int, optional (default=50)
     |             The maximum number of estimators at which boosting is terminated.
     |             In case of perfect fit, the learning procedure is stopped early.
     |      :param bootstrap: boolean, optional (default=True)
     |             Whether samples are drawn with replacement.
     |      :param oob_score: bool, optional (default=False)
     |             Whether to use out-of-bag samples to estimate the generalization error
     |      :param ratio: str, dict, or callable, optional (default='auto')
     |             Ratio to use for resampling the data set.
     |             - If "str", has to be one of: (i) 'minority': resample the minority class;
     |               (ii) 'majority': resample the majority class,
     |               (iii) 'not minority': resample all classes apart of the minority class,
     |               (iv) 'all': resample all classes, and (v) 'auto': correspond to 'all' with for over-sampling
     |               methods and 'not_minority' for under-sampling methods. The classes targeted will be over-sampled or
     |               under-sampled to achieve an equal number of sample with the majority or minority class.
     |             - If "dict`", the keys correspond to the targeted classes. The values correspond to the desired number
     |               of samples.
     |             - If callable, function taking "y" and returns a "dict". The keys correspond to the targeted classes.
     |               The values correspond to the desired number of samples.
     |      :param replacement: boolean, optional (default=False)
     |             Whether the sample is with or without replacement
     |      :param n_jobs: int, optional (default=1)
     |             The number of jobs to run in parallel for both `fit` and `predict`.
     |             If -1, then the number of jobs is set to the number of cores.
     |      :param random_state: int, RandomState instance or None, optional (default=None)
     |             If int, random_state is the seed used by the random number generator; If RandomState instance,
     |             random_state is the random number generator; If None, the random number generator is the RandomState
     |             instance used by 'np.random'.
     |      :param verbose: int, optional (default=0)
     |             Controls the verbosity of the building process.
     |      :param warm_start: bool, optional (default=False)
     |             When set to True, reuse the solution of the previous call to fit and add more estimators to
     |             the ensemble, otherwise, just fit a whole new ensemble.
     |      :param class_weight: dict, list of dicts, "balanced", "balanced_subsample" or None, optional (default=None)
     |             Weights associated with classes in the form ``{class_label: weight}``.
     |             If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts
     |             can be provided in the same order as the columns of y.
     |  
     |  fit(self, X, y, sample_weight=None)
     |      Build a forest of trees from the training set (X, y).
     |      
     |      Copied from sklearn.ensemble.BaseForest.fit() to use the edited _parallel_build_function
     |      
     |      :param X:{array-like, sparse matrix}, shape (n_samples, n_features)
     |             Matrix containing the training data.
     |      :param y: array-like, shape (n_samples,)
     |             Corresponding label for each sample in X.
     |      :param sample_weight: array-like of shape = [n_samples], optional
     |             Sample weights. If None, the sample weights are equally weighted. Splits that would create child
     |             nodes with net zero or negative weight are ignored while searching for a split in each node. In
     |             the case of classification, splits are also ignored if they would result in any single class carrying
     |             a negative weight in either child node.
     |      :return: object; Return self
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __abstractmethods__ = frozenset()
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.ensemble.forest.ForestClassifier:
     |  
     |  predict(self, X)
     |      Predict class for X.
     |      
     |      The predicted class of an input sample is a vote by the trees in
     |      the forest, weighted by their probability estimates. That is,
     |      the predicted class is the one with highest mean probability
     |      estimate across the trees.
     |      
     |      Parameters
     |      ----------
     |      X : array-like or sparse matrix of shape = [n_samples, n_features]
     |          The input samples. Internally, its dtype will be converted to
     |          ``dtype=np.float32``. If a sparse matrix is provided, it will be
     |          converted into a sparse ``csr_matrix``.
     |      
     |      Returns
     |      -------
     |      y : array of shape = [n_samples] or [n_samples, n_outputs]
     |          The predicted classes.
     |  
     |  predict_log_proba(self, X)
     |      Predict class log-probabilities for X.
     |      
     |      The predicted class log-probabilities of an input sample is computed as
     |      the log of the mean predicted class probabilities of the trees in the
     |      forest.
     |      
     |      Parameters
     |      ----------
     |      X : array-like or sparse matrix of shape = [n_samples, n_features]
     |          The input samples. Internally, its dtype will be converted to
     |          ``dtype=np.float32``. If a sparse matrix is provided, it will be
     |          converted into a sparse ``csr_matrix``.
     |      
     |      Returns
     |      -------
     |      p : array of shape = [n_samples, n_classes], or a list of n_outputs
     |          such arrays if n_outputs > 1.
     |          The class probabilities of the input samples. The order of the
     |          classes corresponds to that in the attribute `classes_`.
     |  
     |  predict_proba(self, X)
     |      Predict class probabilities for X.
     |      
     |      The predicted class probabilities of an input sample are computed as
     |      the mean predicted class probabilities of the trees in the forest. The
     |      class probability of a single tree is the fraction of samples of the same
     |      class in a leaf.
     |      
     |      Parameters
     |      ----------
     |      X : array-like or sparse matrix of shape = [n_samples, n_features]
     |          The input samples. Internally, its dtype will be converted to
     |          ``dtype=np.float32``. If a sparse matrix is provided, it will be
     |          converted into a sparse ``csr_matrix``.
     |      
     |      Returns
     |      -------
     |      p : array of shape = [n_samples, n_classes], or a list of n_outputs
     |          such arrays if n_outputs > 1.
     |          The class probabilities of the input samples. The order of the
     |          classes corresponds to that in the attribute `classes_`.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.ensemble.forest.BaseForest:
     |  
     |  apply(self, X)
     |      Apply trees in the forest to X, return leaf indices.
     |      
     |      Parameters
     |      ----------
     |      X : array-like or sparse matrix, shape = [n_samples, n_features]
     |          The input samples. Internally, its dtype will be converted to
     |          ``dtype=np.float32``. If a sparse matrix is provided, it will be
     |          converted into a sparse ``csr_matrix``.
     |      
     |      Returns
     |      -------
     |      X_leaves : array_like, shape = [n_samples, n_estimators]
     |          For each datapoint x in X and for each tree in the forest,
     |          return the index of the leaf x ends up in.
     |  
     |  decision_path(self, X)
     |      Return the decision path in the forest
     |      
     |      .. versionadded:: 0.18
     |      
     |      Parameters
     |      ----------
     |      X : array-like or sparse matrix, shape = [n_samples, n_features]
     |          The input samples. Internally, its dtype will be converted to
     |          ``dtype=np.float32``. If a sparse matrix is provided, it will be
     |          converted into a sparse ``csr_matrix``.
     |      
     |      Returns
     |      -------
     |      indicator : sparse csr array, shape = [n_samples, n_nodes]
     |          Return a node indicator matrix where non zero elements
     |          indicates that the samples goes through the nodes.
     |      
     |      n_nodes_ptr : array of size (n_estimators + 1, )
     |          The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
     |          gives the indicator value for the i-th estimator.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from sklearn.ensemble.forest.BaseForest:
     |  
     |  feature_importances_
     |      Return the feature importances (the higher, the more important the
     |         feature).
     |      
     |      Returns
     |      -------
     |      feature_importances_ : array, shape = [n_features]
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.ensemble.base.BaseEnsemble:
     |  
     |  __getitem__(self, index)
     |      Returns the index'th estimator in the ensemble.
     |  
     |  __iter__(self)
     |      Returns iterator over estimators in the ensemble.
     |  
     |  __len__(self)
     |      Returns the number of estimators in the ensemble.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.base.BaseEstimator:
     |  
     |  __getstate__(self)
     |  
     |  __repr__(self)
     |      Return repr(self).
     |  
     |  __setstate__(self, state)
     |  
     |  get_params(self, deep=True)
     |      Get parameters for this estimator.
     |      
     |      Parameters
     |      ----------
     |      deep : boolean, optional
     |          If True, will return the parameters for this estimator and
     |          contained subobjects that are estimators.
     |      
     |      Returns
     |      -------
     |      params : mapping of string to any
     |          Parameter names mapped to their values.
     |  
     |  set_params(self, **params)
     |      Set the parameters of this estimator.
     |      
     |      The method works on simple estimators as well as on nested objects
     |      (such as pipelines). The latter have parameters of the form
     |      ``<component>__<parameter>`` so that it's possible to update each
     |      component of a nested object.
     |      
     |      Returns
     |      -------
     |      self
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from sklearn.base.BaseEstimator:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.base.ClassifierMixin:
     |  
     |  score(self, X, y, sample_weight=None)
     |      Returns the mean accuracy on the given test data and labels.
     |      
     |      In multi-label classification, this is the subset accuracy
     |      which is a harsh metric since you require for each sample that
     |      each label set be correctly predicted.
     |      
     |      Parameters
     |      ----------
     |      X : array-like, shape = (n_samples, n_features)
     |          Test samples.
     |      
     |      y : array-like, shape = (n_samples) or (n_samples, n_outputs)
     |          True labels for X.
     |      
     |      sample_weight : array-like, shape = [n_samples], optional
     |          Sample weights.
     |      
     |      Returns
     |      -------
     |      score : float
     |          Mean accuracy of self.predict(X) wrt. y.

DATA
    MAX_INT = 2147483647

FILE
    /Users/georgiakapatai/Dropbox/Courses/MScBirkbeck/FinalProject/Project/MaatPy/maatpy/classifiers/balanced_random_forest.py


