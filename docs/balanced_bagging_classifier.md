Help on module balanced_bagging_classifier:

NAME
    balanced_bagging_classifier

CLASSES
    imblearn.ensemble.classifier.BalancedBaggingClassifier(sklearn.ensemble.bagging.BaggingClassifier)
        BalancedBaggingClassifier
    
    class BalancedBaggingClassifier(imblearn.ensemble.classifier.BalancedBaggingClassifier)
     |  BalancedBaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, ratio='auto', replacement=False, n_jobs=1, random_state=None, verbose=0)
     |  
     |  A Bagging classifier with additional balancing.
     |  This implementation of Bagging is similar to the scikit-learn implementation.
     |  It includes an additional step to balance the training set at fit time using a 'RandomUnderSampler'.
     |  
     |  This is the imblearn.ensemble.BalancedBaggingClassifier class with a minor change in the _validate_estimator
     |  function.
     |  Source: https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/ensemble/classifier.py
     |  
     |  Method resolution order:
     |      BalancedBaggingClassifier
     |      imblearn.ensemble.classifier.BalancedBaggingClassifier
     |      sklearn.ensemble.bagging.BaggingClassifier
     |      sklearn.ensemble.bagging.BaseBagging
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
     |  __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, ratio='auto', replacement=False, n_jobs=1, random_state=None, verbose=0)
     |      :param base_estimator: object, optional (default=DecisionTreeClassifier)
     |             The base estimator from which the boosted ensemble is built.
     |             Support for sample weighting is required, as well as proper 'classes_' and 'n_classes_' attributes.
     |      :param n_estimators: int, optional (default=50)
     |             The maximum number of estimators at which boosting is terminated.
     |             In case of perfect fit, the learning procedure is stopped early.
     |      :param max_samples: int or float, optional (default=1.0)
     |             The number of samples to draw from X to train each base estimator.
     |             - If int, then draw `max_samples` samples.
     |             - If float, then draw `max_samples * X.shape[0]` samples.
     |      :param max_features: int or float, optional (default=1.0)
     |             The number of features to draw from X to train each base estimator.
     |             - If int, then draw `max_features` features.
     |             - If float, then draw `max_features * X.shape[1]` features.
     |      :param bootstrap: boolean, optional (default=True)
     |             Whether samples are drawn with replacement.
     |      :param bootstrap_features: boolean, optional (default=False)
     |             Whether features are drawn with replacement.
     |      :param oob_score: bool, optional (default=False)
     |             Whether to use out-of-bag samples to estimate the generalization error
     |      :param warm_start: bool, optional (default=False)
     |             When set to True, reuse the solution of the previous call to fit and add more estimators to
     |             the ensemble, otherwise, just fit a whole new ensemble.
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
     |  
     |  fit(self, X, y)
     |      Find the classes statistics before performing sampling
     |      
     |      :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
     |             Matrix containing the data which have to be sampled.
     |      :param y: array-like, shape (n_samples,)
     |             Corresponding label for each sample in X.
     |      :return: object; Return self
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __abstractmethods__ = frozenset()
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.ensemble.bagging.BaggingClassifier:
     |  
     |  decision_function(self, X)
     |      Average of the decision functions of the base classifiers.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape = [n_samples, n_features]
     |          The training input samples. Sparse matrices are accepted only if
     |          they are supported by the base estimator.
     |      
     |      Returns
     |      -------
     |      score : array, shape = [n_samples, k]
     |          The decision function of the input samples. The columns correspond
     |          to the classes in sorted order, as they appear in the attribute
     |          ``classes_``. Regression and binary classification are special
     |          cases with ``k == 1``, otherwise ``k==n_classes``.
     |  
     |  predict(self, X)
     |      Predict class for X.
     |      
     |      The predicted class of an input sample is computed as the class with
     |      the highest mean predicted probability. If base estimators do not
     |      implement a ``predict_proba`` method, then it resorts to voting.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape = [n_samples, n_features]
     |          The training input samples. Sparse matrices are accepted only if
     |          they are supported by the base estimator.
     |      
     |      Returns
     |      -------
     |      y : array of shape = [n_samples]
     |          The predicted classes.
     |  
     |  predict_log_proba(self, X)
     |      Predict class log-probabilities for X.
     |      
     |      The predicted class log-probabilities of an input sample is computed as
     |      the log of the mean predicted class probabilities of the base
     |      estimators in the ensemble.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape = [n_samples, n_features]
     |          The training input samples. Sparse matrices are accepted only if
     |          they are supported by the base estimator.
     |      
     |      Returns
     |      -------
     |      p : array of shape = [n_samples, n_classes]
     |          The class log-probabilities of the input samples. The order of the
     |          classes corresponds to that in the attribute `classes_`.
     |  
     |  predict_proba(self, X)
     |      Predict class probabilities for X.
     |      
     |      The predicted class probabilities of an input sample is computed as
     |      the mean predicted class probabilities of the base estimators in the
     |      ensemble. If base estimators do not implement a ``predict_proba``
     |      method, then it resorts to voting and the predicted class probabilities
     |      of an input sample represents the proportion of estimators predicting
     |      each class.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape = [n_samples, n_features]
     |          The training input samples. Sparse matrices are accepted only if
     |          they are supported by the base estimator.
     |      
     |      Returns
     |      -------
     |      p : array of shape = [n_samples, n_classes]
     |          The class probabilities of the input samples. The order of the
     |          classes corresponds to that in the attribute `classes_`.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from sklearn.ensemble.bagging.BaseBagging:
     |  
     |  estimators_samples_
     |      The subset of drawn samples for each base estimator.
     |      
     |      Returns a dynamically generated list of boolean masks identifying
     |      the samples used for fitting each member of the ensemble, i.e.,
     |      the in-bag samples.
     |      
     |      Note: the list is re-created at each call to the property in order
     |      to reduce the object memory footprint by not storing the sampling
     |      data. Thus fetching the property may be slower than expected.
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
    __all__ = ['BalancedBaggingClassifier']

FILE
    /Users/georgiakapatai/Dropbox/Courses/MScBirkbeck/FinalProject/Project/MaatPy/maatpy/classifiers/balanced_bagging_classifier.py


