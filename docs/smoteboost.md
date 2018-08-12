Help on module smoteboost:

NAME
    smoteboost

CLASSES
    sklearn.ensemble.weight_boosting.AdaBoostClassifier(sklearn.ensemble.weight_boosting.BaseWeightBoosting, sklearn.base.ClassifierMixin)
        SMOTEBoost
    
    class SMOTEBoost(sklearn.ensemble.weight_boosting.AdaBoostClassifier)
     |  SMOTEBoost(k_neighbors=5, base_estimator=None, n_estimators=50, learning_rate=1.0, ratio='auto', algorithm='SAMME.R', random_state=None, n_jobs=1)
     |  
     |  Implementation of SMOTEBoost.
     |  
     |  SMOTEBoost introduces data sampling into the AdaBoost algorithm by oversampling the minority class
     |  using SMOTE on each boosting iteration [1]. This implementation inherits methods from the scikit-learn
     |  AdaBoostClassifier class and some code from the SMOTEBoost class from
     |  https://github.com/dialnd/imbalanced-algorithms/blob/master/smote.py.
     |  Adjusted to work with the imblearn.over-sampling.SMOTE class
     |  
     |  References
     |  ----------
     |  .. [1] N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer.
     |         "SMOTEBoost: Improving Prediction of the Minority Class in Boosting."
     |         European Conference on Principles of Data Mining and Knowledge Discovery (PKDD), 2003.
     |  
     |  Method resolution order:
     |      SMOTEBoost
     |      sklearn.ensemble.weight_boosting.AdaBoostClassifier
     |      sklearn.ensemble.weight_boosting.BaseWeightBoosting
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
     |  __init__(self, k_neighbors=5, base_estimator=None, n_estimators=50, learning_rate=1.0, ratio='auto', algorithm='SAMME.R', random_state=None, n_jobs=1)
     |      :param k_neighbors: int, optional (default=5)
     |             Number of nearest neighbors.
     |      :param base_estimator:object, optional (default=DecisionTreeClassifier)
     |             The base estimator from which the boosted ensemble is built.
     |             Support for sample weighting is required, as well as proper 'classes_' and 'n_classes_' attributes.
     |      :param n_estimators: int, optional (default=50)
     |             The maximum number of estimators at which boosting is terminated.
     |             In case of perfect fit, the learning procedure is stopped early.
     |      :param learning_rate: float, optional (default=1.)
     |             Learning rate shrinks the contribution of each classifier by "learning_rate".
     |             There is a trade-off between "learning_rate" and "n_estimators".
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
     |      :param algorithm: {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
     |             If 'SAMME.R' then use the SAMME.R real boosting algorithm. The "base_estimator" must support
     |             calculation of class probabilities.
     |             If 'SAMME' then use the SAMME discrete boosting algorithm.
     |             The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with
     |             fewer boosting iterations.
     |      :param random_state: int, RandomState instance or None, optional (default=None)
     |             If int, random_state is the seed used by the random number generator; If RandomState instance,
     |             random_state is the random number generator; If None, the random number generator is the RandomState
     |             instance used by 'np.random'.
     |      :param n_jobs: int, optional (default=1)
     |             The number of jobs to run in parallel for both `fit` and `predict`.
     |             If -1, then the number of jobs is set to the number of cores.
     |  
     |  fit(self, X, y, sample_weight=None)
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
     |  Methods inherited from sklearn.ensemble.weight_boosting.AdaBoostClassifier:
     |  
     |  decision_function(self, X)
     |      Compute the decision function of ``X``.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape = [n_samples, n_features]
     |          The training input samples. Sparse matrix can be CSC, CSR, COO,
     |          DOK, or LIL. DOK and LIL are converted to CSR.
     |      
     |      Returns
     |      -------
     |      score : array, shape = [n_samples, k]
     |          The decision function of the input samples. The order of
     |          outputs is the same of that of the `classes_` attribute.
     |          Binary classification is a special cases with ``k == 1``,
     |          otherwise ``k==n_classes``. For binary classification,
     |          values closer to -1 or 1 mean more like the first or second
     |          class in ``classes_``, respectively.
     |  
     |  predict(self, X)
     |      Predict classes for X.
     |      
     |      The predicted class of an input sample is computed as the weighted mean
     |      prediction of the classifiers in the ensemble.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape = [n_samples, n_features]
     |          The training input samples. Sparse matrix can be CSC, CSR, COO,
     |          DOK, or LIL. DOK and LIL are converted to CSR.
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
     |      the weighted mean predicted class log-probabilities of the classifiers
     |      in the ensemble.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape = [n_samples, n_features]
     |          The training input samples. Sparse matrix can be CSC, CSR, COO,
     |          DOK, or LIL. DOK and LIL are converted to CSR.
     |      
     |      Returns
     |      -------
     |      p : array of shape = [n_samples]
     |          The class probabilities of the input samples. The order of
     |          outputs is the same of that of the `classes_` attribute.
     |  
     |  predict_proba(self, X)
     |      Predict class probabilities for X.
     |      
     |      The predicted class probabilities of an input sample is computed as
     |      the weighted mean predicted class probabilities of the classifiers
     |      in the ensemble.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape = [n_samples, n_features]
     |          The training input samples. Sparse matrix can be CSC, CSR, COO,
     |          DOK, or LIL. DOK and LIL are converted to CSR.
     |      
     |      Returns
     |      -------
     |      p : array of shape = [n_samples]
     |          The class probabilities of the input samples. The order of
     |          outputs is the same of that of the `classes_` attribute.
     |  
     |  staged_decision_function(self, X)
     |      Compute decision function of ``X`` for each boosting iteration.
     |      
     |      This method allows monitoring (i.e. determine error on testing set)
     |      after each boosting iteration.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape = [n_samples, n_features]
     |          The training input samples. Sparse matrix can be CSC, CSR, COO,
     |          DOK, or LIL. DOK and LIL are converted to CSR.
     |      
     |      Returns
     |      -------
     |      score : generator of array, shape = [n_samples, k]
     |          The decision function of the input samples. The order of
     |          outputs is the same of that of the `classes_` attribute.
     |          Binary classification is a special cases with ``k == 1``,
     |          otherwise ``k==n_classes``. For binary classification,
     |          values closer to -1 or 1 mean more like the first or second
     |          class in ``classes_``, respectively.
     |  
     |  staged_predict(self, X)
     |      Return staged predictions for X.
     |      
     |      The predicted class of an input sample is computed as the weighted mean
     |      prediction of the classifiers in the ensemble.
     |      
     |      This generator method yields the ensemble prediction after each
     |      iteration of boosting and therefore allows monitoring, such as to
     |      determine the prediction on a test set after each boost.
     |      
     |      Parameters
     |      ----------
     |      X : array-like of shape = [n_samples, n_features]
     |          The input samples.
     |      
     |      Returns
     |      -------
     |      y : generator of array, shape = [n_samples]
     |          The predicted classes.
     |  
     |  staged_predict_proba(self, X)
     |      Predict class probabilities for X.
     |      
     |      The predicted class probabilities of an input sample is computed as
     |      the weighted mean predicted class probabilities of the classifiers
     |      in the ensemble.
     |      
     |      This generator method yields the ensemble predicted class probabilities
     |      after each iteration of boosting and therefore allows monitoring, such
     |      as to determine the predicted class probabilities on a test set after
     |      each boost.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape = [n_samples, n_features]
     |          The training input samples. Sparse matrix can be CSC, CSR, COO,
     |          DOK, or LIL. DOK and LIL are converted to CSR.
     |      
     |      Returns
     |      -------
     |      p : generator of array, shape = [n_samples]
     |          The class probabilities of the input samples. The order of
     |          outputs is the same of that of the `classes_` attribute.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.ensemble.weight_boosting.BaseWeightBoosting:
     |  
     |  staged_score(self, X, y, sample_weight=None)
     |      Return staged scores for X, y.
     |      
     |      This generator method yields the ensemble score after each iteration of
     |      boosting and therefore allows monitoring, such as to determine the
     |      score on a test set after each boost.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape = [n_samples, n_features]
     |          The training input samples. Sparse matrix can be CSC, CSR, COO,
     |          DOK, or LIL. DOK and LIL are converted to CSR.
     |      
     |      y : array-like, shape = [n_samples]
     |          Labels for X.
     |      
     |      sample_weight : array-like, shape = [n_samples], optional
     |          Sample weights.
     |      
     |      Returns
     |      -------
     |      z : float
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from sklearn.ensemble.weight_boosting.BaseWeightBoosting:
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
    __all__ = ['SMOTEBoost']

FILE
    /Users/georgiakapatai/Dropbox/Courses/MScBirkbeck/FinalProject/Project/MaatPy/maatpy/classifiers/smoteboost.py


