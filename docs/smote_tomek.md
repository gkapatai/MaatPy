Help on module smote_tomek:

NAME
    smote_tomek

CLASSES
    imblearn.combine.smote_tomek.SMOTETomek(imblearn.base.SamplerMixin)
        SMOTETomek
    
    class SMOTETomek(imblearn.combine.smote_tomek.SMOTETomek)
     |  SMOTETomek(ratio='auto', random_state=None, smote=None, tomek=None)
     |  
     |  Class to perform over-sampling using SMOTE and cleaning using Tomek links.
     |  Combine over- and under-sampling using SMOTE and Tomek links.
     |  
     |  Inherits from imblearn.combine.SMOTETomek. Edited to perform under-sampling first to remove Tomek links
     |  and then perform oversampling with SMOTE.
     |  
     |  Method resolution order:
     |      SMOTETomek
     |      imblearn.combine.smote_tomek.SMOTETomek
     |      imblearn.base.SamplerMixin
     |      abc.NewBase
     |      sklearn.base.BaseEstimator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, ratio='auto', random_state=None, smote=None, tomek=None)
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
     |      :param random_state: int, RandomState instance or None, optional (default=None)
     |             If int, random_state is the seed used by the random number generator; If RandomState instance,
     |             random_state is the random number generator; If None, the random number generator is the RandomState
     |             instance used by 'np.random'.
     |      :param smote: object, optional (default=SMOTE())
     |             The :class: imblearn.over_sampling.SMOTE object to use. If none provide a
     |             :class: imblearn.over_sampling.SMOTE object with default parameters will be given.
     |      :param tomek: object, optional (default=TomekLinks())
     |             The :class: imblearn.under_sampling.TomekLinks object to use. If none provide a
     |             :class: imblearn.under_sampling.TomekLinks object with default parameters will be given.
     |  
     |  fit(self, X, y)
     |      Find the classes statistics before to perform sampling.
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
     |  Methods inherited from imblearn.base.SamplerMixin:
     |  
     |  __getstate__(self)
     |      Prevent logger from being pickled.
     |  
     |  __setstate__(self, dict)
     |      Re-open the logger.
     |  
     |  fit_sample(self, X, y)
     |      Fit the statistics and resample the data directly.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix}, shape (n_samples, n_features)
     |          Matrix containing the data which have to be sampled.
     |      
     |      y : array-like, shape (n_samples,)
     |          Corresponding label for each sample in X.
     |      
     |      Returns
     |      -------
     |      X_resampled : {array-like, sparse matrix}, shape (n_samples_new, n_features)
     |          The array containing the resampled data.
     |      
     |      y_resampled : array-like, shape (n_samples_new,)
     |          The corresponding label of `X_resampled`
     |  
     |  sample(self, X, y)
     |      Resample the dataset.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix}, shape (n_samples, n_features)
     |          Matrix containing the data which have to be sampled.
     |      
     |      y : array-like, shape (n_samples,)
     |          Corresponding label for each sample in X.
     |      
     |      Returns
     |      -------
     |      X_resampled : {ndarray, sparse matrix}, shape (n_samples_new, n_features)
     |          The array containing the resampled data.
     |      
     |      y_resampled : ndarray, shape (n_samples_new)
     |          The corresponding label of `X_resampled`
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.base.BaseEstimator:
     |  
     |  __repr__(self)
     |      Return repr(self).
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

DATA
    __all__ = ['SMOTETomek']

FILE
    /Users/georgiakapatai/Dropbox/Courses/MScBirkbeck/FinalProject/Project/MaatPy/maatpy/samplers/smote_tomek.py


