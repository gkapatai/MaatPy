Help on module ensemble:

NAME
    ensemble

CLASSES
    imblearn.ensemble.easy_ensemble.EasyEnsemble(imblearn.ensemble.base.BaseEnsembleSampler)
        EasyEnsemble
    
    class EasyEnsemble(imblearn.ensemble.easy_ensemble.EasyEnsemble)
     |  EasyEnsemble(ratio='auto', return_indices=False, random_state=None, replacement=False, n_subsets=10)
     |  
     |  Create an ensemble sets by iteratively applying random under-sampling.
     |  This method iteratively select a random subset and make an ensemble of the
     |  different sets.
     |  
     |  Inherits from imblearn.ensemble.EasyEnsemble. Edited to:
     |  - output a single dataset instead of subsets which allows it to work with a classifier in make_pipeline
     |  - accept a dictionary as ratio; previously it would crash when one was given.
     |  
     |  Method resolution order:
     |      EasyEnsemble
     |      imblearn.ensemble.easy_ensemble.EasyEnsemble
     |      imblearn.ensemble.base.BaseEnsembleSampler
     |      imblearn.base.BaseSampler
     |      imblearn.base.SamplerMixin
     |      abc.NewBase
     |      sklearn.base.BaseEstimator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, ratio='auto', return_indices=False, random_state=None, replacement=False, n_subsets=10)
     |      :param ratio: tr, dict, or callable, optional (default='auto')
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
     |      :param return_indices: bool, optional (default=False)
     |             Whether or not to return the indices of the samples randomly selected from the majority class.
     |      :param random_state: int, RandomState instance or None, optional (default=None)
     |             If int, random_state is the seed used by the random number generator; If RandomState instance,
     |             random_state is the random number generator; If None, the random number generator is the RandomState
     |             instance used by 'np.random'.
     |      :param replacement:  bool, optional (default=False)
     |             Whether or not to sample randomly with replacement or not.
     |      :param n_subsets: int, optional (default=10)
     |             Number of subsets to generate.
     |  
     |  fit(self, X, y)
     |      Find the classes statistics before performing sampling.
     |      Adapted to allow for ratio = 'dict' by recalculating the dict.values = dict.value / n_subsets.
     |      
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
    __all__ = ['EasyEnsemble']

FILE
    /Users/georgiakapatai/Dropbox/Courses/MScBirkbeck/FinalProject/Project/MaatPy/maatpy/samplers/ensemble.py


