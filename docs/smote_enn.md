Help on module smote_enn:

NAME
    smote_enn

CLASSES
    imblearn.combine.smote_enn.SMOTEENN(imblearn.base.SamplerMixin)
        SMOTEENN
    
    class SMOTEENN(imblearn.combine.smote_enn.SMOTEENN)
     |  SMOTEENN(ratio='auto', random_state=None, smote=None, enn=None)
     |  
     |  Class to perform over-sampling using SMOTE and cleaning using ENN.
     |  
     |  Combine over- and under-sampling using SMOTE and Edited Nearest Neighbours.
     |  
     |  Read more in the :ref:`User Guide <combine>`.
     |  
     |  Parameters
     |  ----------
     |  ratio : str, dict, or callable, optional (default='auto')
     |      Ratio to use for resampling the data set.
     |  
     |      - If ``str``, has to be one of: (i) ``'minority'``: resample the
     |        minority class; (ii) ``'majority'``: resample the majority class,
     |        (iii) ``'not minority'``: resample all classes apart of the minority
     |        class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
     |        correspond to ``'all'`` with for over-sampling methods and ``'not
     |        minority'`` for under-sampling methods. The classes targeted will be
     |        over-sampled or under-sampled to achieve an equal number of sample
     |        with the majority or minority class.
     |      - If ``dict``, the keys correspond to the targeted classes. The values
     |        correspond to the desired number of samples.
     |      - If callable, function taking ``y`` and returns a ``dict``. The keys
     |        correspond to the targeted classes. The values correspond to the
     |        desired number of samples.
     |  
     |  random_state : int, RandomState instance or None, optional (default=None)
     |      If int, ``random_state`` is the seed used by the random number
     |      generator; If ``RandomState`` instance, random_state is the random
     |      number generator; If ``None``, the random number generator is the
     |      ``RandomState`` instance used by ``np.random``.
     |  
     |  smote : object, optional (default=SMOTE())
     |      The :class:`imblearn.over_sampling.SMOTE` object to use. If not given,
     |      a :class:`imblearn.over_sampling.SMOTE` object with default parameters
     |      will be given.
     |  
     |  enn : object, optional (default=EditedNearestNeighbours())
     |      The :class:`imblearn.under_sampling.EditedNearestNeighbours` object to
     |      use. If not given, an
     |      :class:`imblearn.under_sampling.EditedNearestNeighbours` object with
     |      default parameters will be given.
     |  
     |  k : int, optional (default=None)
     |      Number of nearest neighbours to used to construct synthetic
     |      samples.
     |  
     |      .. deprecated:: 0.2
     |         `k` is deprecated from 0.2 and will be replaced in 0.4
     |         Give directly a :class:`imblearn.over_sampling.SMOTE` object.
     |  
     |  m : int, optional (default=None)
     |      Number of nearest neighbours to use to determine if a minority
     |      sample is in danger.
     |  
     |      .. deprecated:: 0.2
     |         `m` is deprecated from 0.2 and will be replaced in 0.4
     |         Give directly a :class:`imblearn.over_sampling.SMOTE` object.
     |  
     |  out_step : float, optional (default=None)
     |      Step size when extrapolating.
     |  
     |      .. deprecated:: 0.2
     |         ``out_step`` is deprecated from 0.2 and will be replaced in 0.4
     |         Give directly a :class:`imblearn.over_sampling.SMOTE` object.
     |  
     |  kind_smote : str, optional (default=None)
     |      The type of SMOTE algorithm to use one of the following
     |      options: ``'regular'``, ``'borderline1'``, ``'borderline2'``,
     |      ``'svm'``.
     |  
     |      .. deprecated:: 0.2
     |         `kind_smote` is deprecated from 0.2 and will be replaced in 0.4
     |         Give directly a :class:`imblearn.over_sampling.SMOTE` object.
     |  
     |  size_ngh : int, optional (default=None)
     |      Size of the neighbourhood to consider to compute the average
     |      distance to the minority point samples.
     |  
     |      .. deprecated:: 0.2
     |         size_ngh is deprecated from 0.2 and will be replaced in 0.4
     |         Use ``n_neighbors`` instead.
     |  
     |  n_neighbors : int, optional (default=None)
     |      Size of the neighbourhood to consider to compute the average
     |      distance to the minority point samples.
     |  
     |      .. deprecated:: 0.2
     |         `n_neighbors` is deprecated from 0.2 and will be replaced in 0.4
     |         Give directly a
     |         :class:`imblearn.under_sampling.EditedNearestNeighbours` object.
     |  
     |  kind_sel : str, optional (default=None)
     |      Strategy to use in order to exclude samples.
     |  
     |      - If ``'all'``, all neighbours will have to agree with the samples of
     |        interest to not be excluded.
     |      - If ``'mode'``, the majority vote of the neighbours will be used in
     |        order to exclude a sample.
     |  
     |      .. deprecated:: 0.2
     |         ``kind_sel`` is deprecated from 0.2 and will be replaced in 0.4 Give
     |         directly a :class:`imblearn.under_sampling.EditedNearestNeighbours`
     |         object.
     |  
     |  n_jobs : int, optional (default=None)
     |      The number of threads to open if possible.
     |  
     |      .. deprecated:: 0.2
     |         `n_jobs` is deprecated from 0.2 and will be replaced in 0.4 Give
     |         directly a :class:`imblearn.over_sampling.SMOTE` and
     |         :class:`imblearn.under_sampling.EditedNearestNeighbours` object.
     |  
     |  Notes
     |  -----
     |  The method is presented in [1]_.
     |  
     |  Supports mutli-class resampling. Refer to SMOTE and ENN regarding the
     |  scheme which used.
     |  
     |  See :ref:`sphx_glr_auto_examples_combine_plot_smote_enn.py` and
     |  :ref:`sphx_glr_auto_examples_combine_plot_comparison_combine.py`.
     |  
     |  See also
     |  --------
     |  SMOTETomek : Over-sample using SMOTE followed by under-sampling removing
     |      the Tomek's links.
     |  
     |  References
     |  ----------
     |  .. [1] G. Batista, R. C. Prati, M. C. Monard. "A study of the behavior of
     |     several methods for balancing machine learning training data," ACM
     |     Sigkdd Explorations Newsletter 6 (1), 20-29, 2004.
     |  
     |  Examples
     |  --------
     |  
     |  >>> from collections import Counter
     |  >>> from sklearn.datasets import make_classification
     |  >>> from imblearn.combine import SMOTEENN # doctest: +NORMALIZE_WHITESPACE
     |  >>> X, y = make_classification(n_classes=2, class_sep=2,
     |  ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
     |  ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
     |  >>> print('Original dataset shape {}'.format(Counter(y)))
     |  Original dataset shape Counter({1: 900, 0: 100})
     |  >>> sme = SMOTEENN(random_state=42)
     |  >>> X_res, y_res = sme.fit_sample(X, y)
     |  >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
     |  Resampled dataset shape Counter({0: 900, 1: 881})
     |  
     |  Method resolution order:
     |      SMOTEENN
     |      imblearn.combine.smote_enn.SMOTEENN
     |      imblearn.base.SamplerMixin
     |      abc.NewBase
     |      sklearn.base.BaseEstimator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, ratio='auto', random_state=None, smote=None, enn=None)
     |      Creates an object of the imblearn.combine.SMOTEENN class.
     |      
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
     |      :param enn: object, optional (default=EditedNearestNeighbours())
     |             The :class: imblearn.under_sampling.EditedNearestNeighbours object to use. If none provide a
     |             :class: imblearn.under_sampling.EditedNearestNeighbours object with default parameters will be given.
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
    __all__ = ['SMOTEENN']

FILE
    /Users/georgiakapatai/Dropbox/Courses/MScBirkbeck/FinalProject/Project/MaatPy/maatpy/samplers/smote_enn.py


