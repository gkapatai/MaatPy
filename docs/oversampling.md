Help on module oversampling:

NAME
    oversampling

CLASSES
    imblearn.over_sampling.random_over_sampler.RandomOverSampler(imblearn.over_sampling.base.BaseOverSampler)
        RandomOverSampler
    imblearn.over_sampling.smote.SMOTE(imblearn.over_sampling.base.BaseOverSampler)
        SMOTE
    
    class RandomOverSampler(imblearn.over_sampling.random_over_sampler.RandomOverSampler)
     |  RandomOverSampler(ratio='auto', random_state=None)
     |  
     |  Class to perform random over-sampling.
     |  
     |  Object to over-sample the minority class(es) by picking samples at random
     |  with replacement.
     |  
     |  Read more in the :ref:`User Guide <random_over_sampler>`.
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
     |  Notes
     |  -----
     |  Supports mutli-class resampling by sampling each class independently.
     |  
     |  See
     |  :ref:`sphx_glr_auto_examples_over-sampling_plot_comparison_over_sampling.py`,
     |  :ref:`sphx_glr_auto_examples_over-sampling_plot_random_over_sampling.py`,
     |  and
     |  :ref:`sphx_glr_auto_examples_applications_plot_over_sampling_benchmark_lfw.py`.
     |  
     |  Examples
     |  --------
     |  
     |  >>> from collections import Counter
     |  >>> from sklearn.datasets import make_classification
     |  >>> from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
     |  >>> X, y = make_classification(n_classes=2, class_sep=2,
     |  ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
     |  ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
     |  >>> print('Original dataset shape {}'.format(Counter(y)))
     |  Original dataset shape Counter({1: 900, 0: 100})
     |  >>> ros = RandomOverSampler(random_state=42)
     |  >>> X_res, y_res = ros.fit_sample(X, y)
     |  >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
     |  Resampled dataset shape Counter({0: 900, 1: 900})
     |  
     |  Method resolution order:
     |      RandomOverSampler
     |      imblearn.over_sampling.random_over_sampler.RandomOverSampler
     |      imblearn.over_sampling.base.BaseOverSampler
     |      imblearn.base.BaseSampler
     |      imblearn.base.SamplerMixin
     |      abc.NewBase
     |      sklearn.base.BaseEstimator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, ratio='auto', random_state=None)
     |      Creates an object of the imblearn.over_sampling.RandomOverSampler class.
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
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __abstractmethods__ = frozenset()
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from imblearn.base.BaseSampler:
     |  
     |  fit(self, X, y)
     |      Find the classes statistics before to perform sampling.
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
     |      self : object,
     |          Return self.
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
    
    class SMOTE(imblearn.over_sampling.smote.SMOTE)
     |  SMOTE(ratio='auto', random_state=None, k_neighbors=5, m_neighbors=10, out_step=0.5, kind='regular', svm_estimator=None, n_jobs=1)
     |  
     |  Class to perform over-sampling using SMOTE.
     |  
     |  This object is an implementation of SMOTE - Synthetic Minority
     |  Over-sampling Technique, and the variants Borderline SMOTE 1, 2 and
     |  SVM-SMOTE.
     |  
     |  Read more in the :ref:`User Guide <smote_adasyn>`.
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
     |  k : int, optional (default=None)
     |      Number of nearest neighbours to used to construct synthetic samples.
     |  
     |      .. deprecated:: 0.2
     |         ``k`` is deprecated from 0.2 and will be replaced in 0.4
     |         Use ``k_neighbors`` instead.
     |  
     |  k_neighbors : int or object, optional (default=5)
     |      If ``int``, number of nearest neighbours to used to construct synthetic
     |      samples.  If object, an estimator that inherits from
     |      :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
     |      find the k_neighbors.
     |  
     |  m : int, optional (default=None)
     |      Number of nearest neighbours to use to determine if a minority sample
     |      is in danger. Used with ``kind={'borderline1', 'borderline2',
     |      'svm'}``.
     |  
     |      .. deprecated:: 0.2
     |         ``m`` is deprecated from 0.2 and will be replaced in 0.4
     |         Use ``m_neighbors`` instead.
     |  
     |  m_neighbors : int int or object, optional (default=10)
     |      If int, number of nearest neighbours to use to determine if a minority
     |      sample is in danger. Used with ``kind={'borderline1', 'borderline2',
     |      'svm'}``.  If object, an estimator that inherits
     |      from :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used
     |      to find the k_neighbors.
     |  
     |  out_step : float, optional (default=0.5)
     |      Step size when extrapolating. Used with ``kind='svm'``.
     |  
     |  kind : str, optional (default='regular')
     |      The type of SMOTE algorithm to use one of the following options:
     |      ``'regular'``, ``'borderline1'``, ``'borderline2'``, ``'svm'``.
     |  
     |  svm_estimator : object, optional (default=SVC())
     |      If ``kind='svm'``, a parametrized :class:`sklearn.svm.SVC`
     |      classifier can be passed.
     |  
     |  n_jobs : int, optional (default=1)
     |      The number of threads to open if possible.
     |  
     |  Notes
     |  -----
     |  See the original papers: [1]_, [2]_, [3]_ for more details.
     |  
     |  Supports mutli-class resampling. A one-vs.-rest scheme is used as
     |  originally proposed in [1]_.
     |  
     |  See
     |  :ref:`sphx_glr_auto_examples_applications_plot_over_sampling_benchmark_lfw.py`,
     |  :ref:`sphx_glr_auto_examples_evaluation_plot_classification_report.py`,
     |  :ref:`sphx_glr_auto_examples_evaluation_plot_metrics.py`,
     |  :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py`,
     |  :ref:`sphx_glr_auto_examples_over-sampling_plot_comparison_over_sampling.py`,
     |  and :ref:`sphx_glr_auto_examples_over-sampling_plot_smote.py`.
     |  
     |  See also
     |  --------
     |  ADASYN : Over-sample using ADASYN.
     |  
     |  References
     |  ----------
     |  .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
     |     synthetic minority over-sampling technique," Journal of artificial
     |     intelligence research, 321-357, 2002.
     |  
     |  .. [2] H. Han, W. Wen-Yuan, M. Bing-Huan, "Borderline-SMOTE: a new
     |     over-sampling method in imbalanced data sets learning," Advances in
     |     intelligent computing, 878-887, 2005.
     |  
     |  .. [3] H. M. Nguyen, E. W. Cooper, K. Kamei, "Borderline over-sampling for
     |     imbalanced data classification," International Journal of Knowledge
     |     Engineering and Soft Data Paradigms, 3(1), pp.4-21, 2001.
     |  
     |  Examples
     |  --------
     |  
     |  >>> from collections import Counter
     |  >>> from sklearn.datasets import make_classification
     |  >>> from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
     |  >>> X, y = make_classification(n_classes=2, class_sep=2,
     |  ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
     |  ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
     |  >>> print('Original dataset shape {}'.format(Counter(y)))
     |  Original dataset shape Counter({1: 900, 0: 100})
     |  >>> sm = SMOTE(random_state=42)
     |  >>> X_res, y_res = sm.fit_sample(X, y)
     |  >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
     |  Resampled dataset shape Counter({0: 900, 1: 900})
     |  
     |  Method resolution order:
     |      SMOTE
     |      imblearn.over_sampling.smote.SMOTE
     |      imblearn.over_sampling.base.BaseOverSampler
     |      imblearn.base.BaseSampler
     |      imblearn.base.SamplerMixin
     |      abc.NewBase
     |      sklearn.base.BaseEstimator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, ratio='auto', random_state=None, k_neighbors=5, m_neighbors=10, out_step=0.5, kind='regular', svm_estimator=None, n_jobs=1)
     |      Creates an object of imblearn.under_sampling.SMOTE class.
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
     |      :param k_neighbors: int or object,, optional (default=5)
     |             Number of nearest neighbours to used to construct synthetic samples. If object, an estimator that
     |             inherits from :class:'sklearn.neighbors.base.KNeighborsMixin' that will be used to find the k_neighbors.
     |      :param m_neighbors: int or object, optional (default=10)
     |             If int, number of nearest neighbours to use to determine if a minority sample is in danger.
     |             Used with "kind={'borderline1', 'borderline2', 'svm'}".  If object, an estimator that inherits from
     |             :class:'sklearn.neighbors.base.KNeighborsMixin' that will be used to find the k_neighbors.
     |      :param out_step: float, optional (default=0.5)
     |             Step size when extrapolating. Used with "kind='svm'".
     |      :param kind: str, optional (default='regular')
     |             The type of SMOTE algorithm to use one of the following options: 'regular', 'borderline1',
     |             'borderline2', 'svm'.
     |      :param svm_estimator: object, optional (default=SVC())
     |             If "kind='svm'", a parametrized :class:'sklearn.svm.SVC' classifier can be passed.
     |      :param n_jobs: int, optional (default=1)
     |             The number of threads to open if possible.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __abstractmethods__ = frozenset()
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from imblearn.base.BaseSampler:
     |  
     |  fit(self, X, y)
     |      Find the classes statistics before to perform sampling.
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
     |      self : object,
     |          Return self.
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
    __all__ = ['RandomOverSampler', 'SMOTE']

FILE
    /Users/georgiakapatai/Dropbox/Courses/MScBirkbeck/FinalProject/Project/MaatPy/maatpy/samplers/oversampling.py


