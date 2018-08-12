Help on module undersampling:

NAME
    undersampling

CLASSES
    imblearn.under_sampling.prototype_generation.cluster_centroids.ClusterCentroids(imblearn.under_sampling.base.BaseUnderSampler)
        ClusterCentroids
    imblearn.under_sampling.prototype_selection.edited_nearest_neighbours.EditedNearestNeighbours(imblearn.under_sampling.base.BaseCleaningSampler)
        EditedNearestNeighbours
    imblearn.under_sampling.prototype_selection.random_under_sampler.RandomUnderSampler(imblearn.under_sampling.base.BaseUnderSampler)
        RandomUnderSampler
    imblearn.under_sampling.prototype_selection.tomek_links.TomekLinks(imblearn.under_sampling.base.BaseCleaningSampler)
        TomekLinks
    
    class ClusterCentroids(imblearn.under_sampling.prototype_generation.cluster_centroids.ClusterCentroids)
     |  ClusterCentroids(ratio='auto', random_state=None, estimator=None, voting='auto', n_jobs=1)
     |  
     |  Perform under-sampling by generating centroids based on
     |  clustering methods.
     |  
     |  Method that under samples the majority class by replacing a
     |  cluster of majority samples by the cluster centroid of a KMeans
     |  algorithm.  This algorithm keeps N majority samples by fitting the
     |  KMeans algorithm with N cluster to the majority class and using
     |  the coordinates of the N cluster centroids as the new majority
     |  samples.
     |  
     |  Read more in the :ref:`User Guide <cluster_centroids>`.
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
     |  estimator : object, optional(default=KMeans())
     |      Pass a :class:`sklearn.cluster.KMeans` estimator.
     |  
     |  voting : str, optional (default='auto')
     |      Voting strategy to generate the new samples:
     |  
     |      - If ``'hard'``, the nearest-neighbors of the centroids found using the
     |        clustering algorithm will be used.
     |      - If ``'soft'``, the centroids found by the clustering algorithm will
     |        be used.
     |      - If ``'auto'``, if the input is sparse, it will default on ``'hard'``
     |        otherwise, ``'soft'`` will be used.
     |  
     |      .. versionadded:: 0.3.0
     |  
     |  n_jobs : int, optional (default=1)
     |      The number of threads to open if possible.
     |  
     |  Notes
     |  -----
     |  Supports mutli-class resampling by sampling each class independently.
     |  
     |  See :ref:`sphx_glr_auto_examples_under-sampling_plot_cluster_centroids.py`.
     |  
     |  Examples
     |  --------
     |  
     |  >>> from collections import Counter
     |  >>> from sklearn.datasets import make_classification
     |  >>> from imblearn.under_sampling import ClusterCentroids # doctest: +NORMALIZE_WHITESPACE
     |  >>> X, y = make_classification(n_classes=2, class_sep=2,
     |  ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
     |  ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
     |  >>> print('Original dataset shape {}'.format(Counter(y)))
     |  Original dataset shape Counter({1: 900, 0: 100})
     |  >>> cc = ClusterCentroids(random_state=42)
     |  >>> X_res, y_res = cc.fit_sample(X, y)
     |  >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
     |  ... # doctest: +ELLIPSIS
     |  Resampled dataset shape Counter({...})
     |  
     |  Method resolution order:
     |      ClusterCentroids
     |      imblearn.under_sampling.prototype_generation.cluster_centroids.ClusterCentroids
     |      imblearn.under_sampling.base.BaseUnderSampler
     |      imblearn.base.BaseSampler
     |      imblearn.base.SamplerMixin
     |      abc.NewBase
     |      sklearn.base.BaseEstimator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, ratio='auto', random_state=None, estimator=None, voting='auto', n_jobs=1)
     |      Creates an object of the imblearn.under_sampling.ClusterCentroids class.
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
     |      :param estimator: object, optional(default=KMeans())
     |             Pass a :class:'sklearn.cluster.KMeans' estimator.
     |      :param voting: str, optional (default='auto')
     |             Voting strategy to generate the new samples:
     |             - If 'hard', the nearest-neighbors of the centroids found using the clustering algorithm will be used.
     |             - If 'soft', the centroids found by the clustering algorithm will be used.
     |             - If 'auto', if the input is sparse, it will default on 'hard' otherwise, 'soft' will be used.
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
    
    class EditedNearestNeighbours(imblearn.under_sampling.prototype_selection.edited_nearest_neighbours.EditedNearestNeighbours)
     |  EditedNearestNeighbours(ratio='auto', return_indices=False, random_state=None, n_neighbors=3, kind_sel='all', n_jobs=1)
     |  
     |  Class to perform under-sampling based on the edited nearest neighbour
     |  method.
     |  
     |  Read more in the :ref:`User Guide <edited_nearest_neighbors>`.
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
     |      .. warning::
     |         This algorithm is a cleaning under-sampling method. When providing a
     |         ``dict``, only the targeted classes will be used; the number of
     |         samples will be discarded.
     |  
     |  return_indices : bool, optional (default=False)
     |      Whether or not to return the indices of the samples randomly
     |      selected from the majority class.
     |  
     |  random_state : int, RandomState instance or None, optional (default=None)
     |      If int, ``random_state`` is the seed used by the random number
     |      generator; If ``RandomState`` instance, random_state is the random
     |      number generator; If ``None``, the random number generator is the
     |      ``RandomState`` instance used by ``np.random``.
     |  
     |  size_ngh : int, optional (default=None)
     |      Size of the neighbourhood to consider to compute the nearest-neighbors.
     |  
     |     .. deprecated:: 0.2
     |        ``size_ngh`` is deprecated from 0.2 and will be replaced in 0.4
     |        Use ``n_neighbors`` instead.
     |  
     |  n_neighbors : int or object, optional (default=3)
     |      If ``int``, size of the neighbourhood to consider to compute the
     |      nearest neighbors. If object, an estimator that inherits from
     |      :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
     |      find the nearest-neighbors.
     |  
     |  kind_sel : str, optional (default='all')
     |      Strategy to use in order to exclude samples.
     |  
     |      - If ``'all'``, all neighbours will have to agree with the samples of
     |        interest to not be excluded.
     |      - If ``'mode'``, the majority vote of the neighbours will be used in
     |        order to exclude a sample.
     |  
     |  n_jobs : int, optional (default=1)
     |      The number of threads to open if possible.
     |  
     |  Notes
     |  -----
     |  The method is based on [1]_.
     |  
     |  Supports mutli-class resampling. A one-vs.-rest scheme is used when
     |  sampling a class as proposed in [1]_.
     |  
     |  See
     |  :ref:`sphx_glr_auto_examples_pipeline_plot_pipeline_classification.py` and
     |  :ref:`sphx_glr_auto_examples_under-sampling_plot_enn_renn_allknn.py`.
     |  
     |  See also
     |  --------
     |  CondensedNearestNeighbour, RepeatedEditedNearestNeighbours, AllKNN
     |  
     |  References
     |  ----------
     |  .. [1] D. Wilson, Asymptotic" Properties of Nearest Neighbor Rules Using
     |     Edited Data," In IEEE Transactions on Systems, Man, and Cybernetrics,
     |     vol. 2 (3), pp. 408-421, 1972.
     |  
     |  Examples
     |  --------
     |  
     |  >>> from collections import Counter
     |  >>> from sklearn.datasets import make_classification
     |  >>> from imblearn.under_sampling import EditedNearestNeighbours # doctest: +NORMALIZE_WHITESPACE
     |  >>> X, y = make_classification(n_classes=2, class_sep=2,
     |  ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
     |  ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
     |  >>> print('Original dataset shape {}'.format(Counter(y)))
     |  Original dataset shape Counter({1: 900, 0: 100})
     |  >>> enn = EditedNearestNeighbours(random_state=42)
     |  >>> X_res, y_res = enn.fit_sample(X, y)
     |  >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
     |  Resampled dataset shape Counter({1: 887, 0: 100})
     |  
     |  Method resolution order:
     |      EditedNearestNeighbours
     |      imblearn.under_sampling.prototype_selection.edited_nearest_neighbours.EditedNearestNeighbours
     |      imblearn.under_sampling.base.BaseCleaningSampler
     |      imblearn.base.BaseSampler
     |      imblearn.base.SamplerMixin
     |      abc.NewBase
     |      sklearn.base.BaseEstimator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, ratio='auto', return_indices=False, random_state=None, n_neighbors=3, kind_sel='all', n_jobs=1)
     |      Creates an object of the imblearn.under_sampling.EditedNearestNeighbours class.
     |      
     |      :param ratio:  str, dict, or callable, optional (default='auto')
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
     |      :param n_neighbors: int or object, optional (default=3)
     |             If ``int``, size of the neighbourhood to consider to compute the nearest neighbors.
     |             If object, an estimator that inherits from :class:'sklearn.neighbors.base.KNeighborsMixin'
     |              that will be used to find the nearest-neighbors.
     |      :param kind_sel: str, optional (default='all')
     |             Strategy to use in order to exclude samples.
     |             - If "all", all neighbours will have to agree with the samples of interest to not be excluded.
     |             - If "mode", the majority vote of the neighbours will be used in order to exclude a sample.
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
    
    class RandomUnderSampler(imblearn.under_sampling.prototype_selection.random_under_sampler.RandomUnderSampler)
     |  RandomUnderSampler(ratio='auto', return_indices=False, random_state=None, replacement=False)
     |  
     |  Class to perform random under-sampling.
     |  
     |  Under-sample the majority class(es) by randomly picking samples
     |  with or without replacement.
     |  
     |  Read more in the :ref:`User Guide <controlled_under_sampling>`.
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
     |  return_indices : bool, optional (default=False)
     |      Whether or not to return the indices of the samples randomly selected
     |      from the majority class.
     |  
     |  random_state : int, RandomState instance or None, optional (default=None)
     |      If int, ``random_state`` is the seed used by the random number
     |      generator; If ``RandomState`` instance, random_state is the random
     |      number generator; If ``None``, the random number generator is the
     |      ``RandomState`` instance used by ``np.random``.
     |  
     |  replacement : boolean, optional (default=False)
     |      Whether the sample is with or without replacement.
     |  
     |  Notes
     |  -----
     |  Supports mutli-class resampling by sampling each class independently.
     |  
     |  See
     |  :ref:`sphx_glr_auto_examples_plot_ratio_usage.py` and
     |  :ref:`sphx_glr_auto_examples_under-sampling_plot_random_under_sampler.py`
     |  
     |  Examples
     |  --------
     |  
     |  >>> from collections import Counter
     |  >>> from sklearn.datasets import make_classification
     |  >>> from imblearn.under_sampling import RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE
     |  >>> X, y = make_classification(n_classes=2, class_sep=2,
     |  ...  weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
     |  ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
     |  >>> print('Original dataset shape {}'.format(Counter(y)))
     |  Original dataset shape Counter({1: 900, 0: 100})
     |  >>> rus = RandomUnderSampler(random_state=42)
     |  >>> X_res, y_res = rus.fit_sample(X, y)
     |  >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
     |  Resampled dataset shape Counter({0: 100, 1: 100})
     |  
     |  Method resolution order:
     |      RandomUnderSampler
     |      imblearn.under_sampling.prototype_selection.random_under_sampler.RandomUnderSampler
     |      imblearn.under_sampling.base.BaseUnderSampler
     |      imblearn.base.BaseSampler
     |      imblearn.base.SamplerMixin
     |      abc.NewBase
     |      sklearn.base.BaseEstimator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, ratio='auto', return_indices=False, random_state=None, replacement=False)
     |      Creates an object of the imblearn.under_sampling.RandomUnderSampler class.
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
     |      :param return_indices: bool, optional (default=False)
     |             Whether or not to return the indices of the samples randomly selected from the majority class.
     |      :param random_state: int, RandomState instance or None, optional (default=None)
     |             If int, random_state is the seed used by the random number generator; If RandomState instance,
     |             random_state is the random number generator; If None, the random number generator is the RandomState
     |             instance used by 'np.random'.
     |      :param replacement: boolean, optional (default=False)
     |             Whether the sample is with or without replacement
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
    
    class TomekLinks(imblearn.under_sampling.prototype_selection.tomek_links.TomekLinks)
     |  TomekLinks(ratio='auto', return_indices=False, random_state=None, n_jobs=1)
     |  
     |  Class to perform under-sampling by removing Tomek's links.
     |  
     |  Read more in the :ref:`User Guide <tomek_links>`.
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
     |      .. warning::
     |         This algorithm is a cleaning under-sampling method. When providing a
     |         ``dict``, only the targeted classes will be used; the number of
     |         samples will be discarded.
     |  
     |  return_indices : bool, optional (default=False)
     |      Whether or not to return the indices of the samples randomly
     |      selected from the majority class.
     |  
     |  random_state : int, RandomState instance or None, optional (default=None)
     |      If int, ``random_state`` is the seed used by the random number
     |      generator; If ``RandomState`` instance, random_state is the random
     |      number generator; If ``None``, the random number generator is the
     |      ``RandomState`` instance used by ``np.random``.
     |  
     |  n_jobs : int, optional (default=1)
     |      The number of threads to open if possible.
     |  
     |  Notes
     |  -----
     |  This method is based on [1]_.
     |  
     |  Supports mutli-class resampling. A one-vs.-rest scheme is used as
     |  originally proposed in [1]_.
     |  
     |  See
     |  :ref:`sphx_glr_auto_examples_under-sampling_plot_tomek_links.py`.
     |  
     |  References
     |  ----------
     |  .. [1] I. Tomek, "Two modifications of CNN," In Systems, Man, and
     |     Cybernetics, IEEE Transactions on, vol. 6, pp 769-772, 2010.
     |  
     |  Examples
     |  --------
     |  
     |  >>> from collections import Counter
     |  >>> from sklearn.datasets import make_classification
     |  >>> from imblearn.under_sampling import TomekLinks # doctest: +NORMALIZE_WHITESPACE
     |  >>> X, y = make_classification(n_classes=2, class_sep=2,
     |  ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
     |  ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
     |  >>> print('Original dataset shape {}'.format(Counter(y)))
     |  Original dataset shape Counter({1: 900, 0: 100})
     |  >>> tl = TomekLinks(random_state=42)
     |  >>> X_res, y_res = tl.fit_sample(X, y)
     |  >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
     |  Resampled dataset shape Counter({1: 897, 0: 100})
     |  
     |  Method resolution order:
     |      TomekLinks
     |      imblearn.under_sampling.prototype_selection.tomek_links.TomekLinks
     |      imblearn.under_sampling.base.BaseCleaningSampler
     |      imblearn.base.BaseSampler
     |      imblearn.base.SamplerMixin
     |      abc.NewBase
     |      sklearn.base.BaseEstimator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, ratio='auto', return_indices=False, random_state=None, n_jobs=1)
     |      Creates an object of the imblearn.under_sampling.TomekLinks class.
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
     |      :param return_indices: bool, optional (default=False)
     |             Whether or not to return the indices of the samples randomly selected from the majority class.
     |      :param random_state: int, RandomState instance or None, optional (default=None)
     |             If int, random_state is the seed used by the random number generator; If RandomState instance,
     |             random_state is the random number generator; If None, the random number generator is the RandomState
     |             instance used by 'np.random'.
     |      :param n_jobs: int, optional (default=1)
     |             The number of threads to open if possible.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __abstractmethods__ = frozenset()
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from imblearn.under_sampling.prototype_selection.tomek_links.TomekLinks:
     |  
     |  is_tomek(y, nn_index, class_type)
     |      is_tomek uses the target vector and the first neighbour of every
     |      sample point and looks for Tomek pairs. Returning a boolean vector with
     |      True for majority Tomek links.
     |      
     |      Parameters
     |      ----------
     |      y : ndarray, shape (n_samples, )
     |          Target vector of the data set, necessary to keep track of whether a
     |          sample belongs to minority or not
     |      
     |      nn_index : ndarray, shape (len(y), )
     |          The index of the closes nearest neighbour to a sample point.
     |      
     |      class_type : int or str
     |          The label of the minority class.
     |      
     |      Returns
     |      -------
     |      is_tomek : ndarray, shape (len(y), )
     |          Boolean vector on len( # samples ), with True for majority samples
     |          that are Tomek links.
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
    __all__ = ['ClusterCentroids', 'RandomUnderSampler', 'TomekLinks', 'Ed...

FILE
    /Users/georgiakapatai/Dropbox/Courses/MScBirkbeck/FinalProject/Project/MaatPy/maatpy/samplers/undersampling.py


