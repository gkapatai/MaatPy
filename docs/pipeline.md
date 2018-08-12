Help on module pipeline:

NAME
    pipeline

CLASSES
    imblearn.pipeline.Pipeline(sklearn.pipeline.Pipeline)
        Pipeline
    
    class Pipeline(imblearn.pipeline.Pipeline)
     |  Pipeline(steps, memory=None)
     |  
     |  Pipeline of transforms and resamples with a final estimator.
     |  
     |  Sequentially apply a list of transforms, samples and a final estimator.
     |  Intermediate steps of the pipeline must be transformers or resamplers,
     |  that is, they must implement fit, transform and sample methods.
     |  The final estimator only needs to implement fit.
     |  The transformers and samplers in the pipeline can be cached using
     |  ``memory`` argument.
     |  
     |  The purpose of the pipeline is to assemble several steps that can be
     |  cross-validated together while setting different parameters.
     |  For this, it enables setting parameters of the various steps using their
     |  names and the parameter name separated by a '__', as in the example below.
     |  
     |  Parameters
     |  ----------
     |  steps : list
     |      List of (name, transform) tuples (implementing
     |      fit/transform/fit_sample) that are chained, in the order in which they
     |      are chained, with the last object an estimator.
     |  
     |  memory : Instance of joblib.Memory or string, optional (default=None)
     |      Used to cache the fitted transformers of the pipeline. By default,
     |      no caching is performed. If a string is given, it is the path to
     |      the caching directory. Enabling caching triggers a clone of
     |      the transformers before fitting. Therefore, the transformer
     |      instance given to the pipeline cannot be inspected
     |      directly. Use the attribute ``named_steps`` or ``steps`` to
     |      inspect estimators within the pipeline. Caching the
     |      transformers is advantageous when fitting is time consuming.
     |  
     |  
     |  Attributes
     |  ----------
     |  named_steps : dict
     |      Read-only attribute to access any step parameter by user given name.
     |      Keys are step names and values are steps parameters.
     |  
     |  Notes
     |  -----
     |  See :ref:`sphx_glr_auto_examples_pipeline_plot_pipeline_classification.py`
     |  
     |  See also
     |  --------
     |  make_pipeline : helper function to make pipeline.
     |  
     |  Examples
     |  --------
     |  
     |  >>> from collections import Counter
     |  >>> from sklearn.datasets import make_classification
     |  >>> from sklearn.model_selection import train_test_split as tts
     |  >>> from sklearn.decomposition import PCA
     |  >>> from sklearn.neighbors import KNeighborsClassifier as KNN
     |  >>> from sklearn.metrics import classification_report
     |  >>> from imblearn.over_sampling import SMOTE
     |  >>> from imblearn.pipeline import Pipeline # doctest: +NORMALIZE_WHITESPACE
     |  >>> X, y = make_classification(n_classes=2, class_sep=2,
     |  ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
     |  ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
     |  >>> print('Original dataset shape {}'.format(Counter(y)))
     |  Original dataset shape Counter({1: 900, 0: 100})
     |  >>> pca = PCA()
     |  >>> smt = SMOTE(random_state=42)
     |  >>> knn = KNN()
     |  >>> pipeline = Pipeline([('smt', smt), ('pca', pca), ('knn', knn)])
     |  >>> X_train, X_test, y_train, y_test = tts(X, y, random_state=42)
     |  >>> pipeline.fit(X_train, y_train) # doctest: +ELLIPSIS
     |  Pipeline(...)
     |  >>> y_hat = pipeline.predict(X_test)
     |  >>> print(classification_report(y_test, y_hat))
     |               precision    recall  f1-score   support
     |  <BLANKLINE>
     |            0       0.87      1.00      0.93        26
     |            1       1.00      0.98      0.99       224
     |  <BLANKLINE>
     |  avg / total       0.99      0.98      0.98       250
     |  <BLANKLINE>
     |  
     |  Method resolution order:
     |      Pipeline
     |      imblearn.pipeline.Pipeline
     |      sklearn.pipeline.Pipeline
     |      sklearn.utils.metaestimators._BaseComposition
     |      abc.NewBase
     |      sklearn.base.BaseEstimator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, steps, memory=None)
     |      :param steps: list
     |             List of (name, transform) tuples (implementing fit/transform/fit_sample) that are chained,
     |             in the order in which they are chained, with the last object an estimator.
     |      :param memory: Instance of joblib.Memory or string, optional (default=None)
     |             Used to cache the fitted transformers of the pipeline. By default, no caching is performed. If a string
     |             is given, it is the path to the caching directory. Enabling caching triggers a clone of the transformers
     |             before fitting. Therefore, the transformer instance given to the pipeline cannot be inspected directly.
     |             Use the attribute "named_steps" or "steps" to inspect estimators within the pipeline. Caching the
     |             transformers is advantageous when fitting is time consuming.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __abstractmethods__ = frozenset()
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from imblearn.pipeline.Pipeline:
     |  
     |  decision_function(self, X)
     |      Apply transformers/samplers, and decision_function of the final
     |      estimator
     |      
     |      Parameters
     |      ----------
     |      X : iterable
     |          Data to predict on. Must fulfill input requirements of first step
     |          of the pipeline.
     |      
     |      Returns
     |      -------
     |      y_score : array-like, shape = [n_samples, n_classes]
     |  
     |  fit(self, X, y=None, **fit_params)
     |      Fit the model
     |      
     |      Fit all the transforms/samplers one after the other and
     |      transform/sample the data, then fit the transformed/sampled
     |      data using the final estimator.
     |      
     |      Parameters
     |      ----------
     |      X : iterable
     |          Training data. Must fulfill input requirements of first step of the
     |          pipeline.
     |      
     |      y : iterable, default=None
     |          Training targets. Must fulfill label requirements for all steps of
     |          the pipeline.
     |      
     |      **fit_params : dict of string -> object
     |          Parameters passed to the ``fit`` method of each step, where
     |          each parameter name is prefixed such that parameter ``p`` for step
     |          ``s`` has key ``s__p``.
     |      
     |      Returns
     |      -------
     |      self : Pipeline
     |          This estimator
     |  
     |  fit_predict(self, X, y=None, **fit_params)
     |      Applies fit_predict of last step in pipeline after transforms.
     |      
     |      Applies fit_transforms of a pipeline to the data, followed by the
     |      fit_predict method of the final estimator in the pipeline. Valid
     |      only if the final estimator implements fit_predict.
     |      
     |      Parameters
     |      ----------
     |      X : iterable
     |          Training data. Must fulfill input requirements of first step of
     |          the pipeline.
     |      
     |      y : iterable, default=None
     |          Training targets. Must fulfill label requirements for all steps
     |          of the pipeline.
     |      
     |      **fit_params : dict of string -> object
     |          Parameters passed to the ``fit`` method of each step, where
     |          each parameter name is prefixed such that parameter ``p`` for step
     |          ``s`` has key ``s__p``.
     |      
     |      Returns
     |      -------
     |      y_pred : array-like
     |  
     |  fit_sample(self, X, y=None, **fit_params)
     |      Fit the model and sample with the final estimator
     |      
     |      Fits all the transformers/samplers one after the other and
     |      transform/sample the data, then uses fit_sample on transformed
     |      data with the final estimator.
     |      
     |      Parameters
     |      ----------
     |      X : iterable
     |          Training data. Must fulfill input requirements of first step of the
     |          pipeline.
     |      
     |      y : iterable, default=None
     |          Training targets. Must fulfill label requirements for all steps of
     |          the pipeline.
     |      
     |      **fit_params : dict of string -> object
     |          Parameters passed to the ``fit`` method of each step, where
     |          each parameter name is prefixed such that parameter ``p`` for step
     |          ``s`` has key ``s__p``.
     |      
     |      Returns
     |      -------
     |      Xt : array-like, shape = [n_samples, n_transformed_features]
     |          Transformed samples
     |      
     |      yt : array-like, shape = [n_samples, n_transformed_features]
     |          Transformed target
     |  
     |  fit_transform(self, X, y=None, **fit_params)
     |      Fit the model and transform with the final estimator
     |      
     |      Fits all the transformers/samplers one after the other and
     |      transform/sample the data, then uses fit_transform on
     |      transformed data with the final estimator.
     |      
     |      Parameters
     |      ----------
     |      X : iterable
     |          Training data. Must fulfill input requirements of first step of the
     |          pipeline.
     |      
     |      y : iterable, default=None
     |          Training targets. Must fulfill label requirements for all steps of
     |          the pipeline.
     |      
     |      **fit_params : dict of string -> object
     |          Parameters passed to the ``fit`` method of each step, where
     |          each parameter name is prefixed such that parameter ``p`` for step
     |          ``s`` has key ``s__p``.
     |      
     |      Returns
     |      -------
     |      Xt : array-like, shape = [n_samples, n_transformed_features]
     |          Transformed samples
     |  
     |  predict(self, X)
     |      Apply transformers/samplers to the data, and predict with the final
     |      estimator
     |      
     |      Parameters
     |      ----------
     |      X : iterable
     |          Data to predict on. Must fulfill input requirements of first step
     |          of the pipeline.
     |      
     |      Returns
     |      -------
     |      y_pred : array-like
     |  
     |  predict_log_proba(self, X)
     |      Apply transformers/samplers, and predict_log_proba of the final
     |      estimator
     |      
     |      Parameters
     |      ----------
     |      X : iterable
     |          Data to predict on. Must fulfill input requirements of first step
     |          of the pipeline.
     |      
     |      Returns
     |      -------
     |      y_score : array-like, shape = [n_samples, n_classes]
     |  
     |  predict_proba(self, X)
     |      Apply transformers/samplers, and predict_proba of the final
     |      estimator
     |      
     |      Parameters
     |      ----------
     |      X : iterable
     |          Data to predict on. Must fulfill input requirements of first step
     |          of the pipeline.
     |      
     |      Returns
     |      -------
     |      y_proba : array-like, shape = [n_samples, n_classes]
     |  
     |  sample(self, X, y)
     |      Sample the data with the final estimator
     |      
     |      Applies transformers/samplers to the data, and the sample
     |      method of the final estimator. Valid only if the final
     |      estimator implements sample.
     |      
     |      Parameters
     |      ----------
     |      X : iterable
     |          Data to predict on. Must fulfill input requirements of first step
     |          of the pipeline.
     |  
     |  score(self, X, y=None, sample_weight=None)
     |      Apply transformers/samplers, and score with the final estimator
     |      
     |      Parameters
     |      ----------
     |      X : iterable
     |          Data to predict on. Must fulfill input requirements of first step
     |          of the pipeline.
     |      
     |      y : iterable, default=None
     |          Targets used for scoring. Must fulfill label requirements for all
     |          steps of the pipeline.
     |      
     |      sample_weight : array-like, default=None
     |          If not None, this argument is passed as ``sample_weight`` keyword
     |          argument to the ``score`` method of the final estimator.
     |      
     |      Returns
     |      -------
     |      score : float
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from imblearn.pipeline.Pipeline:
     |  
     |  inverse_transform
     |      Apply inverse transformations in reverse order
     |      
     |      All estimators in the pipeline must support ``inverse_transform``.
     |      
     |      Parameters
     |      ----------
     |      Xt : array-like, shape = [n_samples, n_transformed_features]
     |          Data samples, where ``n_samples`` is the number of samples and
     |          ``n_features`` is the number of features. Must fulfill
     |          input requirements of last step of pipeline's
     |          ``inverse_transform`` method.
     |      
     |      Returns
     |      -------
     |      Xt : array-like, shape = [n_samples, n_features]
     |  
     |  transform
     |      Apply transformers/samplers, and transform with the final estimator
     |      
     |      This also works where final estimator is ``None``: all prior
     |      transformations are applied.
     |      
     |      Parameters
     |      ----------
     |      X : iterable
     |          Data to transform. Must fulfill input requirements of first step
     |          of the pipeline.
     |      
     |      Returns
     |      -------
     |      Xt : array-like, shape = [n_samples, n_transformed_features]
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.pipeline.Pipeline:
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
     |  set_params(self, **kwargs)
     |      Set the parameters of this estimator.
     |      
     |      Valid parameter keys can be listed with ``get_params()``.
     |      
     |      Returns
     |      -------
     |      self
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from sklearn.pipeline.Pipeline:
     |  
     |  classes_
     |  
     |  named_steps
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
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from sklearn.base.BaseEstimator:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FUNCTIONS
    make_pipeline(*steps, **kwargs)
        Construct a Pipeline from the given estimators.
        This is a shorthand for the Pipeline constructor; it does not require, and does not permit, naming the estimators.
        Instead, their names will be set to the lowercase of their types automatically.
        
        :param steps: list
               List of (name, transform) tuples (implementing fit/transform/fit_sample) that are chained,
               in the order in which they are chained, with the last object an estimator.
        :param kwargs:
        :return: p: Pipeline

FILE
    /Users/georgiakapatai/Dropbox/Courses/MScBirkbeck/FinalProject/Project/MaatPy/maatpy/pipeline.py


