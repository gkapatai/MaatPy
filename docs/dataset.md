Help on module dataset:

NAME
    dataset

CLASSES
    sklearn.utils.Bunch(builtins.dict)
        Dataset
    
    class Dataset(sklearn.utils.Bunch)
     |  Dataset(data=None, target=None, feature_names=None, target_names=None)
     |  
     |  Container object for datasets
     |  
     |  Dictionary-like object that exposes its keys as attributes.
     |  
     |  >>> b = Bunch(a=1, b=2)
     |  >>> b['b']
     |  2
     |  >>> b.b
     |  2
     |  >>> b.a = 3
     |  >>> b['a']
     |  3
     |  >>> b.c = 6
     |  >>> b['c']
     |  6
     |  
     |  Method resolution order:
     |      Dataset
     |      sklearn.utils.Bunch
     |      builtins.dict
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, data=None, target=None, feature_names=None, target_names=None)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  load_from_csv(self, filename, sep=',', output_column=None)
     |      :param filename: path to filename containing the data to load
     |      :param sep: field separator; default ','
     |      :param output_column: column containing the outcome
     |      :return:
     |  
     |  make_imbalance(self, ratio=None, random_state=None)
     |      Built on the imblearn.make_imbalance function
     |      :param ratio: dict or list
     |             Ratio to use for resampling the data set.
     |             - When 'dict', the keys correspond to the targeted classes. The values correspond to the desired number
     |               of samples for each targeted class.
     |             - When 'list', the values correspond to the proportions of samples (float) assigned to each class. In
     |               this case the number of samples is maintained but the samples per class are adjusted to the given
     |               proportions.
     |      :param random_state: int, RandomState instance or None, optional (default=None)
     |             If int, random_state is the seed used by the random number generator; If RandomState instance,
     |             random_state is the random number generator; If None, the random number generator is the RandomState
     |             instance used by `np.random`.
     |      :return:
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.utils.Bunch:
     |  
     |  __dir__(self)
     |      Default dir() implementation.
     |  
     |  __getattr__(self, key)
     |  
     |  __setattr__(self, key, value)
     |      Implement setattr(self, name, value).
     |  
     |  __setstate__(self, state)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from sklearn.utils.Bunch:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from builtins.dict:
     |  
     |  __contains__(self, key, /)
     |      True if the dictionary has the specified key, else False.
     |  
     |  __delitem__(self, key, /)
     |      Delete self[key].
     |  
     |  __eq__(self, value, /)
     |      Return self==value.
     |  
     |  __ge__(self, value, /)
     |      Return self>=value.
     |  
     |  __getattribute__(self, name, /)
     |      Return getattr(self, name).
     |  
     |  __getitem__(...)
     |      x.__getitem__(y) <==> x[y]
     |  
     |  __gt__(self, value, /)
     |      Return self>value.
     |  
     |  __iter__(self, /)
     |      Implement iter(self).
     |  
     |  __le__(self, value, /)
     |      Return self<=value.
     |  
     |  __len__(self, /)
     |      Return len(self).
     |  
     |  __lt__(self, value, /)
     |      Return self<value.
     |  
     |  __ne__(self, value, /)
     |      Return self!=value.
     |  
     |  __repr__(self, /)
     |      Return repr(self).
     |  
     |  __setitem__(self, key, value, /)
     |      Set self[key] to value.
     |  
     |  __sizeof__(...)
     |      D.__sizeof__() -> size of D in memory, in bytes
     |  
     |  clear(...)
     |      D.clear() -> None.  Remove all items from D.
     |  
     |  copy(...)
     |      D.copy() -> a shallow copy of D
     |  
     |  get(self, key, default=None, /)
     |      Return the value for key if key is in the dictionary, else default.
     |  
     |  items(...)
     |      D.items() -> a set-like object providing a view on D's items
     |  
     |  keys(...)
     |      D.keys() -> a set-like object providing a view on D's keys
     |  
     |  pop(...)
     |      D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
     |      If key is not found, d is returned if given, otherwise KeyError is raised
     |  
     |  popitem(...)
     |      D.popitem() -> (k, v), remove and return some (key, value) pair as a
     |      2-tuple; but raise KeyError if D is empty.
     |  
     |  setdefault(self, key, default=None, /)
     |      Insert key with a value of default if key is not in the dictionary.
     |      
     |      Return the value for key if key is in the dictionary, else default.
     |  
     |  update(...)
     |      D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
     |      If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
     |      If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
     |      In either case, this is followed by: for k in F:  D[k] = F[k]
     |  
     |  values(...)
     |      D.values() -> an object providing a view on D's values
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from builtins.dict:
     |  
     |  fromkeys(iterable, value=None, /) from builtins.type
     |      Create a new dictionary with keys from iterable and values set to value.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from builtins.dict:
     |  
     |  __new__(*args, **kwargs) from builtins.type
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from builtins.dict:
     |  
     |  __hash__ = None

FUNCTIONS
    simulate_dataset(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, random_state=None)
        Using sklearn.make_classification function to return a Dataset object
        :param n_samples: int, optional (default=100).
               The number of samples.
        :param n_features: int, optional (default=2)
               The total number of features. These comprise 'n_informative' informative features and 'n_redundant'
               redundant features.
        :param n_informative: int, optional (default=2)
               The number of informative features. Each class is composed of a number  of gaussian clusters each located
               around the vertices of a hypercube in a subspace of dimension 'n_informative'. For each cluster,
               informative features are drawn independently from  N(0, 1) and then randomly linearly combined within
               each cluster in order to add covariance. The clusters are then placed on the vertices of the hypercube.
        :param n_redundant: int, optional (default=0)
               The number of redundant features. These features are generated a random linear combinations of the
               informative features.
        :param n_classes: int, optional (default=2)
               The number of classes (or labels) of the classification problem.
        :param n_clusters_per_class: int, optional (default=1)
               The number of clusters per class.
        :param weights: list of floats or None (default=None)
               The proportions of samples assigned to each class. If None, then classes are balanced. Note that if
               'len(weights) == n_classes - 1' then the last class weight is automatically inferred. More than
               'n_samples' samples may be returned if the sum of `weights` exceeds 1.
        :param flip_y: float, optional (default=0.01)
               The fraction of samples whose class are randomly exchanged. Larger values introduce noise in the labels
               and make the classification task harder.
        :param class_sep: float, optional (default=1.0)
               The factor multiplying the hypercube size.  Larger values spread out the clusters/classes and make the
               classification task easier.
        :param random_state: int, RandomState instance or None, optional (default=None)
               If int, random_state is the seed used by the random number generator; If RandomState instance,
               random_state is the random number generator; If None, the random number generator is the RandomState
               instance used by `np.random`.
        :return: Dataset object

FILE
    /Users/georgiakapatai/Dropbox/Courses/MScBirkbeck/FinalProject/Project/MaatPy/maatpy/dataset.py


