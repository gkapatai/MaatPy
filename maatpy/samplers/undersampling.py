from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, TomekLinks
from imblearn.utils.validation import check_target_type, hash_X_y
from sklearn.utils import check_X_y
from maatpy.utils import check_ratio

__all__ = ['ClusterCentroids', 'RandomUnderSampler', 'TomekLinks']


class ClusterCentroids(ClusterCentroids):
    """
    Class to perform under-sampling by generating centroids based on clustering.

    This is just a dummy class that creates an object of the imblearn.under_sampling.ClusterCentroids class.
    """
    def __init__(self, ratio='auto', random_state=None, estimator=None, voting='auto', n_jobs=1):
        """

        :param ratio: str, dict, or callable, optional (default='auto')
               Ratio to use for resampling the data set.
               - If "str", has to be one of: (i) 'minority': resample the minority class;
                 (ii) 'majority': resample the majority class,
                 (iii) 'not minority': resample all classes apart of the minority class,
                 (iv) 'all': resample all classes, and (v) 'auto': correspond to 'all' with for over-sampling
                 methods and 'not_minority' for under-sampling methods. The classes targeted will be over-sampled or
                 under-sampled to achieve an equal number of sample with the majority or minority class.
               - If "dict`", the keys correspond to the targeted classes. The values correspond to the desired number
                 of samples.
               - If callable, function taking "y" and returns a "dict". The keys correspond to the targeted classes.
                 The values correspond to the desired number of samples.
        :param random_state: int, RandomState instance or None, optional (default=None)
               If int, random_state is the seed used by the random number generator; If RandomState instance,
               random_state is the random number generator; If None, the random number generator is the RandomState
               instance used by 'np.random'.
        :param estimator: object, optional(default=KMeans())
               Pass a :class:'sklearn.cluster.KMeans' estimator.
        :param voting: str, optional (default='auto')
               Voting strategy to generate the new samples:
               - If 'hard', the nearest-neighbors of the centroids found using the clustering algorithm will be used.
               - If 'soft', the centroids found by the clustering algorithm will be used.
               - If 'auto', if the input is sparse, it will default on 'hard' otherwise, 'soft' will be used.
        :param n_jobs: int, optional (default=1)
               The number of threads to open if possible.
        """
        super(ClusterCentroids, self).__init__(ratio=ratio, random_state=random_state)
        self.estimator = estimator
        self.voting = voting
        self.n_jobs = n_jobs


class RandomUnderSampler(RandomUnderSampler):
    """
    Class to perform random under-sampling by randomly picking samples with or without replacement.

    This is just a dummy class that creates an object of the imblearn.under_sampling.RandomUnderSampler class.
    """
    def __init__(self, ratio='auto', return_indices=False, random_state=None, replacement=False):
        """

        :param ratio: str, dict, or callable, optional (default='auto')
               Ratio to use for resampling the data set.
               - If "str", has to be one of: (i) 'minority': resample the minority class;
                 (ii) 'majority': resample the majority class,
                 (iii) 'not minority': resample all classes apart of the minority class,
                 (iv) 'all': resample all classes, and (v) 'auto': correspond to 'all' with for over-sampling
                 methods and 'not_minority' for under-sampling methods. The classes targeted will be over-sampled or
                 under-sampled to achieve an equal number of sample with the majority or minority class.
               - If "dict`", the keys correspond to the targeted classes. The values correspond to the desired number
                 of samples.
               - If callable, function taking "y" and returns a "dict". The keys correspond to the targeted classes.
                 The values correspond to the desired number of samples.
        :param return_indices: bool, optional (default=False)
               Whether or not to return the indices of the samples randomly selected from the majority class.
        :param random_state: int, RandomState instance or None, optional (default=None)
               If int, random_state is the seed used by the random number generator; If RandomState instance,
               random_state is the random number generator; If None, the random number generator is the RandomState
               instance used by 'np.random'.
        :param replacement: boolean, optional (default=False)
               Whether the sample is with or without replacement
        """
        super(RandomUnderSampler, self).__init__(ratio=ratio, random_state=random_state)
        self.return_indices = return_indices
        self.replacement = replacement


class TomekLinks(TomekLinks):
    """
    Class to perform under-sampling by removing Tomek's links.

    This is just a dummy class that creates an object of the imblearn.under_sampling.TomekLinks class.
    """
    def __init__(self, ratio='auto', return_indices=False, random_state=None, n_jobs=1):
        """

        :param ratio: str, dict, or callable, optional (default='auto')
               Ratio to use for resampling the data set.
               - If "str", has to be one of: (i) 'minority': resample the minority class;
                 (ii) 'majority': resample the majority class,
                 (iii) 'not minority': resample all classes apart of the minority class,
                 (iv) 'all': resample all classes, and (v) 'auto': correspond to 'all' with for over-sampling
                 methods and 'not_minority' for under-sampling methods. The classes targeted will be over-sampled or
                 under-sampled to achieve an equal number of sample with the majority or minority class.
               - If "dict`", the keys correspond to the targeted classes. The values correspond to the desired number
                 of samples.
               - If callable, function taking "y" and returns a "dict". The keys correspond to the targeted classes.
                 The values correspond to the desired number of samples.
        :param return_indices: bool, optional (default=False)
               Whether or not to return the indices of the samples randomly selected from the majority class.
        :param random_state: int, RandomState instance or None, optional (default=None)
               If int, random_state is the seed used by the random number generator; If RandomState instance,
               random_state is the random number generator; If None, the random number generator is the RandomState
               instance used by 'np.random'.
        :param n_jobs: int, optional (default=1)
               The number of threads to open if possible.
        """
        super(TomekLinks, self).__init__(ratio=ratio, random_state=random_state)
        self.return_indices = return_indices
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the data which have to be sampled.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :return: X_resampled, y_resampled
        """
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        y = check_target_type(y)
        self.X_hash_, self.y_hash_ = hash_X_y(X, y)
        self.ratio_ = check_ratio(self.ratio, y, self._sampling_type)

        return self
