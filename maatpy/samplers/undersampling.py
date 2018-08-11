from imblearn.under_sampling import (ClusterCentroids,
                                     RandomUnderSampler,
                                     TomekLinks,
                                     EditedNearestNeighbours)

__all__ = ['ClusterCentroids', 'RandomUnderSampler', 'TomekLinks', 'EditedNearestNeighbours']


class ClusterCentroids(ClusterCentroids):

    def __init__(self, ratio='auto', random_state=None, estimator=None, voting='auto', n_jobs=1):
        """
        Creates an object of the imblearn.under_sampling.ClusterCentroids class.

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
        super(ClusterCentroids, self).__init__(ratio=ratio,
                                               random_state=random_state,
                                               estimator=estimator,
                                               voting=voting,
                                               n_jobs=n_jobs)


class RandomUnderSampler(RandomUnderSampler):

    def __init__(self, ratio='auto', return_indices=False, random_state=None, replacement=False):
        """
        Creates an object of the imblearn.under_sampling.RandomUnderSampler class.

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
        super(RandomUnderSampler, self).__init__(ratio=ratio,
                                                 return_indices=return_indices,
                                                 random_state=random_state,
                                                 replacement=replacement)


class TomekLinks(TomekLinks):

    def __init__(self, ratio='auto', return_indices=False, random_state=None, n_jobs=1):
        """
        Creates an object of the imblearn.under_sampling.TomekLinks class.

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
        super(TomekLinks, self).__init__(ratio=ratio,
                                         return_indices=return_indices,
                                         random_state=random_state,
                                         n_jobs=n_jobs)


class EditedNearestNeighbours(EditedNearestNeighbours):

    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 n_neighbors=3,
                 kind_sel='all',
                 n_jobs=1):
        """
        Creates an object of the imblearn.under_sampling.EditedNearestNeighbours class.

        :param ratio:  str, dict, or callable, optional (default='auto')
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
        :param n_neighbors: int or object, optional (default=3)
               If ``int``, size of the neighbourhood to consider to compute the nearest neighbors.
               If object, an estimator that inherits from :class:'sklearn.neighbors.base.KNeighborsMixin'
                that will be used to find the nearest-neighbors.
        :param kind_sel: str, optional (default='all')
               Strategy to use in order to exclude samples.
               - If "all", all neighbours will have to agree with the samples of interest to not be excluded.
               - If "mode", the majority vote of the neighbours will be used in order to exclude a sample.
        :param n_jobs: int, optional (default=1)
               The number of threads to open if possible.
        """
        super(EditedNearestNeighbours, self).__init__(ratio=ratio,
                                                      return_indices=return_indices,
                                                      random_state=random_state,
                                                      n_neighbors=n_neighbors,
                                                      kind_sel=kind_sel,
                                                      n_jobs=n_jobs)
