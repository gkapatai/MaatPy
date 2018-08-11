from imblearn.over_sampling import (RandomOverSampler,
                                    SMOTE)

__all__ = ['RandomOverSampler', 'SMOTE']


class RandomOverSampler(RandomOverSampler):

    def __init__(self, ratio='auto', random_state=None):
        """
        Creates an object of the imblearn.over_sampling.RandomOverSampler class.

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
        """
        super(RandomOverSampler, self).__init__(ratio=ratio,
                                                random_state=random_state)


class SMOTE(SMOTE):

    def __init__(self, ratio='auto', random_state=None, k_neighbors=5, m_neighbors=10, out_step=0.5,
                 kind='regular', svm_estimator=None, n_jobs=1):
        """
        Creates an object of imblearn.under_sampling.SMOTE class.

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
        :param k_neighbors: int or object,, optional (default=5)
               Number of nearest neighbours to used to construct synthetic samples. If object, an estimator that
               inherits from :class:'sklearn.neighbors.base.KNeighborsMixin' that will be used to find the k_neighbors.
        :param m_neighbors: int or object, optional (default=10)
               If int, number of nearest neighbours to use to determine if a minority sample is in danger.
               Used with "kind={'borderline1', 'borderline2', 'svm'}".  If object, an estimator that inherits from
               :class:'sklearn.neighbors.base.KNeighborsMixin' that will be used to find the k_neighbors.
        :param out_step: float, optional (default=0.5)
               Step size when extrapolating. Used with "kind='svm'".
        :param kind: str, optional (default='regular')
               The type of SMOTE algorithm to use one of the following options: 'regular', 'borderline1',
               'borderline2', 'svm'.
        :param svm_estimator: object, optional (default=SVC())
               If "kind='svm'", a parametrized :class:'sklearn.svm.SVC' classifier can be passed.
        :param n_jobs: int, optional (default=1)
               The number of threads to open if possible.
        """
        super(SMOTE, self).__init__(ratio=ratio,
                                    random_state=random_state,
                                    k_neighbors=k_neighbors,
                                    m_neighbors=m_neighbors,
                                    out_step=out_step,
                                    kind=kind,
                                    svm_estimator=svm_estimator,
                                    n_jobs=n_jobs)
