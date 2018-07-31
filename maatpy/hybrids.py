import logging

from imblearn.combine import SMOTEENN, SMOTETomek


class SMOTEENN(SMOTEENN):
    """
    Class to perform over-sampling using SMOTE and cleaning using ENN.
    Combine over- and under-sampling using SMOTE and Edited Nearest Neighbours.

    Inherits from imblearn.combine.SMOTEENN
    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 smote=None,
                 enn=None):
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
        :param smote: object, optional (default=SMOTE())
               The :class: imblearn.over_sampling.SMOTE object to use. If none provide a
               :class: imblearn.over_sampling.SMOTE object with default parameters will be given.
        :param enn: object, optional (default=EditedNearestNeighbours())
               The :class: imblearn.under_sampling.EditedNearestNeighbours object to use. If none provide a
               :class: imblearn.under_sampling.EditedNearestNeighbours object with default parameters will be given.
        """
        super(SMOTEENN, self).__init__()
        self.ratio = ratio
        self.random_state = random_state
        self.smote = smote
        self.enn = enn
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y):
        """

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the data which have to be sampled.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :return: X_resampled, y_resampled
        """
        return super().fit(X, y)

    def _sample(self, X, y):
        """
        Edited to apply ENN first to remove problematic samples and then apply SMOTE.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the data which have to be sampled.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :return: X_resampled, y_resampled
        """
        super()._validate_estimator()

        X_res, y_res = self.enn_.fit_sample(X, y)

        return self.smote_.fit_sample(X_res, y_res)


class SMOTETomek(SMOTETomek):
    """
    Class to perform over-sampling using SMOTE and cleaning using Tomek links.
    Combine over- and under-sampling using SMOTE and Tomek links.

    Inherits from imblearn.combine.SMOTETomek
    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 smote=None,
                 tomek=None):
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
        :param smote: object, optional (default=SMOTE())
               The :class: imblearn.over_sampling.SMOTE object to use. If none provide a
               :class: imblearn.over_sampling.SMOTE object with default parameters will be given.
        :param tomek: object, optional (default=TomekLinks())
               The :class: imblearn.under_sampling.TomekLinks object to use. If none provide a
               :class: imblearn.under_sampling.TomekLinks object with default parameters will be given.
        """
        super(SMOTETomek, self).__init__()
        self.ratio = ratio
        self.random_state = random_state
        self.smote = smote
        self.tomek = tomek
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y):
        """

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the data which have to be sampled.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :return: X_resampled, y_resampled
        """
        return super().fit(X, y)

    def _sample(self, X, y):
        """
        Edited to apply TomekLinks first to remove problematic samples and then apply SMOTE.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the data which have to be sampled.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :return: X_resampled, y_resampled
        """
        super()._validate_estimator()

        X_res, y_res = self.tomek_.fit_sample(X, y)

        return self.smote_.fit_sample(X_res, y_res)
