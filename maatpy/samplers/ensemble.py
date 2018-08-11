import numpy as np
from imblearn.ensemble import EasyEnsemble
from sklearn.utils import check_random_state
from maatpy.samplers.undersampling import RandomUnderSampler
from imblearn.utils import check_ratio

__all__ = ['EasyEnsemble']


class EasyEnsemble(EasyEnsemble):
    """
    Create an ensemble sets by iteratively applying random under-sampling.
    This method iteratively select a random subset and make an ensemble of the
    different sets.

    Inherits from imblearn.ensemble.EasyEnsemble. Edited to:
    - output a single dataset instead of subsets which allows it to work with a classifier in make_pipeline
    - accept a dictionary as ratio; previously it would crash when one was given.
    """
    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 replacement=False,
                 n_subsets=10):
        """

        :param ratio: tr, dict, or callable, optional (default='auto')
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
        :param replacement:  bool, optional (default=False)
               Whether or not to sample randomly with replacement or not.
        :param n_subsets: int, optional (default=10)
               Number of subsets to generate.
        """
        super(EasyEnsemble, self).__init__(ratio=ratio,
                                           random_state=random_state)
        self.return_indices = return_indices
        self.replacement = replacement
        self.n_subsets = n_subsets

    def fit(self, X, y):
        """
        Find the classes statistics before performing sampling.
        Adapted to allow for ratio = 'dict' by recalculating the dict.values = dict.value / n_subsets.


        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the data which have to be sampled.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :return: object; Return self
        """
        super(EasyEnsemble, self).fit(X, y)
        if isinstance(self.ratio, dict):
            self.ratio = {k: v for k, v in self.ratio.iteritems()}
        self.ratio_ = check_ratio(self.ratio, y, 'under-sampling')
        return self

    def _sample(self, X, y):
        """
        Resample the dataset.
        Adapted to return a combined result instead of an a collection of the subset results.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the data which have to be sampled.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :return: X_resampled, y_resampled
        """
        random_state = check_random_state(self.random_state)

        X_resampled = []
        y_resampled = []
        if self.return_indices:
            idx_under = []

        for _ in range(self.n_subsets):
            rus = RandomUnderSampler(
                ratio=self.ratio_, return_indices=True,
                random_state=random_state,
                replacement=self.replacement)
            sel_x, sel_y, sel_idx = rus.fit_sample(X, y)
            X_resampled.append(sel_x)
            y_resampled.append(sel_y)
            if self.return_indices:
                idx_under.append(sel_idx)

        if self.return_indices:
            return (np.array(X_resampled), np.array(y_resampled),
                    np.array(idx_under))
        else:
            return np.array(X_resampled), np.array(y_resampled)
