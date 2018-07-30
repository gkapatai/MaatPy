"""

Utility functions for checking ratio provided for over- and under-sampling

Adjusted from imblearn.utils.validation

"""
import warnings
from collections import Counter
from imblearn.utils.validation import _ratio_dict
from sklearn.externals import six

import numpy as np

SAMPLING_KIND = ('over-sampling', 'under-sampling', 'clean-sampling',
                 'ensemble')
TARGET_KIND = ('binary', 'multiclass')


def _ratio_majority(y, sampling_type):
    """
    Returns ratio by targeting the majority class only.
    Copied from imblearn.utils.validation to avoid calling protected function
    :param y: ndarray, shape (n_samples,)
           The target array.
    :param sampling_type: str,
           The type of sampling. Can be either 'over-sampling', 'under-sampling', 'clean-sampling' and 'ensemble'.
    :return: ratio_ [dict]
    """
    if sampling_type == 'over-sampling':
        raise ValueError("'ratio'='majority' cannot be used with"
                         " over-sampler.")
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        target_stats = Counter(y)
        class_majority = max(target_stats, key=target_stats.get)
        n_sample_minority = min(target_stats.values())
        ratio_ = {key: n_sample_minority
                 for key in target_stats.keys()
                 if key == class_majority}
    else:
        raise NotImplementedError

    return ratio_


def _ratio_dict(ratio, y, sampling_type):
    """
    Returns ratio by converting the dictionary depending of the sampling.
    Copied from imblearn.utils.validation to avoid calling protected function
    :param ratio: dict,
           Keys correspond to the targeted classes. The values correspond to the desired number of samples.
    :param y: ndarray, shape (n_samples,)
           The target array.
    :param sampling_type: str,
           The type of sampling. Can be either 'over-sampling', 'under-sampling', 'clean-sampling' and 'ensemble'.
    :return: ratio_ [dict]
    """
    target_stats = Counter(y)
    # check that all keys in ratio are also in y
    set_diff_ratio_target = set(ratio.keys()) - set(target_stats.keys())
    if len(set_diff_ratio_target) > 0:
        raise ValueError("The {} target class is/are not present in the"
                         " data.".format(set_diff_ratio_target))
    # check that there is no negative number
    if any(n_samples < 0 for n_samples in ratio.values()):
        raise ValueError("The number of samples in a class cannot be negative."
                         "'ratio' contains some negative value: {}".format(
                             ratio))
    ratio_ = {}
    if sampling_type == 'over-sampling':
        n_samples_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        for class_sample, n_samples in ratio.items():
            if n_samples < target_stats[class_sample]:
                raise ValueError("With over-sampling methods, the number"
                                 " of samples in a class should be greater"
                                 " or equal to the original number of samples."
                                 " Originally, there is {} samples and {}"
                                 " samples are asked.".format(
                                     target_stats[class_sample], n_samples))
            if n_samples > n_samples_majority:
                warnings.warn("After over-sampling, the number of samples ({})"
                              " in class {} will be larger than the number of"
                              " samples in the majority class (class #{} ->"
                              " {})".format(n_samples, class_sample,
                                            class_majority,
                                            n_samples_majority))
            ratio_[class_sample] = n_samples - target_stats[class_sample]
    elif sampling_type == 'under-sampling':
        for class_sample, n_samples in ratio.items():
            if n_samples > target_stats[class_sample]:
                raise ValueError("With under-sampling methods, the number of"
                                 " samples in a class should be less or equal"
                                 " to the original number of samples."
                                 " Originally, there is {} samples and {}"
                                 " samples are asked.".format(
                                     target_stats[class_sample], n_samples))
            ratio_[class_sample] = n_samples
    elif sampling_type == 'clean-sampling':
        # clean-sampling can be more permissive since those samplers do not
        # use samples
        for class_sample, n_samples in ratio.items():
            ratio_[class_sample] = n_samples
    else:
        raise NotImplementedError

    return ratio_


def _ratio_auto(y, sampling_type):
    """
    Returns ratio auto for over-sampling and not-minority for under-sampling.
    Copied from imblearn.utils.validation. Calls edited functions _ratio_all and _ratio_not_minority
    :param y: ndarray, shape (n_samples,)
           The target array.
    :param sampling_type: str,
           The type of sampling. Can be either 'over-sampling', 'under-sampling', 'clean-sampling' and 'ensemble'.
    :return: ratio_ [dict]
    """
    if sampling_type == 'over-sampling':
        return _ratio_all(y, sampling_type)
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        return _ratio_not_minority(y, sampling_type)


def check_ratio(ratio, y, sampling_type, **kwargs):
    """
    Ratio validation for samplers.

    Checks ratio for consistent type and return a dictionary containing each targeted class with its corresponding
    number of samples.

    :param ratio: str, dict, or callable, optional (default='auto')
           Ratio to use for resampling the data set.
           - If "str", has to be one of: (i) 'minority': resample the minority class;
             (ii) 'majority': resample the majority class,
             (iii) 'not minority': resample all classes apart of the minority class,
             (iv) 'all': resample all classes, and (v) 'auto': correspond to 'all' with for over-sampling methods and
             'not_minority' for under-sampling methods. The classes targeted will be over-sampled or under-sampled to
             achieve an equal number of sample with the majority or minority class.
           - If "dict`", the keys correspond to the targeted classes. The values correspond to the desired number
             of samples.
           - If callable, function taking "y" and returns a "dict". The keyS correspond to the targeted classes.
             The values correspond to the desired number of samples.
    :param y: ndarray, shape (n_samples,)
           The target array.
    :param sampling_type: str,
           The type of sampling. Can be either 'over-sampling', 'under-sampling', 'clean-sampling' and 'ensemble'.
    :param kwargs: dict, optional
           Dictionary of additional keyword arguments to pass to "ratio".
    :return: ratio_ [dict]
    """

    if sampling_type not in SAMPLING_KIND:
        raise ValueError("'sampling_type' should be one of {}. Got '{}'"
                         " instead.".format(SAMPLING_KIND, sampling_type))

    if np.unique(y).size <= 1:
        raise ValueError("The target 'y' needs to have more than 1 class."
                         " Got {} class instead".format(np.unique(y).size))

    if sampling_type == 'ensemble':
        return ratio

    if isinstance(ratio, six.string_types):
        if ratio not in RATIO_KIND.keys():
            raise ValueError("When 'ratio' is a string, it needs to be one of"
                             " {}. Got '{}' instead.".format(RATIO_KIND,
                                                             ratio))
        return RATIO_KIND[ratio](y, sampling_type)
    elif isinstance(ratio, dict):
        return _ratio_dict(ratio, y, sampling_type)
    elif callable(ratio):
        ratio_ = ratio(y, **kwargs)
        return _ratio_dict(ratio_, y, sampling_type)


RATIO_KIND = {'minority': _ratio_minority,
              'majority': _ratio_majority,
              'not minority': _ratio_not_minority,
              'all': _ratio_all,
              'auto': _ratio_auto}