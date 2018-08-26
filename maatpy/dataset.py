import warnings
import numpy as np
import pandas as pd

from collections import Counter

from sklearn.datasets import make_classification
from sklearn.utils import check_X_y
from sklearn.utils import Bunch
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling.prototype_selection import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


class Dataset(Bunch):

    def __init__(self, data=None, target=None, feature_names=None, target_names=None):
        """

        :param data:
        :param target:
        :param feature_names:
        :param target_names:
        """
        self.data = data
        self.target = target
        self.feature_names = feature_names
        self.target_names = target_names

    def make_imbalance(self, ratio=None, random_state=None):
        """
        Built on the imblearn.make_imbalance function
        :param ratio: dict or list
               Ratio to use for resampling the data set.
               - When 'dict', the keys correspond to the targeted classes. The values correspond to the desired number
                 of samples for each targeted class.
               - When 'list', the values correspond to the proportions of samples (float) assigned to each class. In
                 this case the number of samples is maintained but the samples per class are adjusted to the given
                 proportions.
        :param random_state: int, RandomState instance or None, optional (default=None)
               If int, random_state is the seed used by the random number generator; If RandomState instance,
               random_state is the random number generator; If None, the random number generator is the RandomState
               instance used by `np.random`.
        :return:
        """
        x, y = check_X_y(self.data, self.target)
        original_dataset_size = len(y)
        n_classes = len(self.target_names)

        if isinstance(ratio, dict):
            ratio_ = ratio

        elif isinstance(ratio, list):
            weights = ratio
            if len(weights) != n_classes:
                raise ValueError("{} classes available but only {} values provided".format(n_classes, len(weights)))
            ratio_ = {}
            for i in range(n_classes):
                ratio_[i] = int(round(weights[i] * original_dataset_size, 0))

        else:
            raise TypeError("Expected dict or list; {} provided".format(type(ratio)))

        if sum(ratio_.values()) < original_dataset_size:
            rus = RandomUnderSampler(ratio=ratio_, random_state=random_state)
            self.data, self.target = rus.fit_sample(x, y)

        elif sum(ratio_.values()) == original_dataset_size:
            original_distribution = Counter(y)
            interim_ratio = {}
            for key in ratio_:
                if ratio_[key] >= original_distribution[key]:
                    interim_ratio[key] = original_distribution[key]
                else:
                    interim_ratio[key] = ratio_[key]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rus = RandomUnderSampler(ratio=interim_ratio, random_state=random_state)
                x_int, y_int = rus.fit_sample(x, y)
            with warnings.catch_warnings():
                # Silencing RandomOverSampler UserWarning: After over-sampling, the number of samples in class A will
                # be larger than the number of samples in the majority class
                warnings.simplefilter("ignore")
                ros = RandomOverSampler(ratio=ratio_, random_state=random_state)
                self.data, self.target = ros.fit_sample(x_int, y_int)

        else:
            raise ValueError("The requested dataset cannot be larger than the original dataset")

    def load_from_csv(self, filename, sep=',', output_column=None, ignore=None):
        """

        :param filename: path to filename containing the data to load
        :param sep: field separator; default ','
        :param output_column: column containing the outcome
        :param ignore: column to remove from data; str or list
        :return:
        """
        df = pd.read_csv(filename, sep=sep)
        if output_column:
            le = LabelEncoder()
            le.fit(list(df[output_column]))
            self.target_names = le.classes_
            self.target = le.transform(list(df[output_column]))
            df.drop(output_column, axis=1, inplace=True)
        else:
            raise ValueError('Please define an output_column; column containing the class defined for each observation '
                             '(row)')
        if ignore is not None:
            df.drop(ignore, axis=1, inplace=True)
        self.feature_names = df.columns
        self.data = df.values


def simulate_dataset(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1,
                     weights=None, flip_y=0.01, class_sep=1.0, random_state=None):
    """
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
    """

    data, target = make_classification(n_samples=n_samples, n_features=n_features,
                                                 n_informative=n_informative, n_redundant=n_redundant,
                                                 n_classes=n_classes, n_clusters_per_class=n_clusters_per_class,
                                                 weights=weights, flip_y=flip_y, class_sep=class_sep,
                                                 random_state=random_state)
    feature_names = ['feature#{}'.format(i) for i in range(data.shape[1])]
    target_names = ['class#{}'.format(i) for i in np.unique(target)]

    return Dataset(data, target, feature_names, target_names)
