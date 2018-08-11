"""Test the module SMOTE ENN."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import print_function

import numpy as np
from pytest import raises

from sklearn.utils.testing import (assert_allclose,
                                   assert_array_equal)

from maatpy.samplers.combination import SMOTETomek
from maatpy.samplers.oversampling import SMOTE
from maatpy.samplers.undersampling import TomekLinks

RND_SEED = 0
X = np.array([[0.20622591, 0.0582794], [0.68481731, 0.51935141],
              [1.34192108, -0.13367336], [0.62366841, -0.21312976],
              [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
              [0.74680821, 1.63827342], [0.2184254, 0.24299982],
              [0.61472253, -0.82309052], [0.19893132, -0.47761769],
              [1.06514042, -0.0770537], [0.97407872, 0.44454207],
              [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
              [-0.27410027, -0.54194484], [0.8381014, 0.44085498],
              [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
              [-0.00288378, 0.84259929], [1.79580611, -0.02219234]])
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 1e-4


def test_sample_regular():
    smote = SMOTETomek(random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
    X_gt = np.array([[0.68481731,  0.51935141],
                     [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504],
                     [-0.37162401, -2.19400981],
                     [0.74680821,  1.63827342],
                     [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769],
                     [1.40301027, -0.83648734],
                     [-1.20515198, -1.02689695],
                     [-0.23374509,  0.18370049],
                     [-0.00288378,  0.84259929],
                     [1.79580611, -0.02219234],
                     [1.07298301, -0.29946782],
                     [0.95384033, -0.47721819],
                     [1.73033669, -0.15698016],
                     [1.28289364, -0.10574411]])
    y_gt = np.array([1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_half():
    ratio = 0.8
    smote = SMOTETomek(ratio=ratio, random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
    X_gt = np.array([[0.68481731,  0.51935141],
                     [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504],
                     [-0.37162401, -2.19400981],
                     [0.74680821,  1.63827342],
                     [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769],
                     [1.40301027, -0.83648734],
                     [-1.20515198, -1.02689695],
                     [-0.23374509,  0.18370049],
                     [-0.00288378,  0.84259929],
                     [1.79580611, -0.02219234],
                     [1.01584072, -0.28848764],
                     [1.04831868, -0.55278682]])
    y_gt = np.array([1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_init():
    smote = SMOTE(random_state=RND_SEED, k_neighbors=3, ratio="auto")
    tomek = TomekLinks(random_state=RND_SEED, ratio='all')
    smt = SMOTETomek(smote=smote, tomek=tomek, random_state=RND_SEED)
    X_resampled, y_resampled = smt.fit_sample(X, Y)
    X_gt = np.array([[0.68481731,  0.51935141],
                     [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504],
                     [-0.37162401, -2.19400981],
                     [0.74680821,  1.63827342],
                     [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769],
                     [1.40301027, -0.83648734],
                     [-1.20515198, -1.02689695],
                     [-0.23374509,  0.18370049],
                     [-0.00288378,  0.84259929],
                     [1.79580611, -0.02219234],
                     [1.07298301, -0.29946782],
                     [0.95384033, -0.47721819],
                     [1.73033669, -0.15698016],
                     [1.28289364, -0.10574411]])
    y_gt = np.array([1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_default():
    smt = SMOTETomek(random_state=RND_SEED)
    X_resampled, y_resampled = smt.fit_sample(X, Y)
    X_gt = np.array([[0.68481731,  0.51935141],
                     [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504],
                     [-0.37162401, -2.19400981],
                     [0.74680821,  1.63827342],
                     [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769],
                     [1.40301027, -0.83648734],
                     [-1.20515198, -1.02689695],
                     [-0.23374509,  0.18370049],
                     [-0.00288378,  0.84259929],
                     [1.79580611, -0.02219234],
                     [1.07298301, -0.29946782],
                     [0.95384033, -0.47721819],
                     [1.73033669, -0.15698016],
                     [1.28289364, -0.10574411]])
    y_gt = np.array([1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_deprecation():
    smt = SMOTETomek(random_state=RND_SEED)
    X_resampled, y_resampled = smt.fit_sample(X, Y)
    X_gt = np.array([[0.68481731,  0.51935141],
                     [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504],
                     [-0.37162401, -2.19400981],
                     [0.74680821,  1.63827342],
                     [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769],
                     [1.40301027, -0.83648734],
                     [-1.20515198, -1.02689695],
                     [-0.23374509,  0.18370049],
                     [-0.00288378,  0.84259929],
                     [1.79580611, -0.02219234],
                     [1.07298301, -0.29946782],
                     [0.95384033, -0.47721819],
                     [1.73033669, -0.15698016],
                     [1.28289364, -0.10574411]])
    y_gt = np.array([1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)
    # deprecated from the original imblearn.combine.SMOTETomek therefore the k parameter was not included in this
    # class
    # smt = SMOTETomek(random_state=RND_SEED, k=5)
    # X_resampled, y_resampled = smt.fit_sample(X, Y)
    # assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    # assert_array_equal(y_resampled, y_gt)


def test_error_wrong_object():
    smote = 'rnd'
    tomek = 'rnd'
    smt = SMOTETomek(smote=smote, random_state=RND_SEED)
    with raises(ValueError, match="smote needs to be a SMOTE"):
        smt.fit_sample(X, Y)
    smt = SMOTETomek(tomek=tomek, random_state=RND_SEED)
    with raises(ValueError, match="tomek needs to be a TomekLinks"):
        smt.fit_sample(X, Y)
