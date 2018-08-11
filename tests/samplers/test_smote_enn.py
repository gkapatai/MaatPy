"""Test the module SMOTE ENN."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import print_function

import numpy as np
from pytest import raises

from sklearn.utils.testing import assert_allclose, assert_array_equal

from maatpy.samplers.combination import SMOTEENN
from maatpy.samplers.undersampling import EditedNearestNeighbours
from maatpy.samplers.oversampling import SMOTE

RND_SEED = 0
X = np.array([[-0.8201808, 0.80937119], [-0.1586601, 0.00971665],
              [-1.12007992, 1.18823115], [-0.99029554, 1.02569057],
              [-2.70373622, 3.12619572], [-0.96328176, 0.35748681],
              [-0.31545331, 0.1964533], [0.86136371, 0.71603845],
              [0.07882574, -0.3018894], [-1.0067224, 1.07999716],
              [2.37872741, 0.71035727], [-0.25150486, 0.10763918],
              [-1.97515949, 2.24068106], [-0.6266499, 0.4518563],
              [0.95852386, 1.05044728], [-1.11412465, 1.18655987],
              [0.50844982, 0.82558207], [2.21093052, 0.97568314],
              [1.21152907, 0.93294954], [-0.79620121, 0.78554848]])

Y = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 1e-4


def test_sample_regular():
    smote = SMOTEENN(random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_sample(X, Y)

    X_gt = np.array([[-0.8201808,  0.80937119],
                     [-0.1586601,  0.00971665],
                     [-1.12007992,  1.18823115],
                     [-0.99029554,  1.02569057],
                     [-2.70373622,  3.12619572],
                     [-0.31545331,  0.1964533],
                     [0.07882574, -0.3018894],
                     [-1.0067224,  1.07999716],
                     [-0.25150486,  0.10763918],
                     [-1.97515949,  2.24068106],
                     [-0.79620121,  0.78554848],
                     [0.86136371,  0.71603845],
                     [2.37872741,  0.71035727],
                     [0.95852386,  1.05044728],
                     [0.50844982,  0.82558207],
                     [1.21152907,  0.93294954],
                     [0.99716997,  1.0324997],
                     [1.43255449,  0.71389985],
                     [0.89871029,  0.84457909],
                     [2.03144551,  0.77658617],
                     [0.54832354,  0.8316712],
                     [2.06048344,  0.77104845]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_half():
    ratio = 0.8
    smote = SMOTEENN(ratio=ratio, random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_sample(X, Y)

    X_gt = np.array([[-0.8201808 ,  0.80937119],
       [-0.1586601 ,  0.00971665],
       [-1.12007992,  1.18823115],
       [-0.99029554,  1.02569057],
       [-2.70373622,  3.12619572],
       [-0.31545331,  0.1964533 ],
       [ 0.07882574, -0.3018894 ],
       [-1.0067224 ,  1.07999716],
       [-0.25150486,  0.10763918],
       [-1.97515949,  2.24068106],
       [-0.79620121,  0.78554848],
       [ 0.86136371,  0.71603845],
       [ 2.37872741,  0.71035727],
       [ 0.95852386,  1.05044728],
       [ 0.50844982,  0.82558207],
       [ 1.21152907,  0.93294954],
       [ 1.0590268 ,  1.00377287],
       [ 1.55194145,  0.71345285],
       [ 0.90252607,  0.85771236]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    assert_allclose(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_init():
    smote = SMOTE(random_state=RND_SEED, ratio='auto', k_neighbors=3,)
    enn = EditedNearestNeighbours(random_state=RND_SEED, ratio='all', kind_sel="mode")
    smt = SMOTEENN(smote=smote, enn=enn, random_state=RND_SEED)
    X_resampled, y_resampled = smt.fit_sample(X, Y)
    X_gt = np.array([[-0.8201808,  0.80937119],
                     [-0.1586601,  0.00971665],
                     [-1.12007992,  1.18823115],
                     [-0.99029554,  1.02569057],
                     [-2.70373622,  3.12619572],
                     [-0.31545331,  0.1964533],
                     [0.07882574, -0.3018894],
                     [-1.0067224,  1.07999716],
                     [-0.25150486,  0.10763918],
                     [-1.97515949,  2.24068106],
                     [-0.79620121,  0.78554848],
                     [0.86136371,  0.71603845],
                     [2.37872741,  0.71035727],
                     [0.95852386,  1.05044728],
                     [0.50844982,  0.82558207],
                     [1.21152907,  0.93294954],
                     [0.99716997,  1.0324997],
                     [1.43255449,  0.71389985],
                     [0.89871029,  0.84457909],
                     [2.03144551,  0.77658617],
                     [0.54832354,  0.8316712],
                     [2.06048344,  0.77104845]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_default():
    smt = SMOTEENN(random_state=RND_SEED)
    X_resampled, y_resampled = smt.fit_sample(X, Y)
    X_gt = np.array([[-0.8201808,  0.80937119],
                     [-0.1586601,  0.00971665],
                     [-1.12007992,  1.18823115],
                     [-0.99029554,  1.02569057],
                     [-2.70373622,  3.12619572],
                     [-0.31545331,  0.1964533],
                     [0.07882574, -0.3018894],
                     [-1.0067224,  1.07999716],
                     [-0.25150486,  0.10763918],
                     [-1.97515949,  2.24068106],
                     [-0.79620121,  0.78554848],
                     [0.86136371,  0.71603845],
                     [2.37872741,  0.71035727],
                     [0.95852386,  1.05044728],
                     [0.50844982,  0.82558207],
                     [1.21152907,  0.93294954],
                     [0.99716997,  1.0324997],
                     [1.43255449,  0.71389985],
                     [0.89871029,  0.84457909],
                     [2.03144551,  0.77658617],
                     [0.54832354,  0.8316712],
                     [2.06048344,  0.77104845]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_deprecation():
    smt = SMOTEENN(random_state=RND_SEED)
    X_resampled, y_resampled = smt.fit_sample(X, Y)
    X_gt = np.array([[-0.8201808,  0.80937119],
                     [-0.1586601,  0.00971665],
                     [-1.12007992,  1.18823115],
                     [-0.99029554,  1.02569057],
                     [-2.70373622,  3.12619572],
                     [-0.31545331,  0.1964533],
                     [0.07882574, -0.3018894],
                     [-1.0067224,  1.07999716],
                     [-0.25150486,  0.10763918],
                     [-1.97515949,  2.24068106],
                     [-0.79620121,  0.78554848],
                     [0.86136371,  0.71603845],
                     [2.37872741,  0.71035727],
                     [0.95852386,  1.05044728],
                     [0.50844982,  0.82558207],
                     [1.21152907,  0.93294954],
                     [0.99716997,  1.0324997],
                     [1.43255449,  0.71389985],
                     [0.89871029,  0.84457909],
                     [2.03144551,  0.77658617],
                     [0.54832354,  0.8316712],
                     [2.06048344,  0.77104845]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)
    # parameter k was deprecated in the imblearn.combine.SMOTEENN class so was not included in this class
    # smt = SMOTEENN(random_state=RND_SEED, k=5)
    # X_resampled, y_resampled = smt.fit_sample(X, Y)
    # assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    # assert_array_equal(y_resampled, y_gt)


def test_error_wrong_object():
    smote = 'rnd'
    enn = 'rnd'
    smt = SMOTEENN(smote=smote, random_state=RND_SEED)
    with raises(ValueError, match="smote needs to be a SMOTE"):
        smt.fit_sample(X, Y)
    smt = SMOTEENN(enn=enn, random_state=RND_SEED)
    with raises(ValueError, match="enn needs to be an "):
        smt.fit_sample(X, Y)
