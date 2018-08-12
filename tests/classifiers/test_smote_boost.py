"""Testing for the boost module (sklearn.ensemble.boost)."""

import numpy as np
from sklearn.utils.testing import (assert_array_equal,
                                   assert_array_less,
                                   assert_array_almost_equal,
                                   assert_equal,
                                   assert_true,
                                   assert_greater,
                                   assert_raises,
                                   assert_raises_regexp)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (AdaBoostClassifier,
                              weight_boosting)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn import datasets

from maatpy.classifiers import SMOTEBoost


# Common random state
rng = np.random.RandomState(0)

# Toy sample
X = [[-2, -1], [-1, -1], [-1, -2],
     [-2, -2], [-3, -1], [-3, -2],
     [-3, -3], [1, 1], [1, 2], [2, 1],
     [2, 2], [1, 3], [2, 3], [3, 3]]
y_class = ["foo", "foo", "foo", "foo",
           "foo", "foo", "foo", 1, 1, 1,
           1, 1, 1, 1]    # test string class labels
y_regr = [-1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
y_t_class = ["foo", 1, 1]

# Load the iris dataset and randomly permute it
iris = datasets.load_iris()
perm = rng.permutation(iris.target.size)
iris.data, iris.target = shuffle(iris.data, iris.target, random_state=rng)

# Load the boston dataset and randomly permute it
boston = datasets.load_boston()
boston.data, boston.target = shuffle(boston.data, boston.target,
                                     random_state=rng)


def test_samme_proba():
    # Test the `_samme_proba` helper function.

    # Define some example (bad) `predict_proba` output.
    probs = np.array([[1, 1e-6, 0],
                      [0.19, 0.6, 0.2],
                      [-999, 0.51, 0.5],
                      [1e-6, 1, 1e-9]])
    probs /= np.abs(probs.sum(axis=1))[:, np.newaxis]

    # _samme_proba calls estimator.predict_proba.
    # Make a mock object so I can control what gets returned.
    class MockEstimator(object):
        def predict_proba(self, X):
            assert_array_equal(X.shape, probs.shape)
            return probs
    mock = MockEstimator()

    samme_proba = weight_boosting._samme_proba(mock, 3, np.ones_like(probs))

    assert_array_equal(samme_proba.shape, probs.shape)
    assert_true(np.isfinite(samme_proba).all())

    # Make sure that the correct elements come out as smallest --
    # `_samme_proba` should preserve the ordering in each example.
    assert_array_equal(np.argmin(samme_proba, axis=1), [2, 0, 0, 2])
    assert_array_equal(np.argmax(samme_proba, axis=1), [0, 1, 1, 1])


def test_oneclass_adaboost_proba():
    # Test predict_proba robustness for one class label input.
    # In response to issue #7501
    # https://github.com/scikit-learn/scikit-learn/issues/7501
    y_t = np.ones(len(X))
    clf = SMOTEBoost().fit(X, y_t)
    assert_array_equal(clf.predict_proba(X), np.ones((len(X), 1)))


def test_classification_toy():
    # Check classification on a toy dataset.
    for alg in ['SAMME', 'SAMME.R']:
        clf = SMOTEBoost(algorithm=alg, k_neighbors=3, random_state=0)
        clf.fit(X, y_class)
        assert_array_equal(clf.predict(T), y_t_class)
        assert_array_equal(np.unique(np.asarray(y_t_class)), clf.classes_)
        assert_equal(clf.predict_proba(T).shape, (len(T), 2))
        assert_equal(clf.decision_function(T).shape, (len(T),))


def test_iris():
    # Check consistency on dataset iris.
    classes = np.unique(iris.target)
    clf_samme = prob_samme = None

    for alg in ['SAMME', 'SAMME.R']:
        clf = SMOTEBoost(algorithm=alg, random_state=0)
        clf.fit(iris.data, iris.target)

        assert_array_equal(classes, clf.classes_)
        proba = clf.predict_proba(iris.data)
        if alg == "SAMME":
            clf_samme = clf
            prob_samme = proba
        assert_equal(proba.shape[1], len(classes))
        assert_equal(clf.decision_function(iris.data).shape[1], len(classes))

        score = clf.score(iris.data, iris.target)
        assert score > 0.9, "Failed with algorithm %s and score = %f" % \
            (alg, score)

        # Check we used multiple estimators
        assert_greater(len(clf.estimators_), 1)
        # Check for distinct random states (see issue #7408)
        assert_equal(len(set(est.random_state for est in clf.estimators_)),
                     len(clf.estimators_))

    # Somewhat hacky regression test: prior to
    # ae7adc880d624615a34bafdb1d75ef67051b8200,
    # predict_proba returned SAMME.R values for SAMME.
    clf_samme.algorithm = "SAMME.R"
    assert_array_less(0,
                      np.abs(clf_samme.predict_proba(iris.data) - prob_samme))


def test_gridsearch():
    # Check that base trees can be grid-searched.
    # AdaBoost classification
    boost = SMOTEBoost(base_estimator=DecisionTreeClassifier())
    parameters = {'n_estimators': (1, 2),
                  'base_estimator__max_depth': (1, 2),
                  'algorithm': ('SAMME', 'SAMME.R')}
    clf = GridSearchCV(boost, parameters)
    clf.fit(iris.data, iris.target)


def test_pickle():
    # Check pickability.
    import pickle

    # Adaboost classifier
    for alg in ['SAMME', 'SAMME.R']:
        obj = SMOTEBoost(algorithm=alg)
        obj.fit(iris.data, iris.target)
        score = obj.score(iris.data, iris.target)
        s = pickle.dumps(obj)

        obj2 = pickle.loads(s)
        assert_equal(type(obj2), obj.__class__)
        score2 = obj2.score(iris.data, iris.target)
        assert_equal(score, score2)


def test_importances():
    # Check variable importances.
    X, y = datasets.make_classification(n_samples=2000,
                                        n_features=10,
                                        n_informative=3,
                                        n_redundant=0,
                                        n_repeated=0,
                                        shuffle=False,
                                        random_state=1)

    for alg in ['SAMME', 'SAMME.R']:
        clf = SMOTEBoost(algorithm=alg, random_state=0)

        clf.fit(X, y)
        importances = clf.feature_importances_

        assert_equal(importances.shape[0], 10)
        assert_equal((importances[:3, np.newaxis] >= importances[3:]).all(),
                     True)


def test_error():
    # Test that it gives proper exception on deficient input.
    assert_raises(ValueError,
                  SMOTEBoost(learning_rate=-1).fit,
                  X, y_class)

    assert_raises(ValueError,
                  SMOTEBoost(algorithm="foo").fit,
                  X, y_class)

    assert_raises(TypeError,
                  SMOTEBoost().fit,
                  X, y_class, sample_weight=np.asarray([-1]))


def test_base_estimator():
    # Test different base estimators.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    # XXX doesn't work with y_class because RF doesn't support classes_
    # Shouldn't AdaBoost run a LabelBinarizer?
    clf = SMOTEBoost(base_estimator=RandomForestClassifier(), k_neighbors=3, random_state=0)
    clf.fit(X, y_regr)

    clf = SMOTEBoost(base_estimator=SVC(), algorithm="SAMME", k_neighbors=3, random_state=0)
    clf.fit(X, y_class)

    # Check that an empty discrete ensemble fails in fit, not predict.
    X_fail = [[1, 1], [1, 1], [1, 1], [1, 1],
              [1, 1], [1, 1], [1, 1], [1, 1],
              [1, 1], [1, 1], [1, 1], [1, 1],
              [1, 1], [1, 1], [1, 1], [1, 1],
              [1, 1], [1, 1], [1, 1], [1, 1],
              [1, 1], [1, 1], [1, 1], [1, 1],
              [1, 1], [1, 1], [1, 1], [1, 1],
              [1, 1], [1, 1], [1, 1], [1, 1]]
    y_fail = ["foo", "foo", "foo", "foo", "foo", "foo", "foo", "foo",
              "bar", "bar", "bar", "bar", "bar", "bar", "bar", "bar",
              1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
    clf = SMOTEBoost(base_estimator=SVC(), k_neighbors=3, algorithm="SAMME", random_state=0)
    assert_raises_regexp(ValueError, "worse than random",
                         clf.fit, X_fail, y_fail)
    
    
def test_imb_performance():
    from maatpy.dataset import simulate_dataset
    from sklearn.metrics import cohen_kappa_score
    from sklearn.model_selection import StratifiedShuffleSplit
    imb = simulate_dataset(n_samples=100, n_features=2, n_classes=2, weights=[0.9, 0.1], random_state=0)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    sss.get_n_splits(imb.data, imb.target)
    for train_index, test_index in sss.split(imb.data, imb.target):
        X_train, X_test = imb.data[train_index], imb.data[test_index]
        y_train, y_test = imb.target[train_index], imb.target[test_index]
    adaboost = AdaBoostClassifier(random_state=0)
    adaboost.fit(X_train, y_train)
    adaboost_score = cohen_kappa_score(adaboost.predict(X_test), y_test)


    clf = SMOTEBoost(random_state=0)
    clf.fit(X_train, y_train)
    score = cohen_kappa_score(clf.predict(X_test), y_test)
    assert score >= adaboost_score, "Failed with score = %f; AdaBoostClassifier score= %f" % (score, adaboost_score)
