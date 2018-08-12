"""Testing for the boost module (sklearn.ensemble.boost)."""

import numpy as np

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal, assert_true, assert_greater
from sklearn.utils.testing import assert_raises, assert_raises_regexp

from sklearn.model_selection import GridSearchCV
from maatpy.classifiers import AdaCost
from sklearn.ensemble import weight_boosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn import datasets


# Common random state
rng = np.random.RandomState(0)

# Toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y_class = ["foo", "foo", "foo", 1, 1, 1]
y_regr = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
y_t_class = ["foo", 1, 1]
y_t_regr = [-1, 1, 1]



# Load the iris dataset and randomly permute it
iris = datasets.load_iris()
perm = rng.permutation(iris.target.size)
iris.data, iris.target = shuffle(iris.data, iris.target, random_state=rng)


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


def test_oneclass_error():
    # Test a single class dataset is not accepted.
    # In response to issue #7501
    # https://github.com/scikit-learn/scikit-learn/issues/7501
    y_t = np.ones(len(X))
    assert_raises(ValueError, AdaCost().fit, X, y_t)


def test_classification_toy():
    # Check classification on a toy dataset.
    for alg in ['adacost', 'adac1', 'adac2', 'adac3']:
        clf = AdaCost(algorithm=alg, random_state=0)
        clf.fit(X, y_class)
        assert_array_equal(clf.predict(T), y_t_class)
        assert_array_equal(np.unique(np.asarray(y_t_class)), clf.classes)
        assert_equal(clf.predict_proba(T).shape, (len(T), 2))
        assert_equal(clf.decision_function(T).shape, (len(T),2))


def test_iris():
    # Check consistency on dataset iris.
    classes = np.unique(iris.target)

    adaboost = AdaBoostClassifier()
    adaboost.fit(iris.data, iris.target)

    for alg in ['adacost', 'adac1', 'adac2', 'adac3']:
        clf = AdaCost(algorithm=alg)
        clf.fit(iris.data, iris.target)

        assert_array_equal(classes, clf.classes_)
        proba = clf.predict_proba(iris.data)
        assert_equal(proba.shape[1], len(classes))
        assert_equal(clf.decision_function(iris.data).shape[1], len(classes))

        score = clf.score(iris.data, iris.target)
        assert score >= adaboost.score(iris.data, iris.target), "Failed with algorithm %s and score = %f" % \
            (alg, score)

        # Check we used multiple estimators
        assert_greater(len(clf.estimators_), 1)
        # Check for distinct random states (see issue #7408)
        assert_equal(len(set(est.random_state for est in clf.estimators_)),
                     len(clf.estimators_))


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

    for alg in ['adacost', 'adac1', 'adac2', 'adac3']:
        clf = AdaCost(algorithm=alg)
        clf.fit(X_train, y_train)
        score = cohen_kappa_score(clf.predict(X_test), y_test)
        assert score >= adaboost_score, "Failed with algorithm %s and score = %f" % (alg, score)


def test_staged_predict():
    # Check staged predictions.
    rng = np.random.RandomState(0)
    iris_weights = rng.randint(10, size=iris.target.shape).astype(float)

    # AdaBoost classification
    for alg in ['adacost', 'adac1', 'adac2', 'adac3']:
        clf = AdaCost(algorithm=alg, n_estimators=10)
        clf.fit(iris.data, iris.target, sample_weight=iris_weights)

        predictions = clf.predict(iris.data)
        staged_predictions = [p for p in clf.staged_predict(iris.data)]
        proba = clf.predict_proba(iris.data)
        staged_probas = [p for p in clf.staged_predict_proba(iris.data)]
        score = clf.score(iris.data, iris.target, sample_weight=iris_weights)
        staged_scores = [
            s for s in clf.staged_score(
                iris.data, iris.target, sample_weight=iris_weights)]

        assert_equal(len(staged_predictions), 10)
        assert_array_almost_equal(predictions, staged_predictions[-1])
        assert_equal(len(staged_probas), 10)
        assert_array_almost_equal(proba, staged_probas[-1])
        assert_equal(len(staged_scores), 10)
        assert_array_almost_equal(score, staged_scores[-1])


def test_gridsearch():
    # Check that base trees can be grid-searched.
    # AdaBoost classification
    boost = AdaCost(base_estimator=DecisionTreeClassifier())
    parameters = {'n_estimators': (1, 2),
                  'base_estimator__max_depth': (1, 2),
                  'algorithm': ('adacost', 'adac1', 'adac2', 'adac3')}
    clf = GridSearchCV(boost, parameters)
    clf.fit(iris.data, iris.target)


def test_pickle():
    # Check pickability.
    import pickle

    # Adaboost classifier
    for alg in ['adacost', 'adac1', 'adac2', 'adac3']:
        obj = AdaCost(algorithm=alg)
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

    for alg in ['adacost', 'adac1', 'adac2', 'adac3']:
        clf = AdaCost(algorithm=alg)

        clf.fit(X, y)
        importances = clf.feature_importances_

        assert_equal(importances.shape[0], 10)
        assert_equal((importances[:3, np.newaxis] >= importances[3:]).all(),
                     True)


def test_error():
    # Test that it gives proper exception on deficient input.
    assert_raises(ValueError,
                  AdaCost(learning_rate=-1).fit,
                  X, y_class)

    assert_raises(ValueError,
                  AdaCost(algorithm="foo").fit,
                  X, y_class)

    assert_raises(ValueError,
                  AdaCost().fit,
                  X, y_class, sample_weight=np.asarray([-1]))


def test_base_estimator():
    # Check that an empty discrete ensemble fails in fit, not predict.
    X_fail = [[1, 1], [1, 1], [1, 1], [1, 1]]
    y_fail = ["foo", "bar", 1, 2]
    clf = AdaCost(DecisionTreeClassifier(max_depth=1), algorithm="adacost")
    assert_raises_regexp(ValueError, "worse than random",
                         clf.fit, X_fail, y_fail)
