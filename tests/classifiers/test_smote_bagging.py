"""Test the module ensemble classifiers."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np

from sklearn.datasets import load_iris, make_hastie_10_2
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     train_test_split)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import (Perceptron,
                                  LogisticRegression)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.utils.testing import (assert_array_equal,
                                   assert_array_almost_equal,
                                   assert_raises,
                                   assert_warns,
                                   assert_warns_message)

from maatpy.dataset import Dataset
from maatpy.classifiers import SMOTEBagging
from maatpy.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

iris = load_iris()
imb_iris = Dataset(data=iris.data, target=iris.target, feature_names=iris.feature_names,
                   target_names=iris.target_names)
imb_iris.make_imbalance(ratio={0: 20, 1: 30, 2: 50}, random_state=0)

def test_smote_bagging():
    # Check classification for various parameter settings.
    X_train, X_test, y_train, y_test = train_test_split(imb_iris.data, imb_iris.target,
                                                        random_state=0)
    grid = ParameterGrid({"max_samples": [0.5, 1.0],
                          "max_features": [1, 2, 4],
                          "bootstrap": [True, False],
                          "bootstrap_features": [True, False]})

    for base_estimator in [None,
                           DummyClassifier(),
                           Perceptron(),
                           DecisionTreeClassifier(),
                           KNeighborsClassifier(),
                           SVC()]:
        for params in grid:
            SMOTEBagging(
                base_estimator=base_estimator,
                k_neighbors=3,
                random_state=0,
                **params).fit(X_train, y_train).predict(X_test)


def test_bootstrap_samples():
    # Test that bootstrapping samples generate non-perfect base estimators.
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                        random_state=0)

    base_estimator = DecisionTreeClassifier().fit(X_train, y_train)

    # without bootstrap, all trees are perfect on the training set
    # disable the resampling by passing an empty dictionary.
    ensemble = SMOTEBagging(
        base_estimator=DecisionTreeClassifier(),
        max_samples=1.0,
        bootstrap=False,
        k_neighbors=3,
        n_estimators=10,
        ratio={},
        random_state=0).fit(X_train, y_train)

    assert (ensemble.score(X_train, y_train) ==
            base_estimator.score(X_train, y_train))

    # with bootstrap, trees are no longer perfect on the training set
    ensemble = SMOTEBagging(
        base_estimator=DecisionTreeClassifier(),
        max_samples=1.0,
        bootstrap=True,
        k_neighbors=3,
        random_state=0).fit(X_train, y_train)

    assert (ensemble.score(X_train, y_train) <
            base_estimator.score(X_train, y_train))


def test_bootstrap_features():
    # Test that bootstrapping features may generate duplicate features.
    X_train, X_test, y_train, y_test = train_test_split(imb_iris.data, imb_iris.target,
                                                        random_state=0)

    ensemble = SMOTEBagging(
        base_estimator=DecisionTreeClassifier(),
        max_features=1.0,
        bootstrap_features=False,
        random_state=0).fit(X_train, y_train)

    for features in ensemble.estimators_features_:
        assert np.unique(features).shape[0] == imb_iris.data.shape[1]

    ensemble = SMOTEBagging(
        base_estimator=DecisionTreeClassifier(),
        max_features=1.0,
        bootstrap_features=True,
        random_state=0).fit(X_train, y_train)

    unique_features = [np.unique(features).shape[0]
                       for features in ensemble.estimators_features_]
    assert np.median(unique_features) < imb_iris.data.shape[1]


def test_probability():
    # Predict probabilities.
    X_train, X_test, y_train, y_test = train_test_split(imb_iris.data, imb_iris.target,
                                                        random_state=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Normal case
        ensemble = SMOTEBagging(
            base_estimator=DecisionTreeClassifier(), k_neighbors=3,
            random_state=0).fit(X_train, y_train)

        assert_array_almost_equal(np.sum(ensemble.predict_proba(X_test),
                                         axis=1),
                                  np.ones(len(X_test)))

        assert_array_almost_equal(ensemble.predict_proba(X_test),
                                  np.exp(ensemble.predict_log_proba(X_test)))


def test_oob_score_classification():
    # Check that oob prediction is a good estimation of the generalization
    # error.
    X_train, X_test, y_train, y_test = train_test_split(imb_iris.data, imb_iris.target,
                                                        random_state=0)

    for base_estimator in [DecisionTreeClassifier(), SVC()]:
        clf = SMOTEBagging(
            base_estimator=base_estimator,
            n_estimators=100,
            bootstrap=True,
            oob_score=True,
            random_state=0).fit(X_train, y_train)

        test_score = clf.score(X_test, y_test)

        assert abs(test_score - clf.oob_score_) < 0.1

        # Test with few estimators
        assert_warns(UserWarning,
                     SMOTEBagging(
                         base_estimator=base_estimator,
                         n_estimators=1,
                         bootstrap=True,
                         oob_score=True,
                         random_state=0).fit,
                     X_train,
                     y_train)


def test_single_estimator():
    # Check singleton ensembles.
    X_train, X_test, y_train, y_test = train_test_split(imb_iris.data, imb_iris.target,
                                                        random_state=0)

    clf1 = SMOTEBagging(
        base_estimator=KNeighborsClassifier(),
        n_estimators=1,
        bootstrap=False,
        bootstrap_features=False,
        random_state=0).fit(X_train, y_train)

    clf2 = make_pipeline(SMOTE(
        random_state=clf1.estimators_[0].steps[0][1].random_state),
                         KNeighborsClassifier()).fit(X_train, y_train)

    assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))


def test_error():
    # Test that it gives proper exception on deficient input.
    base = DecisionTreeClassifier()
    X, y = [imb_iris.data, imb_iris.target]
    # Test n_estimators
    assert_raises(ValueError,
                  SMOTEBagging(base, n_estimators=1.5).fit, X, y)
    assert_raises(ValueError,
                  SMOTEBagging(base, n_estimators=-1).fit, X, y)

    # Test max_samples
    assert_raises(ValueError,
                  SMOTEBagging(base, max_samples=-1).fit, X, y)
    assert_raises(ValueError,
                  SMOTEBagging(base, max_samples=0.0).fit, X, y)
    assert_raises(ValueError,
                  SMOTEBagging(base, max_samples=2.0).fit, X, y)
    assert_raises(ValueError,
                  SMOTEBagging(base, max_samples=1000).fit, X, y)
    assert_raises(ValueError,
                  SMOTEBagging(base, max_samples="foobar").fit,
                  X, y)

    # Test max_features
    assert_raises(ValueError,
                  SMOTEBagging(base, max_features=-1).fit, X, y)
    assert_raises(ValueError,
                  SMOTEBagging(base, max_features=0.0).fit, X, y)
    assert_raises(ValueError,
                  SMOTEBagging(base, max_features=2.0).fit, X, y)
    assert_raises(ValueError,
                  SMOTEBagging(base, max_features=5).fit, X, y)
    assert_raises(ValueError,
                  SMOTEBagging(base, max_features="foobar").fit,
                  X, y)

    # Test support of decision_function
    assert not (hasattr(SMOTEBagging(base).fit(X, y),
                        'decision_function'))


def test_gridsearch():
    # Check that bagging ensembles can be grid-searched.
    # Transform iris into a binary classification task
    X, y = iris.data, iris.target.copy()
    y[y == 2] = 1

    # Grid search with scoring based on decision_function
    parameters = {'n_estimators': (1, 2),
                  'base_estimator__C': (1, 2)}

    GridSearchCV(SMOTEBagging(SVC()),
                 parameters,
                 scoring="roc_auc").fit(X, y)


def test_base_estimator():
    # Check base_estimator and its default values.
    X_train, X_test, y_train, y_test = train_test_split(imb_iris.data, imb_iris.target,
                                                        random_state=0)

    ensemble = SMOTEBagging(None,
                                         n_jobs=3,
                                         random_state=0).fit(X_train, y_train)

    assert isinstance(ensemble.base_estimator_.steps[-1][1],
                      DecisionTreeClassifier)

    ensemble = SMOTEBagging(DecisionTreeClassifier(),
                                         n_jobs=3,
                                         random_state=0).fit(X_train, y_train)

    assert isinstance(ensemble.base_estimator_.steps[-1][1],
                      DecisionTreeClassifier)

    ensemble = SMOTEBagging(Perceptron(),
                                         n_jobs=3,
                                         random_state=0).fit(X_train, y_train)

    assert isinstance(ensemble.base_estimator_.steps[-1][1],
                      Perceptron)


def test_bagging_with_pipeline():
    estimator = SMOTEBagging(
        make_pipeline(SelectKBest(k=1),
                      DecisionTreeClassifier()),
        max_features=2)
    estimator.fit(imb_iris.data, imb_iris.target).predict(imb_iris.data)


def test_warm_start(random_state=42):
    # Test if fitting incrementally with warm start gives a forest of the
    # right size and the same results as a normal fit.
    X, y = make_hastie_10_2(n_samples=100, random_state=1)

    clf_ws = None
    for n_estimators in [5, 10]:
        if clf_ws is None:
            clf_ws = SMOTEBagging(n_estimators=n_estimators,
                                               random_state=random_state,
                                               warm_start=True)
        else:
            clf_ws.set_params(n_estimators=n_estimators)
        clf_ws.fit(X, y)
        assert len(clf_ws) == n_estimators

    clf_no_ws = SMOTEBagging(n_estimators=10,
                                          random_state=random_state,
                                          warm_start=False)
    clf_no_ws.fit(X, y)

    assert (set([pipe.steps[-1][1].random_state for pipe in clf_ws]) ==
            set([pipe.steps[-1][1].random_state for pipe in clf_no_ws]))


def test_warm_start_smaller_n_estimators():
    # Test if warm start'ed second fit with smaller n_estimators raises error.
    X, y = make_hastie_10_2(n_samples=100, random_state=1)
    clf = SMOTEBagging(n_estimators=5, warm_start=True)
    clf.fit(X, y)
    clf.set_params(n_estimators=4)
    assert_raises(ValueError, clf.fit, X, y)


def test_warm_start_equal_n_estimators():
    # Test that nothing happens when fitting without increasing n_estimators
    X, y = make_hastie_10_2(n_samples=100, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43)

    clf = SMOTEBagging(n_estimators=5, warm_start=True,
                                    random_state=83)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # modify X to nonsense values, this should not change anything
    X_train += 1.

    assert_warns_message(UserWarning,
                         "Warm-start fitting without increasing n_estimators"
                         " does not", clf.fit, X_train, y_train)
    assert_array_equal(y_pred, clf.predict(X_test))


def test_warm_start_equivalence():
    # warm started classifier with 5+5 estimators should be equivalent to
    # one classifier with 10 estimators
    X, y = make_hastie_10_2(n_samples=100, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43)

    clf_ws = SMOTEBagging(n_estimators=5, warm_start=True,
                                       random_state=3141)
    clf_ws.fit(X_train, y_train)
    clf_ws.set_params(n_estimators=10)
    clf_ws.fit(X_train, y_train)
    y1 = clf_ws.predict(X_test)

    clf = SMOTEBagging(n_estimators=10, warm_start=False,
                                    random_state=3141)
    clf.fit(X_train, y_train)
    y2 = clf.predict(X_test)

    assert_array_almost_equal(y1, y2)


def test_warm_start_with_oob_score_fails():
    # Check using oob_score and warm_start simultaneously fails
    X, y = make_hastie_10_2(n_samples=100, random_state=1)
    clf = SMOTEBagging(n_estimators=5, warm_start=True,
                                    oob_score=True)
    assert_raises(ValueError, clf.fit, X, y)


def test_oob_score_removed_on_warm_start():
    X, y = make_hastie_10_2(n_samples=2000, random_state=1)

    clf = SMOTEBagging(n_estimators=50, oob_score=True)
    clf.fit(X, y)

    clf.set_params(warm_start=True, oob_score=False, n_estimators=100)
    clf.fit(X, y)

    assert_raises(AttributeError, getattr, clf, "oob_score_")


def test_oob_score_consistency():
    # Make sure OOB scores are identical when random_state, estimator, and
    # training data are fixed and fitting is done twice
    X, y = make_hastie_10_2(n_samples=200, random_state=1)
    bagging = SMOTEBagging(KNeighborsClassifier(),
                                        max_samples=0.5,
                                        max_features=0.5, oob_score=True,
                                        random_state=1)
    assert bagging.fit(X, y).oob_score_ == bagging.fit(X, y).oob_score_


def test_estimators_samples():
    # Check that format of estimators_samples_ is correct
    X, y = make_hastie_10_2(n_samples=100, random_state=1)

    # remap the y outside of the SMOTEBagging
    # _, y = np.unique(y, return_inverse=True)
    bagging = SMOTEBagging(LogisticRegression(), max_samples=0.5,
                                        max_features=0.5, random_state=0,
                                        bootstrap=False)
    bagging.fit(X, y)

    # Get relevant attributes
    estimators_samples = bagging.estimators_samples_
    estimators_features = bagging.estimators_features_
    estimators = bagging.estimators_

    # Test for correct formatting
    assert len(estimators_samples) == len(estimators)
    assert len(estimators_samples[0]) == len(X)
    assert estimators_samples[0].dtype.kind == 'b'


def test_max_samples_consistency():
    # Make sure validated max_samples and original max_samples are identical
    # when valid integer max_samples supplied by user
    max_samples = 100
    X, y = make_hastie_10_2(n_samples=2*max_samples, random_state=1)
    bagging = SMOTEBagging(KNeighborsClassifier(),
                                        max_samples=max_samples,
                                        max_features=0.5, random_state=1)
    bagging.fit(X, y)
    assert bagging._max_samples == max_samples


def test_imb_performance():
    from maatpy.dataset import simulate_dataset
    from sklearn.metrics import cohen_kappa_score
    from sklearn.model_selection import StratifiedShuffleSplit
    imb = simulate_dataset(n_samples=500, n_features=2, n_classes=2, weights=[0.9, 0.1], random_state=0)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    sss.get_n_splits(imb.data, imb.target)
    for train_index, test_index in sss.split(imb.data, imb.target):
        X_train, X_test = imb.data[train_index], imb.data[test_index]
        y_train, y_test = imb.target[train_index], imb.target[test_index]
    orig_clf = BaggingClassifier(random_state=0)
    orig_clf.fit(X_train, y_train)
    orig_score = cohen_kappa_score(orig_clf.predict(X_test), y_test)
    clf = SMOTEBagging(random_state=0)
    clf.fit(X_train, y_train)
    score = cohen_kappa_score(clf.predict(X_test), y_test)
    assert score >= orig_score, "Failed with score = %f; BaggingClassifier score= %f" % \
                                (score, orig_score)