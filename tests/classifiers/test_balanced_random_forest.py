import unittest
import numpy as np
from maatpy.classifiers import BalancedRandomForestClassifier
from sklearn import datasets
from sklearn.utils.validation import check_random_state
from sklearn.utils.testing import (assert_array_almost_equal,
                                   assert_array_equal,
                                   assert_equal)


class Test_BalancedRandomForestClassifier(unittest.TestCase):

    def setUp(self):
        self.X = [[-2, -1], [-1, -1], [2, 2], [1, 1], [1, 2], [2, 1]]
        self.y = [-1, -1, 1, 1, 1, 1]
        self.T = [[-1, -1], [2, 2], [3, 2]]
        self.true_result = [-1, 1, 1]
        self.iris = datasets.load_iris()
        self.criterion = ["gini", "entropy"]

    def test_parallel_train(self):
        rng = check_random_state(12321)
        n_samples, n_features = 80, 30
        X_train = rng.randn(n_samples, n_features)
        y_train = rng.randint(0, 2, n_samples)

        clfs = [
            BalancedRandomForestClassifier(n_estimators=20, n_jobs=n_jobs,
                                           random_state=12345).fit(X_train, y_train)
            for n_jobs in [1, 2, 3, 8, 16, 32]
        ]

        X_test = rng.randn(n_samples, n_features)
        probas = [clf.predict_proba(X_test) for clf in clfs]
        for proba1, proba2 in zip(probas, probas[1:]):
            assert_array_almost_equal(proba1, proba2)

    def test_dtype_convert(self):
        n_classes=15
        classifier = BalancedRandomForestClassifier(random_state=0, bootstrap=False)

        X = np.eye(n_classes)
        y = [ch for ch in 'ABCDEFGHIJKLMNOPQRSTU'[:n_classes]]

        result = classifier.fit(X, y).predict(X)
        assert_array_equal(classifier.classes_, y)
        assert_array_equal(result, y)

    def test_class_weight_errors(self):
        # Test if class_weight raises errors and warnings when expected.
        _y = np.vstack((self.y, np.array(self.y) * 2)).T

        # Invalid preset string
        clf = BalancedRandomForestClassifier(class_weight='the larch', random_state=0)
        self.assertRaises(ValueError, clf.fit, self.X, self.y)
        self.assertRaises(ValueError, clf.fit, self.X, _y)

        # Warning warm_start with preset
        clf = BalancedRandomForestClassifier(class_weight='balanced', warm_start=True,
                                             random_state=0)
        self.assertWarns(UserWarning, clf.fit, self.X, self.y)
        self.assertWarns(UserWarning, clf.fit, self.X, _y)

        # Not a list or preset for multi-output
        clf = BalancedRandomForestClassifier(class_weight=1, random_state=0)
        self.assertRaises(ValueError, clf.fit, self.X, _y)

        # Incorrect length list for multi-output
        clf = BalancedRandomForestClassifier(class_weight=[{-1: 0.5, 1: 1.}], random_state=0)
        self.assertRaises(ValueError, clf.fit, self.X, _y)

    def test_iris_criterion(self):
        # Check consistency on dataset iris.

        for criterion in self.criterion:
            clf = BalancedRandomForestClassifier(n_estimators=10, criterion=criterion,
                                                 random_state=1)
            clf.fit(self.iris.data, self.iris.target)
            score = clf.score(self.iris.data, self.iris.target)
            self.assertGreater(score, 0.9, "Failed with criterion %s and score = %f"
                               % (criterion, score))

            clf = BalancedRandomForestClassifier(n_estimators=10, criterion=criterion,
                                                 max_features=2, random_state=1)
            clf.fit(self.iris.data, self.iris.target)
            score = clf.score(self.iris.data, self.iris.target)
            self.assertGreater(score, 0.5, "Failed with criterion %s and score = %f"
                               % (criterion, score))

    def test_classification(self):
        # Check classification on a small dataset

        clf = BalancedRandomForestClassifier(n_estimators=10, random_state=1)
        clf.fit(self.X, self.y)
        assert_array_equal(clf.predict(self.T), self.true_result)
        assert_equal(10, len(clf))

        clf = BalancedRandomForestClassifier(n_estimators=10, max_features=1, random_state=1)
        clf.fit(self.X, self.y)
        assert_array_equal(clf.predict(self.T), self.true_result)
        assert_equal(10, len(clf))

        # also test apply
        leaf_indices = clf.apply(self.X)
        assert_equal(leaf_indices.shape, (len(self.X), clf.n_estimators))

    def test_probability(self):
        # Predict probabilities.
        with np.errstate(divide="ignore"):
            clf = BalancedRandomForestClassifier(n_estimators=10, random_state=1, max_features=1,
                                                 max_depth=1)
            clf.fit(self.iris.data, self.iris.target)
            assert_array_almost_equal(np.sum(clf.predict_proba(self.iris.data), axis=1),
                                      np.ones(self.iris.data.shape[0]))
            assert_array_almost_equal(clf.predict_proba(self.iris.data),
                                      np.exp(clf.predict_log_proba(self.iris.data)))

    def test_oob_score(self):
        # Check that oob prediction is a good estimation of the generalization
        # error.

        # Proper behavior
        est = BalancedRandomForestClassifier(oob_score=True, random_state=1,
                                             n_estimators=10, bootstrap=True)
        n_samples = self.iris.data.shape[0]
        est.fit(self.iris.data[:n_samples // 2, :], self.iris.target[:n_samples // 2])
        test_score = est.score(self.iris.data[n_samples // 2:, :], self.iris.target[n_samples // 2:])
        self.assertLess(abs(test_score - est.oob_score_), 0.1)
        print(test_score, est.oob_score_)

        # Check warning if not enough estimators
        with np.errstate(divide="ignore", invalid="ignore"):
            est = BalancedRandomForestClassifier(oob_score=True, random_state=0,
                                                 n_estimators=1, bootstrap=True)
            self.assertWarns(UserWarning, est.fit, self.X, self.y)

    def test_parallel(self):
        """Check parallel computations in classification"""
        forest = BalancedRandomForestClassifier(n_estimators=10, n_jobs=3, random_state=0)

        forest.fit(self.iris.data, self.iris.target)
        assert_equal(len(forest), 10)

        forest.set_params(n_jobs=1)
        y1 = forest.predict(self.iris.data)
        forest.set_params(n_jobs=2)
        y2 = forest.predict(self.iris.data)
        assert_array_almost_equal(y1, y2, 3)

    def test_oob_score_raise_error(self):

            # Unfitted /  no bootstrap / no oob_score
        for oob_score, bootstrap in [(True, False), (False, True),
                                     (False, False)]:
            est = BalancedRandomForestClassifier(oob_score=oob_score, bootstrap=bootstrap,
                                                 random_state=0)
            self.assertFalse(hasattr(est, "oob_score_"))

            # No bootstrap
        self.assertRaises(ValueError, BalancedRandomForestClassifier(oob_score=True,
                                                                     bootstrap=False).fit, self.X, self.y)

    def test_ratio_error(self):

        self.assertRaises(ValueError, BalancedRandomForestClassifier(ratio='minority', random_state=0).fit,
                          self.X, self.y)
