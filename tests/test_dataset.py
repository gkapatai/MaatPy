import os
import unittest

import numpy as np
from sklearn.utils.testing import assert_array_equal, assert_allclose
from maatpy.dataset import (Dataset,
                            simulate_dataset)


class Test_Dataset(unittest.TestCase):

    def setUp(self):
        self.seed = 0
        self.r_tol = 1e-4
        self.csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     'test_data/test_dataset.csv')

        self.data = np.array([[2.38363567,  1.56914464],
                              [0.29484027, -0.79249401],
                              [-1.01647325, -0.78917179],
                              [0.05168253, -0.81186936],
                              [1.97572045,  1.40780946],
                              [1.51783379,  1.22140561],
                              [1.3001697,  1.11938505],
                              [0.56043168, -0.24565814],
                              [-1.44942515, -0.01068492],
                              [3.42305228,  2.00853206],
                              [1.1913926, -0.92825855],
                              [1.18740521, -0.6260027]])
        self.target = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0])

    def test_dataset_init(self):
        myclass = Dataset(data=self.data, target=self.target)
        assert_array_equal(myclass.data, self.data)
        assert_array_equal(myclass.target, self.target)
        assert myclass.feature_names is None
        assert myclass.target_names is None

    def test_make_imbalance_dict(self):
        ratio = {0: 8, 1: 4}
        myclass = Dataset(data=self.data, target=self.target, feature_names=['X1', 'X2'], target_names=[0,1])

        myclass.make_imbalance(ratio=ratio, random_state=self.seed)

        X_gt = np.array([[1.18740521, -0.6260027],
                         [0.05168253, -0.81186936],
                         [-1.01647325, -0.78917179],
                         [0.56043168, -0.24565814],
                         [0.29484027, -0.79249401],
                         [1.1913926, -0.92825855],
                         [1.97572045,  1.40780946],
                         [1.3001697,  1.11938505],
                         [-1.44942515, -0.01068492],
                         [2.38363567,  1.56914464],
                         [0.29484027, -0.79249401],
                         [1.1913926, -0.92825855]])
        y_gt = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
        assert_array_equal(myclass.data, X_gt)
        assert_array_equal(myclass.target, y_gt)

    def test_make_imbalance_list(self):
        ratio = [0.8, 0.2]
        myclass = Dataset(data=self.data, target=self.target, feature_names=['X1', 'X2'], target_names=[0,1])

        myclass.make_imbalance(ratio=ratio, random_state=self.seed)

        X_gt = np.array([[1.18740521, -0.6260027],
                         [0.05168253, -0.81186936],
                         [-1.01647325, -0.78917179],
                         [0.56043168, -0.24565814],
                         [0.29484027, -0.79249401],
                         [1.1913926, -0.92825855],
                         [1.97572045,  1.40780946],
                         [1.3001697,  1.11938505],
                         [0.29484027, -0.79249401],
                         [1.1913926, -0.92825855],
                         [1.18740521, -0.6260027],
                         [0.56043168, -0.24565814]])
        y_gt = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
        assert_array_equal(myclass.data, X_gt)
        assert_array_equal(myclass.target, y_gt)

    def test_simulate_dataset(self):
        myclass = simulate_dataset(n_samples=12, random_state=self.seed)
        assert_allclose(myclass.data, self.data, rtol=self.r_tol)
        assert_array_equal(myclass.target, self.target)

    def test_make_imbalance_ratio_type_error(self):
        myclass = Dataset(data=self.data, target=self.target, feature_names=['X1', 'X2'], target_names=[0, 1])
        self.assertRaises(TypeError, myclass.make_imbalance, ratio="auto", random_state=self.seed)
