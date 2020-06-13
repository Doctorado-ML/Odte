import unittest
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning

from odte import Odte
from .utils import load_dataset


class Odte_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._random_state = 1
        super().__init__(*args, **kwargs)

    def test_max_samples_bogus(self):
        values = [0, 3000, 1.1, 0.0, "hi!"]
        for max_samples in values:
            with self.assertRaises(ValueError):
                tclf = Odte(max_samples=max_samples)
                tclf.fit(*load_dataset(self._random_state))

    def test_get_bootstrap_nsamples(self):
        expected_values = [(1, 1), (1500, 1500), (0.1, 150)]
        for value, expected in expected_values:
            tclf = Odte(max_samples=value)
            computed = tclf._get_bootstrap_n_samples(1500)
            self.assertEqual(expected, computed)

    def test_initialize_sample_weight(self):
        m = 5
        ones = np.ones(m,)
        weights = np.random.rand(m,)
        expected_values = [(None, ones), (weights, weights)]
        for value, expected in expected_values:
            tclf = Odte()
            computed = tclf._initialize_sample_weight(value, m)
            self.assertListEqual(expected.tolist(), computed.tolist())

    def test_initialize_random(self):
        expected = [37, 235, 908]
        tclf = Odte(random_state=self._random_state)
        box = tclf._initialize_random()
        computed = box.randint(0, 1000, 3)
        self.assertListEqual(expected, computed.tolist())
        # test None
        tclf = Odte()
        box = tclf._initialize_random()
        computed = box.randint(101, 1000, 3)
        for value in computed.tolist():
            self.assertGreaterEqual(value, 101)
            self.assertLessEqual(value, 1000)

    def test_bogus_n_estimator(self):
        values = [0, -1, 2]
        for n_estimators in values:
            with self.assertRaises(ValueError):
                tclf = Odte(n_estimators=n_estimators)
                tclf.fit(*load_dataset(self._random_state))

    def test_simple_predict(self):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        X, y = [[1, 2], [5, 6], [9, 10], [16, 17]], [0, 1, 1, 2]
        expected = [0, 1, 1, 0]
        tclf = Odte(
            random_state=self._random_state, n_estimators=10, kernel="rbf"
        )
        computed = tclf.fit(X, y).predict(X)
        self.assertListEqual(expected, computed.tolist())

    def test_predict(self):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        X, y = load_dataset(self._random_state)
        expected = y
        tclf = Odte(
            random_state=self._random_state, n_estimators=10, kernel="linear"
        )
        computed = tclf.fit(X, y).predict(X)
        self.assertListEqual(expected[:27].tolist(), computed[:27].tolist())

    def test_score(self):
        X, y = load_dataset(self._random_state)
        expected = 0.9526666666666667
        tclf = Odte(random_state=self._random_state, n_estimators=10)
        computed = tclf.fit(X, y).score(X, y)
        self.assertAlmostEqual(expected, computed)
