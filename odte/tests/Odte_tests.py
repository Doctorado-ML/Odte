# type: ignore
import unittest
import os
import warnings
import json
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.svm import SVC
from odte import Odte
from stree import Stree
from .utils import load_dataset
from .._version import __version__


class Odte_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._random_state = 1
        super().__init__(*args, **kwargs)

    def test_max_samples_bogus(self):
        values = [0, 3000, 1.1, 0.0, "duck"]
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

    def test_initialize_max_feature(self):
        expected_values = [
            [4, 7, 12, 14],
            [2, 4, 6, 7, 12, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [4, 7, 12, 14],
            [4, 7, 12, 14],
            [4, 7, 12, 14],
        ]
        X, y = load_dataset(
            random_state=self._random_state, n_features=16, n_samples=10
        )
        for max_features in [4, 0.4, 1.0, None, "auto", "sqrt", "log2"]:
            tclf = Odte(
                random_state=self._random_state,
                max_features=max_features,
                n_jobs=1,
            )
            tclf.fit(X, y)
            computed = tclf._get_random_subspace(X, y, tclf.max_features_)
            expected = expected_values.pop(0)
            self.assertListEqual(expected, list(computed))
            # print(f"{list(computed)},")

    def test_bogus_max_features(self):
        values = ["duck", -0.1, 0.0]
        for max_features in values:
            with self.assertRaises(ValueError):
                tclf = Odte(max_features=max_features)
                tclf.fit(*load_dataset(self._random_state))

    def test_bogus_n_estimator(self):
        values = [0, -1, 2]
        for n_estimators in values:
            with self.assertRaises(ValueError):
                tclf = Odte(n_estimators=n_estimators)
                tclf.fit(*load_dataset(self._random_state))

    def test_simple_predict(self):
        os.environ["PYTHONWARNINGS"] = "ignore"
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        X, y = [[1, 2], [5, 6], [9, 10], [16, 17]], [0, 1, 1, 2]
        expected = [0, 1, 1, 2]
        tclf = Odte(
            base_estimator=Stree(),
            random_state=self._random_state,
            n_estimators=10,
            n_jobs=-1,
        )
        tclf.set_params(
            **dict(
                base_estimator__kernel="rbf",
                base_estimator__random_state=self._random_state,
            )
        )
        computed = tclf.fit(X, y).predict(X)
        self.assertListEqual(expected, computed.tolist())

    def test_predict(self):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        X, y = load_dataset(self._random_state)
        expected = y
        tclf = Odte(
            base_estimator=Stree(),
            random_state=self._random_state,
            max_features=1.0,
            max_samples=0.1,
        )
        tclf.set_params(
            **dict(
                base_estimator__kernel="linear",
            )
        )
        computed = tclf.fit(X, y).predict(X)
        self.assertListEqual(expected[:27].tolist(), computed[:27].tolist())

    def test_score(self):
        X, y = load_dataset(self._random_state)
        expected = 0.9533333333333334
        tclf = Odte(
            random_state=self._random_state,
            max_features=None,
            n_estimators=10,
        )
        computed = tclf.fit(X, y).score(X, y)
        self.assertAlmostEqual(expected, computed)

    def test_score_splitter_max_features(self):
        X, y = load_dataset(self._random_state, n_features=16, n_samples=500)
        results = [
            0.958,  # best auto
            0.942,  # random auto
            0.932,  # trandom auto
            0.95,  # mutual auto
            0.944,  # iwss auto
            0.946,  # cfs auto
            0.97,  # best None
            0.97,  # random None
            0.97,  # trandom None
            0.97,  # mutual None
            0.97,  # iwss None
            0.97,  # cfs None
        ]
        for max_features in ["auto", None]:
            for splitter in [
                "best",
                "random",
                "trandom",
                "mutual",
                "iwss",
                "cfs",
            ]:
                tclf = Odte(
                    base_estimator=Stree(),
                    random_state=self._random_state,
                    n_estimators=3,
                    n_jobs=1,
                )
                tclf.set_params(
                    **dict(
                        base_estimator__max_features=max_features,
                        base_estimator__splitter=splitter,
                        base_estimator__random_state=self._random_state,
                    )
                )
                expected = results.pop(0)
                computed = tclf.fit(X, y).score(X, y)
                # print(computed, splitter, max_features)
                self.assertAlmostEqual(expected, computed, msg=splitter)

    def test_generate_subspaces(self):
        features = 250
        for max_features in range(2, features):
            num = len(Odte._generate_spaces(features, max_features))
            self.assertEqual(5, num)
        self.assertEqual(3, len(Odte._generate_spaces(3, 2)))
        self.assertEqual(4, len(Odte._generate_spaces(4, 3)))

    @staticmethod
    def test_is_a_sklearn_classifier():
        os.environ["PYTHONWARNINGS"] = "ignore"
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        from sklearn.utils.estimator_checks import check_estimator

        check_estimator(Odte())

    def test_nodes_leaves_not_fitted(self):
        tclf = Odte(
            base_estimator=Stree(),
            random_state=self._random_state,
            n_estimators=3,
        )
        with self.assertRaises(NotFittedError):
            tclf.nodes_leaves()

    def test_nodes_leaves_depth(self):
        tclf = Odte(
            base_estimator=Stree(),
            random_state=self._random_state,
            n_estimators=5,
            n_jobs=1,
        )
        tclf_p = Odte(
            base_estimator=Stree(),
            random_state=self._random_state,
            n_estimators=5,
            n_jobs=-1,
        )
        X, y = load_dataset(self._random_state, n_features=16, n_samples=500)
        tclf.fit(X, y)
        tclf_p.fit(X, y)
        for clf in [tclf, tclf_p]:
            self.assertAlmostEqual(5.8, clf.depth_)
            self.assertAlmostEqual(9.4, clf.leaves_)
            self.assertAlmostEqual(17.8, clf.nodes_)
            nodes, leaves = clf.nodes_leaves()
            self.assertAlmostEqual(9.4, leaves)
            self.assertAlmostEqual(17.8, nodes)

    def test_nodes_leaves_SVC(self):
        tclf = Odte(
            base_estimator=SVC(),
            random_state=self._random_state,
            n_estimators=3,
        )
        X, y = load_dataset(self._random_state, n_features=16, n_samples=500)
        tclf.fit(X, y)
        self.assertAlmostEqual(0.0, tclf.leaves_)
        self.assertAlmostEqual(0.0, tclf.nodes_)
        nodes, leaves = tclf.nodes_leaves()
        self.assertAlmostEqual(0.0, leaves)
        self.assertAlmostEqual(0.0, nodes)

    def test_base_estimator_hyperparams(self):
        data = [
            (Stree(), {"max_features": 7, "max_depth": 2}),
            (SVC(), {"kernel": "linear", "cache_size": 100}),
        ]
        for clf, hyperparams in data:
            hyperparams_ = json.dumps(hyperparams)
            tclf = Odte(
                base_estimator=clf,
                random_state=self._random_state,
                n_estimators=3,
                be_hyperparams=hyperparams_,
            )
            self.assertEqual(hyperparams_, tclf.be_hyperparams)
            X, y = load_dataset(
                self._random_state, n_features=16, n_samples=500
            )
            tclf.fit(X, y)
            for estimator in tclf.estimators_:
                for key, value in hyperparams.items():
                    self.assertEqual(value, estimator.get_params()[key])

    def test_version(self):
        tclf = Odte()
        self.assertEqual(__version__, tclf.version())

    def test_parallel_score(self):
        tclf_p = Odte(
            n_jobs=-1, random_state=self._random_state, n_estimators=30
        )
        tclf_s = Odte(
            n_jobs=1, random_state=self._random_state, n_estimators=30
        )
        X, y = load_dataset(self._random_state, n_features=56, n_samples=1500)
        tclf_p.fit(X, y)
        tclf_s.fit(X, y)
        self.assertAlmostEqual(tclf_p.score(X, y), tclf_s.score(X, y))
