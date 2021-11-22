"""
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
Build a forest of oblique trees based on STree
"""
from __future__ import annotations
import random
import sys
from math import factorial
from typing import Union, Optional, Tuple, List, Set
import numpy as np
from sklearn.utils.multiclass import (  # type: ignore
    check_classification_targets,
)
from sklearn.base import clone, BaseEstimator, ClassifierMixin  # type: ignore
from sklearn.ensemble import BaseEnsemble  # type: ignore
from sklearn.utils.validation import (  # type: ignore
    check_is_fitted,
    _check_sample_weight,
)
from joblib import Parallel, delayed  # type: ignore
from stree import Stree  # type: ignore


class Odte(BaseEnsemble, ClassifierMixin):
    def __init__(
        self,
        # n_jobs = -1 to use all available cores
        n_jobs: int = 1,
        base_estimator: BaseEstimator = None,
        random_state: int = 0,
        max_features: Optional[Union[str, int, float]] = None,
        max_samples: Optional[Union[int, float]] = None,
        n_estimators: int = 100,
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
        )
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_features = max_features
        self.max_samples = max_samples  # size of bootstrap

    def _initialize_random(self) -> np.random.mtrand.RandomState:
        if self.random_state is None:
            self.random_state = random.randint(0, sys.maxsize)
            return np.random.mtrand._rand
        return np.random.RandomState(self.random_state)

    def _validate_estimator(self) -> None:
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=Stree(random_state=self.random_state)
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None
    ) -> Odte:
        # Check parameters are Ok.
        if self.n_estimators < 3:
            raise ValueError(
                f"n_estimators must be greater than 2 but got (n_estimators=\
                    {self.n_estimators})"
            )
        check_classification_targets(y)
        X, y = self._validate_data(X, y)
        # if sample_weight is None return np.ones
        sample_weights = _check_sample_weight(
            sample_weight, X, dtype=np.float64
        )
        check_classification_targets(y)
        # Initialize computed parameters
        #  Build the estimator
        self.max_features_ = self._initialize_max_features()
        # build base_estimator_
        self._validate_estimator()
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_: int = self.classes_.shape[0]
        self.estimators_: List[BaseEstimator] = []
        self.subspaces_: List[Tuple[int, ...]] = []
        result = self._train(X, y, sample_weights)
        self.estimators_, self.subspaces_ = tuple(zip(*result))  # type: ignore
        self._compute_metrics()
        return self

    def _compute_metrics(self) -> None:
        tdepth = tnodes = tleaves = 0
        for estimator in self.estimators_:
            nodes, leaves = estimator.nodes_leaves()
            depth = estimator.depth_
            tdepth += depth
            tnodes += nodes
            tleaves += leaves
        self.depth_ = tdepth / self.n_estimators
        self.leaves_ = tleaves / self.n_estimators
        self.nodes_ = tnodes / self.n_estimators

    @staticmethod
    def _parallel_build_tree(
        base_estimator_: Stree,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        random_box: np.random.mtrand.RandomState,
        random_seed: int,
        boot_samples: int,
        max_features: int,
    ) -> Tuple[BaseEstimator, Tuple[int, ...]]:
        clf = clone(base_estimator_)
        clf.set_params(random_state=random_seed)
        n_samples = X.shape[0]
        # bootstrap
        indices = random_box.randint(0, n_samples, boot_samples)
        # update weights with the chosen samples
        weights_update = np.bincount(indices, minlength=n_samples)
        current_weights = weights * weights_update
        # random subspace
        features = Odte._get_random_subspace(X, y, max_features)
        # train the classifier
        bootstrap = X[indices, :]
        clf.fit(bootstrap[:, features], y[indices], current_weights[indices])
        return (clf, features)

    def _train(
        self, X: np.ndarray, y: np.ndarray, weights: np.ndarray
    ) -> Tuple[List[BaseEstimator], List[Tuple[int, ...]]]:
        random_box = self._initialize_random()
        n_samples = X.shape[0]
        boot_samples = self._get_bootstrap_n_samples(n_samples)
        clf = clone(self.base_estimator_)
        return Parallel(n_jobs=self.n_jobs, prefer="threads")(  # type: ignore
            delayed(Odte._parallel_build_tree)(
                clf,
                X,
                y,
                weights,
                random_box,
                random_seed,
                boot_samples,
                self.max_features_,
            )
            for random_seed in range(
                self.random_state, self.random_state + self.n_estimators
            )
        )

    def _get_bootstrap_n_samples(self, n_samples: int) -> int:
        if self.max_samples is None:
            return n_samples
        if isinstance(self.max_samples, int):
            if not (1 <= self.max_samples <= n_samples):
                message = f"max_samples should be in the range 1 to \
                    {n_samples} but got {self.max_samples}"
                raise ValueError(message)
            return self.max_samples
        if isinstance(self.max_samples, float):
            if not (0 < self.max_samples < 1):
                message = f"max_samples should be in the range (0, 1)\
                    but got {self.max_samples}"
                raise ValueError(message)
            return int(round(self.max_samples * n_samples))
        raise ValueError(
            f"Expected values int, float but got \
            {type(self.max_samples)}"
        )

    def _initialize_max_features(self) -> int:
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
            else:
                raise ValueError(
                    "Invalid value for max_features. "
                    "Allowed string values are 'auto', "
                    "'sqrt' or 'log2'."
                )
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, int):
            max_features = abs(self.max_features)
        else:  # float
            if self.max_features > 0.0:
                max_features = max(
                    1, int(self.max_features * self.n_features_in_)
                )
            else:
                raise ValueError(
                    "Invalid value for max_features."
                    "Allowed float must be in range (0, 1] "
                    f"got ({self.max_features})"
                )
        return max_features

    @staticmethod
    def _generate_spaces(features: int, max_features: int) -> list:
        comb: Set[Tuple[int, ...]] = set()
        # Generate at most 5 combinations
        if max_features == features:
            set_length = 1
        else:
            number = factorial(features) / (
                factorial(max_features) * factorial(features - max_features)
            )
            set_length = min(5, int(number))
        while len(comb) < set_length:
            comb.add(
                tuple(sorted(random.sample(range(features), max_features)))
            )
        return list(comb)

    @staticmethod
    def _get_random_subspace(
        dataset: np.ndarray, labels: np.ndarray, max_features: int
    ) -> Tuple[int, ...]:
        features_sets = Odte._generate_spaces(dataset.shape[1], max_features)
        if len(features_sets) > 1:
            index = random.randint(0, len(features_sets) - 1)
            return features_sets[index]
        else:
            return features_sets[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        # Input validation
        X = self._validate_data(X, reset=False)
        n_samples = X.shape[0]
        result = np.zeros((n_samples, self.n_classes_))
        for tree, features in zip(self.estimators_, self.subspaces_):
            predictions = tree.predict(X[:, features])
            for i in range(n_samples):
                result[i, predictions[i]] += 1
        return result / self.n_estimators

    def nodes_leaves(self) -> Tuple[float, float]:
        check_is_fitted(self, "estimators_")
        return self.nodes_, self.leaves_
