"""
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.1"
Build a forest of oblique trees based on STree
"""
from __future__ import annotations
import random
import sys
from typing import Union, Optional, Tuple, List
from itertools import combinations
import numpy as np  # type: ignore
from sklearn.utils.multiclass import (  # type: ignore
    check_classification_targets,
)
from sklearn.base import clone, BaseEstimator, ClassifierMixin  # type: ignore
from sklearn.ensemble import BaseEnsemble  # type: ignore
from sklearn.utils.validation import (  # type: ignore
    check_is_fitted,
    _check_sample_weight,
)

from stree import Stree  # type: ignore


class Odte(BaseEnsemble, ClassifierMixin):  # type: ignore
    def __init__(
        self,
        base_estimator: BaseEstimator = None,
        random_state: int = 0,
        max_features: Optional[Union[str, int, float]] = None,
        max_samples: Optional[Union[int, float]] = None,
        n_estimators: int = 100,
    ):
        base_estimator = (
            Stree(random_state=random_state)
            if base_estimator is None
            else base_estimator
        )
        super().__init__(
            base_estimator=base_estimator, n_estimators=n_estimators,
        )
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_features = max_features
        self.max_samples = max_samples  # size of bootstrap

    def _initialize_random(self) -> np.random.mtrand.RandomState:
        if self.random_state is None:
            self.random_state = random.randint(0, sys.maxint)
            return np.random.mtrand._rand
        return np.random.RandomState(self.random_state)

    @staticmethod
    def _initialize_sample_weight(
        sample_weight: np.array, n_samples: int
    ) -> np.array:
        if sample_weight is None:
            return np.ones((n_samples,), dtype=np.float64)
        return sample_weight.copy()

    def _validate_estimator(self) -> None:
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=Stree(random_state=self.random_state)
        )

    def fit(
        self, X: np.array, y: np.array, sample_weight: np.array = None
    ) -> Odte:
        # Check parameters are Ok.
        if self.n_estimators < 3:
            raise ValueError(
                f"n_estimators must be greater than 2 but got (n_estimators=\
                    {self.n_estimators})"
            )
        check_classification_targets(y)
        X, y = self._validate_data(X, y)
        sample_weight = _check_sample_weight(
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
        self._train(X, y, sample_weight)
        return self

    def _train(
        self, X: np.array, y: np.array, sample_weight: np.array
    ) -> None:
        random_box = self._initialize_random()
        random_seed = self.random_state
        n_samples = X.shape[0]
        weights = self._initialize_sample_weight(sample_weight, n_samples)
        boot_samples = self._get_bootstrap_n_samples(n_samples)
        for _ in range(self.n_estimators):
            # Build clf
            clf = clone(self.base_estimator_)
            clf.random_state = random_seed
            random_seed += 1
            self.estimators_.append(clf)
            # bootstrap
            indices = random_box.randint(0, n_samples, boot_samples)
            # update weights with the chosen samples
            weights_update = np.bincount(indices, minlength=n_samples)
            features = self._get_random_subspace(X, y)
            self.subspaces_.append(features)
            current_weights = weights * weights_update
            # train the classifier
            bootstrap = X[indices, :]
            clf.fit(
                bootstrap[:, features], y[indices], current_weights[indices]
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

    def _get_random_subspace(
        self, dataset: np.array, labels: np.array
    ) -> Tuple[int, ...]:
        features = range(dataset.shape[1])
        features_sets = list(combinations(features, self.max_features_))
        if len(features_sets) > 1:
            index = random.randint(0, len(features_sets) - 1)
            return features_sets[index]
        else:
            return features_sets[0]

    def predict(self, X: np.array) -> np.array:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.array) -> np.array:
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
