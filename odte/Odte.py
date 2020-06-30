"""
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.1"
Build a forest of oblique trees based on STree
"""

import random
from typing import Union
from itertools import combinations
import numpy as np
from sklearn.utils import check_consistent_length
from sklearn.metrics._classification import _weighted_sum, _check_targets
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import clone, ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    _check_sample_weight,
)

from stree import Stree


class Odte(BaseEnsemble, ClassifierMixin):
    def __init__(
        self,
        base_estimator=None,
        random_state: int = None,
        max_features: Union[str, int, float] = 1.0,
        max_samples: Union[int, float] = None,
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

    def _more_tags(self) -> dict:
        return {"requires_y": True}

    def _initialize_random(self) -> np.random.mtrand.RandomState:
        if self.random_state is None:
            return np.random.mtrand._rand
        return np.random.RandomState(self.random_state)

    @staticmethod
    def _initialize_sample_weight(
        sample_weight: np.array, n_samples: int
    ) -> np.array:
        if sample_weight is None:
            return np.ones((n_samples,), dtype=np.float64)
        return sample_weight.copy()

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=Stree(random_state=self.random_state)
        )

    def fit(
        self, X: np.array, y: np.array, sample_weight: np.array = None
    ) -> "Odte":
        # Check parameters are Ok.
        if self.n_estimators < 3:
            raise ValueError(
                f"n_estimators must be greater than 2 but got (n_estimators=\
                    {self.n_estimators})"
            )
        check_classification_targets(y)
        X, y = check_X_y(X, y)
        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=np.float64
        )
        check_classification_targets(y)
        # Initialize computed parameters
        #  Build the estimator
        self.n_features_in_ = X.shape[1]
        self.n_features_ = X.shape[1]
        self.max_features_ = self._initialize_max_features()
        self._validate_estimator()
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        self.estimators_ = []
        self.subspaces_ = []
        self._train(X, y, sample_weight)
        return self

    def _train(
        self, X: np.array, y: np.array, sample_weight: np.array
    ) -> "Odte":
        random_box = self._initialize_random()
        n_samples = X.shape[0]
        weights = self._initialize_sample_weight(sample_weight, n_samples)
        boot_samples = self._get_bootstrap_n_samples(n_samples)
        for _ in range(self.n_estimators):
            # Build clf
            clf = clone(self.base_estimator_)
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

    def _get_bootstrap_n_samples(self, n_samples) -> int:
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
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    "Invalid value for max_features. "
                    "Allowed string values are 'auto', "
                    "'sqrt' or 'log2'."
                )
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, int):
            max_features = abs(self.max_features)
        else:  # float
            if self.max_features > 0.0:
                max_features = max(
                    1, int(self.max_features * self.n_features_)
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
    ) -> np.array:
        features = range(dataset.shape[1])
        features_sets = list(combinations(features, self.max_features_))
        if len(features_sets) > 1:
            index = random.randint(0, len(features_sets) - 1)
            return features_sets[index]
        else:
            return features_sets[0]

    def predict(self, X: np.array) -> np.array:
        proba = self.predict_proba(X)
        return self.classes_.take((np.argmax(proba, axis=1)), axis=0)

    def predict_proba(self, X: np.array) -> np.array:
        check_is_fitted(self, ["estimators_"])
        # Input validation
        X = check_array(X)
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))
        for tree, features in zip(self.estimators_, self.subspaces_):
            n_samples = X.shape[0]
            result = np.zeros((n_samples, self.n_classes_))
            predictions = tree.predict(X[:, features])
            for i in range(n_samples):
                result[i, predictions[i]] += 1
        return result

    def score(
        self, X: np.array, y: np.array, sample_weight: np.array = None
    ) -> float:
        check_classification_targets(y)
        X, y = check_X_y(X, y)
        y_pred = self.predict(X).reshape(y.shape)
        # Compute accuracy for each possible representation
        _, y_true, y_pred = _check_targets(y, y_pred)
        check_consistent_length(y_true, y_pred, sample_weight)
        score = y_true == y_pred
        return _weighted_sum(score, sample_weight, normalize=True)
