"""
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
__version__ = "0.1"
Build a forest of oblique trees based on STree
"""

import numpy as np

from sklearn.utils import check_consistent_length
from sklearn.metrics._classification import _weighted_sum, _check_targets
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import mode
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    _check_sample_weight,
)

from stree import Stree


class Odte(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        random_state: int = None,
        C: int = 1,
        n_estimators: int = 100,
        max_iter: int = 1000,
        max_depth: int = None,
        min_samples_split: int = 0,
        bootstrap: bool = True,
        split_criteria: str = "min_distance",
        tol: float = 1e-4,
        gamma="scale",
        degree: int = 3,
        kernel: str = "linear",
        max_features="auto",
        max_samples=None,
    ):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.max_features = max_features
        self.max_samples = max_samples
        self.estimator_params = dict(
            C=C,
            random_state=random_state,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            split_criteria=split_criteria,
            kernel=kernel,
            max_iter=max_iter,
            tol=tol,
            degree=degree,
            gamma=gamma,
        )

    def _initialize_random(self) -> np.random.mtrand.RandomState:
        if self.random_state is None:
            return np.random.mtrand._rand
        else:
            return np.random.RandomState(self.random_state)

    def _initialize_sample_weight(
        self, sample_weight: np.array, n_samples: int
    ) -> np.array:
        if sample_weight is None:
            return np.ones((n_samples,), dtype=np.float64)
        else:
            return sample_weight.copy()

    def fit(
        self, X: np.array, y: np.array, sample_weight: np.array = None
    ) -> "Odte":
        # Check parameters are Ok.
        if self.n_estimators < 3:
            raise ValueError(
                f"n_estimators must be greater than 3... got (n_estimators=\
                    {self.n_estimators:f})"
            )
        # the rest of parameters are checked in estimator
        check_classification_targets(y)
        X, y = check_X_y(X, y)
        sample_weight = _check_sample_weight(sample_weight, X)
        check_classification_targets(y)
        # Initialize computed parameters
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        self.estimators_ = []
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
            clf = Stree().set_params(**self.estimator_params)
            self.estimators_.append(clf)
            # bootstrap
            indices = random_box.randint(0, n_samples, boot_samples)
            # update weights with the chosen samples
            weights_update = np.bincount(indices, minlength=n_samples)
            current_weights = weights * weights_update
            # train the classifier
            clf.fit(X[indices, :], y[indices], current_weights[indices])

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

    def predict(self, X: np.array) -> np.array:
        # todo
        check_is_fitted(self, ["estimators_"])
        # Input validation
        X = check_array(X)
        result = np.empty((X.shape[0], self.n_estimators))
        for index, tree in enumerate(self.estimators_):
            result[:, index] = tree.predict(X)
        return mode(result, axis=1).mode.ravel()

    def score(
        self, X: np.array, y: np.array, sample_weight: np.array = None
    ) -> float:
        # todo
        check_is_fitted(self, ["estimators_"])
        check_classification_targets(y)
        X, y = check_X_y(X, y)
        y_pred = self.predict(X).reshape(y.shape)
        # Compute accuracy for each possible representation
        _, y_true, y_pred = _check_targets(y, y_pred)
        check_consistent_length(y_true, y_pred, sample_weight)
        score = y_true == y_pred
        return _weighted_sum(score, sample_weight, normalize=True)
