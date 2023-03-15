"""This module is used to build Random Forest algorithm from scratch."""
# pylint: disable=keyword-arg-before-vararg
from collections import Counter

import numpy as np

from src import Model
from src.decision_trees import DecisionTree


class RandomForest(Model):
    """This class is used to implement the Random Forest classifier."""

    def __init__(
        self,
        min_samples_split: int = 2,
        max_depth: int = 50,
        n_features: int = None,
        n_trees: int = 20,
        *args,
        **kwargs,
    ) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.n_trees = n_trees
        self.trees = []
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(min_num_sample={self.min_samples_split} "
            f"max_depth={self.max_depth}, "
            f"n_features={self.n_features}, "
            f"n_trees={self.n_trees})"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for _ in np.arange(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features,
            )
            # Randomly select samples
            X_sampled, y_sampled = self._bootstrap(X=X, y=y)
            # Fit and Train
            tree.fit(X=X_sampled, y=y_sampled)
            self.trees.append(tree)
        return self

    @staticmethod
    def _bootstrap(*, X: np.ndarray, y: np.ndarray) -> tuple[list[int], list[int]]:
        """This returns random samples from the data having the
        same size as the training data."""
        n_samples = X.shape[0]
        # With replace=True ensures that not all the samples are chosen
        # because a few samples will be repeated and chosen_sample == n_samples
        chosen_samples = np.random.choice(n_samples, n_samples, replace=True)
        return (X[chosen_samples, :], y[chosen_samples])

    @staticmethod
    def _get_most_common_label(*, input_: np.ndarray) -> int:
        """This returns the most common label."""
        counter = Counter(input_)
        return counter.most_common(n=1)[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        # This returns the predicted labels for each data point per tree.
        # i.e tree_0_pred, tree_1_pred, tree_2_pred, ...
        # [[0,1,1], [1,1,0], [0,0,1], ...]
        tree_preds = [tree.predict(X) for tree in self.trees]
        # But what we actually want are all the predictions by the trees
        # for each data point in a single array.
        # e.g. [[0,1,0] [1,1,0], [1,0,1], ...]
        preds = np.swapaxes(tree_preds, axis1=0, axis2=1)
        predictions = [self._get_most_common_label(input_=pred) for pred in preds]
        return np.array(predictions)
