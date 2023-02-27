"""This module is used to implement Decision Trees from scratch."""
from collections import Counter
from typing import Optional

import numpy as np

from src.base import Model

# pylint: disable=unnecessary-pass
# pylint: disable=unused-variable
# pylint: disable=useless-return
# pylint: disable=assignment-from-none


class Node:
    """This is used to implement the nodes of a Decision Tree."""

    def __init__(self, left, right, feature: int, threshold: int, *, value: int) -> None:
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value

    def _is_leaf_node(self) -> bool:
        """This returns True if the Node is a leaf node
        otherwise, False."""
        return self.value is not None


class DecisionTree(Model):
    """This is used to implement the Decision Trees algorithm."""

    def __init__(
        self,
        min_num_samples: int,
        max_depth: Optional[int] = None,
        num_features: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.min_num_samples = min_num_samples
        self.max_depth = max_depth
        self.num_features = num_features

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(min_num_sample={self.min_num_samples} "
            f"max_depth={self.max_depth}, "
            f"num_features={self.num_features})"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        return self

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> None:
        """This is used to recursively grow the tree.
        It returns a leaf node."""
        # Extract the some attributes from the input data.
        # Verify that the num_features is valid!
        n_samples, n_feats = X.shape
        self.num_features = (
            n_feats if self.num_features is None else min(self.num_features, n_feats)
        )
        n_K = len(np.unique(y))
        # Base case: If one of the stopping criteria is met. Return the most
        # common label (class) i.e if we have samples < min_num_samples or
        # depth >= max_depth or if we have a pure node (a single class). i.e n_K == 1
        if n_samples < self.min_num_samples or depth >= self.max_depth or n_K == 1:
            return Node(value=DecisionTree._most_common_label(y=y))

        # Add some randomness. Select features at random w/o replacement
        selected_features = np.random.choice(a=n_feats, size=self.num_features, replace=False)

        # Calculate the best split (using info gain)
        best_feature, best_label_threshold = self._determine_best_split(selected_features)
        pass

    @staticmethod
    def _most_common_label(*, y: np.ndarray) -> int:
        """This is used to determine the most common label at a node."""
        counts = Counter(y)
        label = counts.most_common(n=1)[0][0]
        return label

    @staticmethod
    def _determine_best_split(X: np.ndarray, y: np.ndarray, features: np.ndarray):
        """This is used to determine the best feature and threshold
        label for splitting a node using information gain.

        X: 2-D array
        features: 1-D array
        """
        best_feat, best_label_threshold = None, None
        # For each feature, determine the best_feature and best_label_threshold
        # that will be used for the split.
        for feat in features:
            current_X_arr = X[:, feat]
            all_labels = np.unique(current_X_arr)

            for thresh in all_labels:
                best_gain = -1  # (since it ranges from 0 to 1)
                # Compare the info gain with the best_gain and update the
                # best_gain if possible (i.e when info_gain > best_gain)
                # best_feat and best_label_threshold.
                info_gain = DecisionTree._calculate_information_gain(
                    current_X_arr, best_label_threshold, y
                )
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feat, best_label_threshold = feat, thresh

        return best_feat, best_label_threshold

    @staticmethod
    def _calculate_entropy(y: np.ndarray) -> float:
        """This is used to calculate the entropy at a node."""
        total, counts = len(y), np.bincount(y)
        probs: list[float] = counts / total
        entropy = -np.sum([(p_k * np.log2(p_k)) for p_k in probs if p_k > 0])
        return entropy

    def _calculate_information_gain(
        self, X: np.ndarray, best_label_threshold: int, y: np.ndarray
    ) -> float:
        """This returns the information gain of a feature."""
        total_nodes = len(y)
        # Calculate the entropy of the parent
        parent_entropy = DecisionTree._calculate_entropy(y=y)

        # Calculate the entropy of the child nodes
        left_node, right_node = DecisionTree._split_into_nodes(X, best_label_threshold)
        # left_entropy =

        # right entropy =
        return

    @staticmethod
    def _split_into_nodes(X: np.ndarray, best_label_threshold: int) -> tuple(list[int], list[int]):
        """This is used to split the node into left and right nodes
        using the X_arr and best_label_threshold.

        X: 2-D array
        best_label_threshold: int
        """
        # Select the indices of the array that satisfy the condition
        left_node = np.argwhere(X <= best_label_threshold).flatten()
        right_node = np.argwhere(X > best_label_threshold).flatten()
        return (left_node, right_node)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
        return
