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

    def __init__(
        self, left, right, feature: int, threshold: int, *, value: Optional[int] = None
    ) -> None:
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value

    def is_leaf_node(self) -> bool:
        """This returns True if the Node is a leaf node
        otherwise, False."""
        return self.value is not None


class DecisionTree(Model):
    """This is used to implement the Decision Trees algorithm."""

    def __init__(
        self,
        min_num_samples: int,
        max_depth: Optional[int] = 100,
        num_features: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.min_num_samples = min_num_samples
        self.max_depth = max_depth
        self.num_features = num_features
        self.root = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(min_num_sample={self.min_num_samples} "
            f"max_depth={self.max_depth}, "
            f"num_features={self.num_features})"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.root = self._grow_tree(X, y)
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
            leaf_node = DecisionTree._most_common_label(y=y)
            return Node(value=leaf_node)

        # Add some randomness. Select features indices at random w/o replacement
        selected_features = np.random.choice(a=n_feats, size=self.num_features, replace=False)

        # Calculate the best split (using info gain)
        best_feature, best_label_threshold = self._determine_best_split(X, y, selected_features)

        # Split into nodes using the best feature and label threshold
        left_node, right_node = DecisionTree._split_into_nodes(
            X=X[:, best_feature], best_label_threshold=best_label_threshold
        )
        # Recursively grow the tree
        left = self._grow_tree(X[left_node, :], y[left_node], depth=depth + 1)
        right = self._grow_tree(X[right_node, :], y[right_node], depth=depth + 1)
        return Node(left=left, right=right, feature=best_feature, threshold=best_label_threshold)

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

        Params:
            X: 2-D array
            features: 1-D array
        """
        best_feat, best_label_threshold = None, None
        best_gain = -1  # (since it ranges from 0 to 1)

        # For each feature, determine the best_feature and best_label_threshold
        # that will be used for the split.
        for feat in features:
            current_X_arr = X[:, feat]
            all_labels = np.unique(current_X_arr)

            for thresh in all_labels:
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

    @staticmethod
    def _calculate_information_gain(
        X: np.ndarray, best_label_threshold: int, y: np.ndarray
    ) -> float:
        """This returns the information gain of a feature.
        It ranges between 0 and 1."""
        total_nodes = X.shape[0]
        # Calculate the entropy of the parent
        parent_entropy = DecisionTree._calculate_entropy(y=y)

        # Calculate the entropy of the child nodes
        left_idxs, right_idxs = DecisionTree._split_into_nodes(X, best_label_threshold)
        num_left_idxs, num_rigft_idxs = len(left_idxs), len(right_idxs)

        # If you don't have binary branches there's no need to split the node w/that feature.
        if num_left_idxs == 0 or num_rigft_idxs == 0:
            info_gain = 0

        weighted_left_entropy = (num_left_idxs / total_nodes) * DecisionTree._calculate_entropy(
            y[left_idxs]
        )
        weighted_right_entropy = (num_rigft_idxs / total_nodes) * DecisionTree._calculate_entropy(
            y[right_idxs]
        )
        info_gain = parent_entropy - (weighted_left_entropy + weighted_right_entropy)
        return info_gain

    @staticmethod
    def _split_into_nodes(X: np.ndarray, best_label_threshold: int) -> tuple[list[int], list[int]]:
        """This is used to split the node into left and right nodes
        using X and best_label_threshold. It returns a tuple of lists
        which represent the observations.

        Params:
            X: 2-D array
            best_label_threshold: int
        """
        # Select the indices of the array that satisfy the condition
        left_idxs = np.argwhere(X <= best_label_threshold).flatten()
        right_idxs = np.argwhere(X > best_label_threshold).flatten()
        return (left_idxs, right_idxs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = [DecisionTree._traverse_tree(x=x_, node=self.root) for x_ in X]
        return np.array(y_pred)

    @staticmethod
    def _traverse_tree(*, x: np.ndarray, node: Node):
        """This is used to traverse the DecisionTree nodes.
        It returns the predicted value for a given observation."""
        # Base case: If it's a leaf node:
        if node.is_leaf_node():
            return node.value

        # If the value is less than the threshold, traverse left recursively
        if x[node.feature] <= node.threshold:
            return DecisionTree._traverse_tree(x=x, node=node.left)
        # If the value is greater than the threshold, traverse right recursively
        return DecisionTree._traverse_tree(x=x, node=node.right)
