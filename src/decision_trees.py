"""This module is used to implement Decision Trees from scratch."""
from collections import Counter
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from src.base import Model


class Node:
    """This is used to implement the nodes of a Decision Tree."""

    def __init__(
        self,
        left=None,
        right=None,
        feature: Optional[int] = None,
        threshold: Optional[int] = None,
        *,
        value: Optional[int] = None,
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
        min_samples_split: int = 2,
        max_depth: Optional[int] = 100,
        n_features: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(min_num_sample={self.min_samples_split} "
            f"max_depth={self.max_depth}, "
            f"n_features={self.n_features})"
        )

    def fit(
        self, X: npt.NDArray[Union[np.int_, np.float_]], y: npt.NDArray[Union[np.int_, np.float_]]
    ) -> None:
        self.root = self._grow_tree(X, y)  # type: ignore
        return self  # type: ignore

    def _grow_tree(
        self,
        X: npt.NDArray[Union[np.int_, np.float_]],
        y: npt.NDArray[Union[np.int_, np.float_]],
        depth: int = 0,
    ) -> Node:
        """This is used to recursively grow the tree.
        It returns a leaf node."""
        # Extract the some attributes from the input data.
        # Verify that the n_features is valid!
        n_samples, n_feats = X.shape
        self.n_features = n_feats if self.n_features is None else min(self.n_features, n_feats)
        n_K = len(np.unique(y))

        # Base case: If one of the stopping criteria is met. Return the most
        # common label (class) i.e if we have samples < min_samples_split or
        # depth >= max_depth or if we have a pure node (a single class). i.e n_K == 1
        if n_samples < self.min_samples_split or depth >= self.max_depth or n_K == 1:  # type: ignore
            leaf_node = DecisionTree._most_common_label(y=y)
            return Node(value=leaf_node)

        # Add some randomness. Select features indices at random w/o replacement
        selected_features = np.random.choice(a=n_feats, size=self.n_features, replace=False)

        # Calculate the best split (using info gain)
        best_feature, best_label_threshold = DecisionTree._determine_best_split(
            X=X, y=y, features=selected_features
        )

        # Split into nodes using the best feature and label threshold
        left_idxs, right_idxs = DecisionTree._split_into_nodes(
            X=X[:, best_feature], label_threshold=best_label_threshold
        )
        # Recursively grow the tree
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth=depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth=depth + 1)
        return Node(left=left, right=right, feature=best_feature, threshold=best_label_threshold)

    @staticmethod
    def _most_common_label(*, y: Union[npt.NDArray[Union[np.int_, np.float_]], list[int]]) -> int:
        """This is used to determine the most common label at a node."""
        counts = Counter(y)
        label = counts.most_common(n=1)[0][0]
        return label

    @staticmethod
    def _determine_best_split(
        *,
        X: npt.NDArray[Union[np.int_, np.float_]],
        y: npt.NDArray[Union[np.int_, np.float_]],
        features: npt.NDArray[np.int_],
    ):
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
                    X=current_X_arr, label_threshold=thresh, y=y
                )
                if info_gain > best_gain:
                    best_gain = info_gain  # type: ignore
                    best_feat, best_label_threshold = feat, thresh

        return best_feat, best_label_threshold

    @staticmethod
    def _calculate_entropy(y: npt.NDArray[Union[np.int_, np.float_]]) -> float:
        """This is used to calculate the entropy at a node."""
        total, counts = len(y), np.bincount(y)
        probs: list[float] = counts / total  # type: ignore
        entropy = -np.sum([(p_k * np.log2(p_k)) for p_k in probs if p_k > 0])
        return entropy

    @staticmethod
    def _calculate_information_gain(
        *,
        X: npt.NDArray[Union[np.int_, np.float_]],
        label_threshold: int,
        y: npt.NDArray[Union[np.int_, np.float_]],
    ) -> float:
        """This returns the information gain of a feature.
        It ranges between 0 and 1."""
        total_nodes = X.shape[0]
        # Calculate the entropy of the parent
        parent_entropy = DecisionTree._calculate_entropy(y=y)

        # Calculate the entropy of the child nodes
        left_idxs, right_idxs = DecisionTree._split_into_nodes(X=X, label_threshold=label_threshold)  # type: ignore
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
        info_gain = parent_entropy - (weighted_left_entropy + weighted_right_entropy)  # type: ignore
        return info_gain

    @staticmethod
    def _split_into_nodes(
        *, X: npt.NDArray[np.int_], label_threshold: int
    ) -> tuple[list[int], list[int]]:
        """This is used to split the node into left and right nodes
        using X and best_label_threshold. It returns a tuple of lists
        which represent the observations.

        Params:
            X: 2-D array
            label_threshold: int
        """
        # Select the indices of the array that satisfy the condition
        left_idxs = np.argwhere(X <= label_threshold).flatten()
        right_idxs = np.argwhere(X > label_threshold).flatten()
        return (left_idxs, right_idxs)  # type: ignore

    def predict(
        self, X: npt.NDArray[Union[np.int_, np.float_]]
    ) -> npt.NDArray[Union[np.int_, np.float_]]:
        y_pred = [DecisionTree._traverse_tree(x=x_, node=self.root) for x_ in X]  # type: ignore
        return np.array(y_pred)

    @staticmethod
    def _traverse_tree(*, x: npt.NDArray[np.int_], node: Node):
        """This is used to traverse the DecisionTree nodes.
        It returns the predicted value for a given observation."""
        # Base case: If it's a leaf node
        if node.is_leaf_node():
            return node.value

        # If the value is less than the threshold, traverse left recursively
        if x[node.feature] <= node.threshold:  # type: ignore
            return DecisionTree._traverse_tree(x=x, node=node.left)
        # If the value is greater than the threshold, traverse right recursively
        return DecisionTree._traverse_tree(x=x, node=node.right)
