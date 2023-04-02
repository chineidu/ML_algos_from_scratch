"""This module is used to implement Decision Trees from scratch."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from src.base import Model


class KMeans(Model):
    """This class is used for creating K-clusters from scratch."""

    def __init__(self, K: int = 5, max_iters: int = 300, plot_steps: bool = False) -> None:
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in np.arange(self.K)]  # type: ignore
        self.centroids = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K}, max_iters={self.max_iters})"

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike = None) -> npt.NDArray[np.int_]:
        """This is used to train the model."""
        n_samples = X.shape[0]

        # Init centroids randomly
        random_idxs = np.random.choice(n_samples, self.K, replace=False)
        self.centroids = [X[idx] for idx in random_idxs]  # type: ignore

        # Optimize the algorithm:
        # 1: Create clusters by assigning the samples to the closest centroid.
        # 2: Update the centroids and compare the new and old centroids
        # 3: Check if the algorithm has converged. i.e whether the centroids
        # are still changing.
        self.n_iter_ = 1  # pylint: disable=attribute-defined-outside-init

        for idx in np.arange(self.max_iters):
            # 1
            self.clusters = self._create_clusters(X, self.centroids)  # type: ignore

            if self.plot_steps:
                self._plot(X)
            # 2
            old_centroids = self.centroids
            self.centroids = self._get_centroids(X, self.clusters)

            # 3
            if self._is_converged(self.centroids, old_centroids):  # type: ignore
                break
            self.n_iter_ += idx

            if self.plot_steps:
                self._plot(X)

        return self._get_cluster_labels(X, self.clusters)

    def predict(self, X: npt.ArrayLike, y: npt.ArrayLike = None) -> npt.NDArray[np.int_]:
        """This is used to classify the labels of the samples."""
        return self.fit(X, y)

    def fit_predict(self, X: npt.ArrayLike, y: npt.ArrayLike = None) -> npt.NDArray[np.int_]:
        """This is used to train and make predictions."""
        self.fit(X, y)
        return self.predict(X, y)

    def _create_clusters(self, X: npt.ArrayLike, centroids: list[int]):
        """This is used to create new clusters by assigning the
        samples to the closest centroid.
        """
        self.clusters = [[] for _ in np.arange(self.K)]

        # For each data point/sample, calculate the nearest centroid (n_centroid)
        # to that sample and assign the sample to n_centroid and store in the
        # list of clusters.
        for idx, sample in enumerate(X):
            centroid_idx = self._get_nearest_centroid(sample, centroids)
            self.clusters[centroid_idx].append(idx)
        return self.clusters

    def _get_centroids(self, X: npt.ArrayLike, clusters: list[list[int]]):
        """This is used to calculate the mean position of each cluster.
        i.e. calculate the mean position of all the data points w/in each cluster.
        """
        # Matrix (K, n_feature)
        n_features = X.shape[1]
        centroids = np.zeros(shape=(self.K, n_features))

        for cluster_idx, _cluster in enumerate(clusters):
            cluster_mean = np.mean(X[_cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids: list[int], old_centroids: list[int]) -> bool:
        """This checks whether the centroids are changing or not."""
        # Compare the centroids
        return np.array_equal(centroids, old_centroids)

    def _get_cluster_labels(
        self, X: npt.ArrayLike, clusters: list[list[int]]
    ) -> npt.NDArray[np.int_]:
        """This returns the labels of the clusters that have been assigned
        to each cluster."""
        n_samples = X.shape[0]
        labels = np.empty(shape=(n_samples))

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    @staticmethod
    def _calculate_euclidean_distance(*, point_a: float, point_b: float) -> float:
        """This returns the Euclidean distance between 2 points."""
        return np.sqrt(np.sum(np.square(point_a - point_b)))

    def _get_nearest_centroid(self, sample, centroids: list[int]) -> int:
        """This returns the nearest centroid to a sample/data point."""
        # Calculate the distance of the sample from each centroid
        # and select the centroid with the min. distance
        distances = [
            self._calculate_euclidean_distance(point_a=sample, point_b=_centr)
            for _centr in centroids
        ]
        closest_centroid = np.argmin(distances)
        return closest_centroid

    def _plot(self, X: npt.ArrayLike) -> None:
        """This is used to visualize the clusters and the labels during
        the optimization process of the algorithm."""

        fig, ax = plt.subplots(figsize=(12, 8))

        for _cluster in self.clusters:
            point = X[_cluster].T
            ax.scatter(*point)

        for point in self.centroids:  # type: ignore
            ax.scatter(*point, marker="x", color="k", linewidths=5)

        fig.get_tight_layout()
        plt.show()
