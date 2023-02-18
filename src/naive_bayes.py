"""This module is used to build Naive Bayes algorithm from scratch."""

import numpy as np

from src import Model


class NaiveBayes(Model):
    """This is an implementation of Naive Bayes algorithm."""

    def __init__(self) -> None:
        super().__init__()
        self.means = None
        self.variances = None
        self.priors = None
        self.K = None
        self.n_K = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_classes={self.n_K!r}, prior={self.priors!r})"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Init the parameters
        n_samples, n_features = X.shape
        self.K = np.unique(y)
        self.n_K = len(self.K)

        # Init params for the classes. i.e if k is 2 then, k=0 or 1
        self.means = np.zeros((self.n_K, n_features))  # Matrix
        self.variances = np.zeros((self.n_K, n_features))  # Matrix
        self.priors = np.zeros((self.n_K)).reshape(-1, 1)  # Column vector

        # Compute the parameters for each class.
        # Calculate the mean, variance and priors given each class.
        for k in self.K:
            X_k = X[k == y]
            self.means[k, :] = np.mean(X_k, axis=0)
            self.variances[k, :] = np.var(X_k, axis=0)
            self.priors[k] = X_k.shape[0] / float(n_samples)
        return self

    def _predict(self, x: np.ndarray) -> np.ndarray:
        posteriors = []
        # Shape of x: (1, n_features)
        for k in self.K:
            log_prior = np.log(self.priors[k])
            posterior = np.sum(np.log(self._prob_density_func(x, k))) + log_prior
            posteriors.append(posterior)

        # This returns : 0 or 1 since the list `posteriors` has a size of 2.
        # i.e [posterior_cl_0, posterior_cl_1] and np.argmax returns
        # the index that has the maximum value (which is 0 or 1).
        return np.argmax(posteriors)

    def _prob_density_func(self, x: np.ndarray, k: int) -> float:
        """This is used to calculate the Gaussian Probability Density Function\n
        given the class. i.e for class=0 or 1"""
        # Shape of x, mean and variance: (1, n_features)
        mean, variance = self.means[k], self.variances[k]
        numerator = np.exp(-np.square(x - mean) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator

    def predict(self, X: np.ndarray) -> np.ndarray:
        """This is used for making predictions for ALL the training examples."""
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
