"""This module is used to build Linear Regression from scratch."""

import numpy as np

from src import Model


class LogisticRegression(Model):
    """This is an implementation of Logistic Regression."""

    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1_000) -> None:
        self.l_rate = learning_rate
        self.n_iters = n_iters
        self.weight = None
        self.bias = None
        self.THRESH = 0.5

    def __repr__(self) -> str:
        return f"{__class__.__name__}(learning_rate={self.l_rate!r}, " f"n_iters={self.n_iters:,})"

    def fit(self, X=np.ndarray, y=np.ndarray) -> None:
        n_samples, n_features = X.shape

        # Step 1: Initialize the weight and bias
        self.weight = np.zeros((n_features))  # Vector
        self.bias = 0  # Scalar

        # Step 2: Estimate the y_value given the data points
        # Note: shape of X: (n_samples, n_features) and shape of weight: (n_features, 1)
        # Dot product: (A, B) x (B, C). i.e the inner dimensions MUST be equal.
        # For more info check: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
        for _ in range(self.n_iters):
            # Make predictions. Convert the continuous variable
            # to a number between 0 and 1.
            y_hat = np.dot(X, self.weight) + self.bias
            y_pred = self._sigmoid(y_hat)

            # Step 3: Calculate the change in weight and bias values for each training
            # example using gradient descent.
            # shape of x_i: (1, n_features), shape of (y - y_hat): (1,) a rank1 array
            # so we need to transpose x_i. so that shape of x_i.T: (n_features, 1)
            # Note that np.dot also performs the summation.
            dw = (1 / n_samples) * 2 * (np.dot(X.T, (y_pred - y)))
            db = 2 * np.mean(y_pred - y)

            # Step 4: Update the parameters
            self.weight -= self.l_rate * dw
            self.bias -= self.l_rate * db
        return self

    def _sigmoid(self, y_hat: np.ndarray) -> np.ndarray:
        """This returns a number between 0 and 1. in an array"""
        _y_pred = 1 / (1 + np.exp(-y_hat))
        return _y_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """This is used to make predictions."""
        y_hat = np.dot(X, self.weight) + self.bias
        _y_pred = self._sigmoid(y_hat)
        y_pred = [1 if val > self.THRESH else 0 for val in _y_pred]
        return np.array(y_pred)
