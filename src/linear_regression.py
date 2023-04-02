"""This module is used to build Linear Regression from scratch."""
from typing import Union

import numpy as np
import numpy.typing as npt

from src import Model


class LinearRegression(Model):
    """This is an implementation of linear regression."""

    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1_000) -> None:
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(learning_rate={self.learning_rate!r}, "
            f"n_iters={self.n_iters:,})"
        )

    def fit(
        self, X: npt.NDArray[Union[np.int_, np.float_]], y: npt.NDArray[Union[np.int_, np.float_]]
    ) -> None:
        n_samples, n_features = X.shape

        # Step 1: Initialize the weight and bias
        self.weight = np.zeros((n_features))  # Vector
        self.bias = 0  # type: ignore

        # Step 2: Estimate the y_value given the data points
        # Note: shape of X: (n_samples, n_features) and shape of weight: (n_features, 1)
        # Dot product: (A, B) x (B, C). i.e the inner dimensions MUST be equal.
        # For more info check: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weight) + self.bias

            # Step 3: Calculate the change in weight and bias values for each training
            # example using gradient descent.
            # shape of x_i: (1, n_features), shape of (y - y_hat): (1,) a rank1 array
            # so we need to transpose x_i. so that shape of x_i.T: (n_features, 1)
            # Note that np.dot also performs the summation.
            dw = (1 / n_samples) * 2 * (np.dot(X.T, (y_pred - y)))
            db = 2 * np.mean(y_pred - y)

            # Step 4: Update the parameters
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self  # type: ignore

    def predict(
        self, X: npt.NDArray[Union[np.int_, np.float_]]
    ) -> npt.NDArray[Union[np.int_, np.float_]]:
        # Step 5. Use the updated parameters to make predictions.
        y_pred = np.dot(X, self.weight) + self.bias
        return y_pred

    @staticmethod
    def calculate_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """This is used to calculate the mean square error."""
        mse = np.mean(np.square(y_true - y_pred))
        return round(mse, 2)
