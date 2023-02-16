import numpy as np

#### Linear Regression (from scratch)
class LinearRegression:
    def __init__(self, n_iters: int = 10_000, learning_rate: int = 0.0001):
        self.n_iters = n_iters
        self.l_r = learning_rate
        self.weights = 0
        self.bias = 0

    def fit(self, X, y):
        # set the parameters
        n_samples, n_features = X.shape
        # assign the weights using the n_features
        self.weights = np.zeros(n_features)

        # gradient descent
        for _ in np.arange(self.n_iters):
            y_pred = self.bias + np.dot(X, self.weights)

            # Comute the change in values
            # np.dot(X.T, (y_pred - y) finds the product and the sum. For more info
            # check: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
            dw = (2 * np.dot(X.T, (y_pred - y))) / n_samples
            db = (2 * np.sum(y_pred - y)) / n_samples
            # update the values
            self.weights -= self.l_r * dw
            self.bias -= self.l_r * db

    def predict(self, X):
        y_pred = self.bias + np.dot(X, self.weights)
        return y_pred

    def _mse(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the Mean Squared Error of the model."""
        mse = np.mean(np.square(y_true - y_pred))
        return mse
