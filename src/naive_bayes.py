"""This module is used to build Linear Regression from scratch."""

import numpy as np

from src import Model


class NaiveBayes(Model):
    """This is an implementation of Naive Bayes algorithm."""

    def __init__(self) -> None:
        super().__init__()
        self.mean = None
        self.variance = None
        self.priors = None
        self.classes = None
        self.n_classes = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(prior={self.priors!r})"

    def fit(self, X: np.ndarray, y=np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.classes = np.unique(y)  # 0 or 1
        self.n_classes = len(self.classes)  # 2

        # Initialize parameters: mean, variance and prior
        # with zeros for each class
        self.mean = np.zeros((self.n_classes, n_features))  # Matrix
        self.variance = np.zeros((self.n_classes, n_features))  # Matrix
        self.priors = np.zeros(self.n_classes).reshape(-1, 1)  # Column vector

        # Setup class conditional probability
        # since the inputs depend on the class.
        # Calculate the mean, variance and the prior each class.
        for cl_ in self.classes:
            X_cl = X[cl_ == y]
            self.mean[cl_, :] = np.mean(X_cl, axis=0)  # Scalar
            self.variance[cl_, :] = np.var(X_cl, axis=0)  # Scalar
            self.priors[cl_] = X_cl.shape[0] / float(n_samples)  # Scalar
        return self

    def __predict(self, x: np.ndarray) -> int:
        """This is used to make prediction using the argmax of the posteriors\n
        for a training example.

        Params:
            x (np.ndarray): A single training example with shape: (1, n_features).

        Note:
            np.argmax: This returns the indices of the maximum values along an axis.

        Returns:
            pred (int): The prediction of the model for a training example.
        """
        # After the iteration, the list has size = n_classes
        # i.e for both classes.
        posteriors = []
        for cl_ in self.classes:
            log_prior = np.log(self.priors[cl_])
            log_class_conditional_prob = np.sum(np.log(self.__prob_density_func(x, cl_)))
            posterior = log_class_conditional_prob + log_prior
            posteriors.append(posterior)

        # This returns : 0 or 1 since the list `posteriors` has a size of 2.
        # i.e [posterior_cl_0, posterior_cl_1] and np.argmax returns
        # the index that has the maximum value (which is 0 or 1).
        pred = np.argmax(posteriors)
        return pred

    def __prob_density_func(self, x: np.ndarray, cl_: int) -> float:
        """This is used to calculate the Gaussian Probability Density Function\n
        given the class. i.e for class=0 or 1"""
        mean, variance = self.mean[cl_], self.variance[cl_]
        numerator = np.exp(-(np.square(x - mean)) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator

    def predict(self, X: np.ndarray) -> np.ndarray:
        """This is used to make predictions using the argmax of the posteriors\n
        for ALL the training examples."""
        y_pred = [self.__predict(x=x) for x in X]
        return np.array(y_pred)
