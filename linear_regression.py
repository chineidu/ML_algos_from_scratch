import numpy as np
import pandas as pd

#### Linear Regression (from scratch)
class LinearRegression:
	def __init__(self, learning_rate: int=0.0001, n_iters: int=10_000):
		"""Initialize the parameters"""
		self.learning_rate = learning_rate
		self.n_iters = n_iters
		self.wieghts = 0
		self.bias = 0

	def fit(self, X, y):
		"""Train the model."""
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)  # initialize the weights with zeros
		
		for i in np.arange(self.n_iters):
			# make predictions
			y_pred = self.bias + np.dot(X, self.weights)

			# gradient descent
			d_weights = (1 / n_samples) * (2 * np.dot(X.T, (y_pred - y)))
			d_bias = (1 / n_samples) * (2 * np.sum(y_pred - y))

			# update the parameters
			self.weights -= self.learning_rate * d_weights
			self.bias -= self.learning_rate * d_bias

	def predict(self, X):
		y_predicted = self.bias + np.dot(X, self.weights)
		return y_predicted