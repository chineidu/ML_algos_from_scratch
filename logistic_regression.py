import numpy as np

class LogisticRegression:
	def __init__(self, learning_rate: int=0.0001, n_iters: int=10_000):
		self.learning_rate = learning_rate
		self.n_iters = n_iters
		self.weights = 0
		self.bias = 0


	def fit(self, X, y):
		# train the model
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)

		# gradient descent
		for _ in np.arange(self.n_iters):
			# linear model
			y_linear = self.bias + np.dot(X, self.weights)
			
			# approximate y_linear to have values between 0 and 1
			y_pred = 1 / (1 + np.exp(y_linear))   # sigmoid function

			d_weights = (1 / n_samples) * (2 * np.dot(X.T, (y_pred - y)))
			d_bias = (1 / n_samples) * (2 * np.sum(y_pred, y))

			# update the values
			self.weights -= self.learning_rate * d_weights
			self.bias-= self.learning_rate * d_bias

		def predict(self, X):
			"""Make classifications."""
			y_linear = self.bias + np.dot(X, self.weights)
			y_pred = y_pred = 1 / (1 + np.exp(y_linear))
			clf = [1 if val > 0.5 else 0 for val in y_pred]