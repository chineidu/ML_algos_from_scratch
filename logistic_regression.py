import numpy as np

#### Logistic Regression (from scratch)
class LogisticRegression:
    def __init__(self, n_iters: int=10_000, learning_rate: int=0.0001):
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
            y_lin = self.bias + np.dot(X, self.weights)
            y_pred = self._sigmoid_func(y_lin)
            
            # comute the change in values
            dw = (2 * np.dot(X.T, (y_pred - y))) / n_samples
            db = (2 * np.sum(y_pred - y)) / n_samples
            # update the values
            self.weights -= self.l_r * dw 
            self.bias -= self.l_r * db 
            
    def predict(self, X):
        y_lin = self.bias + np.dot(X, self.weights)
        y_pred = self._sigmoid_func(y_lin)
        # make classifications
        clf = [1 if i > 0.50 else 0 for i in y_pred]
        return np.array(clf)
    
    def _sigmoid_func(self, val: int) -> int:
        return 1 / (1 + np.exp(-val))
    
    def _accuracy(self,  y_true: np.array, y_pred: np.array) -> float:
        """Calculate the model accuracy."""
        acc = np.sum(y_true == y_pred) / len(y_true)
        return round(acc, 2)



