import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.linear_regression import LinearRegression


def cal_mse(y_true, y_pred):
    mse = np.mean(np.square(y_true - y_pred))
    return mse


# # visualize the relationship between the predictor and the target
# plt.figure(figsize=(10, 6))
# plt.scatter(X[:, 0], y, s=20)
# plt.show()

def run():
    """Run the entire file"""
    # create a synthetic data
    data = make_regression(n_samples=300, n_features=1, noise=27, random_state=4)
    X, y = data

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    # instantiate model
    lin_reg = LinearRegression()

    # train model
    lin_reg.fit(X_train, y_train)

    # make predictions
    y_pred = lin_reg.predict(X_test)

    mse = cal_mse(y_test, y_pred)
    return mse



if __name__ == '__main__':
    mse = run()
    print(mse)