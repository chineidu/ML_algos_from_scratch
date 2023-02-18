"""This module is used to test the linear algotithms."""
from unittest.mock import Mock

import numpy as np

from src.logistic_regression import LogisticRegression


def test_linear_regression(continuous_data: np.ndarray, mock_linear_model: Mock) -> None:
    """This is used to test the linear regression algotithm."""
    # Given
    X_train, X_test, y_train, y_test = continuous_data
    learning_rate, n_iters = 0.001, 2_000
    expected_mse, expected_pred_array = 125, np.array([20.02, 25.08, 32.00, 18.17])
    LinearRegression = mock_linear_model

    # When
    lin_reg = LinearRegression(learning_rate=learning_rate, n_iters=n_iters)
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    mse = lin_reg.calculate_mean_squared_error(y_test, y_pred)

    # Then
    assert expected_mse == mse
    assert np.array_equal(expected_pred_array, y_pred)


def test_logistic_regression(classif_data: np.ndarray, mock_log_model_predict: Mock) -> None:
    """This is used to test the logistic regression algotithm."""
    # Given
    X_train, X_test, y_train, _ = classif_data
    learning_rate, n_iters = 0.001, 2_000
    expected_pred_array = np.array([1, 0, 1, 1, 1, 1])

    # When
    log_model = LogisticRegression(learning_rate=learning_rate, n_iters=n_iters)
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_test)

    # Then
    assert log_model.l_rate == learning_rate
    assert log_model.n_iters == n_iters
    assert np.array_equal(expected_pred_array, y_pred)
