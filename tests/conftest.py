"""This module is used to load fixtures used for testing."""
from unittest.mock import Mock, patch

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from run_algos import RANDOM_STATE, TEST_SIZE, generate_mock_data


@pytest.fixture
def continuous_data() -> tuple[np.ndarray]:
    """This loads the data for regression."""
    X, y = generate_mock_data(type_="regression")
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def classif_data() -> tuple[np.ndarray]:
    """This loads the data for classification."""
    X, y = generate_mock_data(type_="classification")
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def mock_linear_model():
    """This returns the mocked Linear model."""
    expected_mse, expected_pred_array = 125, np.array([20.02, 25.08, 32.00, 18.17])
    expected_mock_values = Mock(
        name="mock_linear_regression",
        **{
            "predict.return_value": expected_pred_array,
            "calculate_mean_squared_error.return_value": expected_mse,
        },
    )
    LinearRegression = Mock(return_value=expected_mock_values)
    yield LinearRegression


@pytest.fixture
def mock_log_model_predict():
    """This returns the mocked Logistic model predict method."""
    with patch(
        "src.logistic_regression.LogisticRegression.predict",
        return_value=np.array([1, 0, 1, 1, 1, 1]),
        autospec=True,
    ) as m:
        yield m
