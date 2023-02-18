"""This module contains the utility functions."""
from typing import NewType

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_regression

# Config
RANDOM_STATE = 123
TEST_SIZE = 0.1
N_SAMPLES = 2_000
N_FEATURES = 1
N_CLASSES = 2
NOISE = 10
plt.style.use("ggplot")


def generate_mock_data(*, type_: str) -> tuple[np.ndarray, np.ndarray]:
    """This generates the synthetic data required for classification
    or regression.

    Params:
        type_ (str): 'classification' or 'regression'

    Returns:
        (X, y) (tuple): It returns the predictor and the target variable.
    """
    type_value = ["classification", "regression"]
    if type_ not in type_value:
        raise ValueError(f"{type_!r} should be {type_value[0]!r} or {type_value[1]!r}.")

    regres_data = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        noise=NOISE,
        random_state=RANDOM_STATE,
    )
    classif_data = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES + 10,
        n_classes=N_CLASSES,
        random_state=RANDOM_STATE,
    )

    data = regres_data if type == "regression" else classif_data
    X, y = data  # pylint: disable=unbalanced-tuple-unpacking
    return (X, y)


# Create a new type
Plot = NewType("Plot", str)


def visualize(*, X: np.ndarray, y: np.ndarray) -> Plot:
    """Visualize the relationship between the predictor and the target."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, s=20)
    plt.show()
