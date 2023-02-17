"""This module is used to run the built Naive Bayes' algorithm."""
from typing import NewType

import click
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.naive_bayes import NaiveBayes

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


@click.command()
@click.option("-v", "--verbose", help="increase the verbosity", count=True, default=0)
def run_naive_bayes(verbose: int):
    """Train and evaluate the model."""
    # create a synthetic data
    X, y = generate_mock_data(type_="regression")

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # instantiate model
    nb = NaiveBayes()
    if 1 <= verbose < 2:
        click.echo(nb)
        click.echo(f"Training data shape: {X_train.shape}\n")

    if 2 <= verbose < 4:
        # Visualize data
        click.echo("Visualizing data ...\n")
        visualize(X=X, y=y)

    # train model
    click.echo("=== Training the model ===")
    nb.fit(X_train, y_train)

    # make predictions
    click.echo("=== Making predictions with the model ===")
    y_pred = nb.predict(X_test)

    click.echo("\n=== Evaluating the model performance ===")
    accuracy = np.mean(y_pred == y_test)
    click.echo(f"Accuracy: {accuracy}\n")
    return accuracy


if __name__ == "__main__":
    run_naive_bayes()
