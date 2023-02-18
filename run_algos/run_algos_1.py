"""This module is used to run the built Naive Bayes' algorithm."""

import click
import numpy as np
from sklearn.model_selection import train_test_split

from run_algos.utils import RANDOM_STATE, TEST_SIZE, generate_mock_data, visualize
from src.naive_bayes import NaiveBayes


@click.command()
@click.option("-v", "--verbose", help="increase the verbosity", count=True, default=0)
def run_naive_bayes(verbose: int):
    """Train and evaluate the model."""
    # create a synthetic data
    X, y = generate_mock_data(type_="classification")

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
