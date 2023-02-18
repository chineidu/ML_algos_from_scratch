"""This module is used to run the built Linear and Logistic Regression algorithms."""

import click
import numpy as np
from sklearn.model_selection import train_test_split

from run_algos.utils import RANDOM_STATE, TEST_SIZE, generate_mock_data, visualize
from src.linear_regression import LinearRegression
from src.logistic_regression import LogisticRegression


@click.command()
@click.option("-lr", "--learning-rate", default=0.001)
@click.option("-ni", "--n-iters", default=1_000)
@click.option("-v", "--verbose", help="increase the verbosity", count=True, default=0)
def run_linear_reg(learning_rate: float, n_iters: int, verbose: int):
    """Train and evaluate the model."""
    # create a synthetic data
    X, y = generate_mock_data(type_="regression")

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # instantiate model
    lin_reg = LinearRegression(learning_rate=learning_rate, n_iters=n_iters)
    if 1 <= verbose < 2:
        click.echo(lin_reg)
        click.echo(f"Training data shape: {X_train.shape}\n")

    if 2 <= verbose < 4:
        # Visualize data
        click.echo("Visualizing data ...\n")
        visualize(X=X, y=y)

    # train model
    click.echo("=== Training the model ===")
    lin_reg.fit(X_train, y_train)

    # make predictions
    click.echo("=== Making predictions with the model ===")
    y_pred = lin_reg.predict(X_test)

    click.echo("\n=== Evaluating the model performance ===")
    mse = lin_reg.calculate_mean_squared_error(y_test, y_pred)
    click.echo(f"MSE: {mse}\n")
    return mse


@click.command()
@click.option("-lr", "--learning-rate", default=0.001)
@click.option("-ni", "--n-iters", default=1_000)
@click.option("-v", "--verbose", help="increase the verbosity", count=True, default=0)
def run_logistic_reg(learning_rate: float, n_iters: int, verbose: int):
    """Train and evaluate the model."""
    # create a synthetic data
    X, y = generate_mock_data(type_="classification")

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # instantiate model
    log_model = LogisticRegression(learning_rate=learning_rate, n_iters=n_iters)
    if 1 <= verbose < 2:
        click.echo(log_model)
        click.echo(f"Training data shape: {X_train.shape}\n")

    if 2 <= verbose < 4:
        # Visualize data
        click.echo("Visualizing data ...\n")
        visualize(X=X, y=y)

    # train model
    click.echo("=== Training the model ===")
    log_model.fit(X_train, y_train)

    # make predictions
    click.echo("=== Making predictions with the model ===")
    y_pred = log_model.predict(X_test)

    click.echo("\n=== Evaluating the model performance ===")
    accuracy = np.mean(y_pred == y_test)
    click.echo(f"Accuracy: {accuracy}\n")
    return accuracy


if __name__ == "__main__":
    # mse = run_linear_reg()
    run_logistic_reg()
