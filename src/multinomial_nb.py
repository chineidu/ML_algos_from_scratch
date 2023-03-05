"""This module is used to build Multi-nomial Naive Bayes
algorithm from scratch."""

from typing import Any, Union

import numpy as np
import pandas as pd

from base import Model


# pylint: disable=too-many-locals
class MultiNomial_NB(Model):
    """This classifier uses MArkov model for classifiication.\n
    It's trained using two models. i.e for 2 classes (labels).

    Params:
        vocab (dict): A dictionary containing the vocabulary.

    Returns:
        None
    """

    def __init__(self, *, vocab: dict) -> None:
        self.vocab = vocab
        self.transition_matrix = None
        self.initial_state_distr = None
        self.log_priors = [None, None]
        self.K = [None, None]
        self.priors = [None, None]

    def __repr__(self) -> str:
        priors_dict = {
            self.K[0]: round(self.priors[0], 2),
            self.K[1]: round(self.priors[1], 2),
        }
        return (
            f"{__class__.__name__}(log_priors={self.log_priors[0], self.log_priors[1]}, "
            f"priors={priors_dict})"
        )

    def fitt(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> None:
        """This is used for training the model."""
        # Since we have 2 class labels, we need to initialize and
        # create 2 models. Compute count and log probs for each model.
        # Retrieve the probabilities
        A_0, Pi_0 = self._init_A_and_Pi()
        A_1, Pi_1 = self._init_A_and_Pi()
        k_0, k_1 = np.unique(y)
        log_y0, log_y1 = self._log_priors(y=y)

        # Model 0
        input_0 = self._get_input(tokenized_doc=X, y=y, k_=k_0)
        A_hat_0, Pi_hat_0 = self._count_state_transitions(X=input_0, A_hat=A_0, Pi_hat=Pi_0)
        log_A_hat_0, log_Pi_hat_0 = self._convert_counts_to_log_prob(
            X=input_0, A_hat=A_hat_0, Pi_hat=Pi_hat_0
        )

        # Model 1
        input_1 = self._get_input(tokenized_doc=X, y=y, k_=k_1)
        A_hat_1, Pi_hat_1 = self._count_state_transitions(X=input_1, A_hat=A_1, Pi_hat=Pi_1)
        log_A_hat_1, log_Pi_hat_1 = self._convert_counts_to_log_prob(
            X=input_1, A_hat=A_hat_1, Pi_hat=Pi_hat_1
        )

        self.transition_matrix = (log_A_hat_0, log_A_hat_1)
        self.initial_state_distr = (log_Pi_hat_0, log_Pi_hat_1)
        self.log_priors = (log_y0, log_y1)
        self.K = k_0, k_1

        return self

    def _init_A_and_Pi(self) -> tuple:
        """This is used to initialize the state transition matrix
        and the initial state distribution."""
        V = len(self.vocab)
        # Add add-one smoothering
        # A is a matrix and Pi is a vector
        A = np.ones(shape=(V, V), dtype=float)
        Pi = np.ones(shape=(V), dtype=float)
        return (A, Pi)

    @staticmethod
    def _get_input(*, tokenized_doc: list[int], y: np.ndarray, k_: int) -> list[int]:
        """This returns an input given a specific class label.
        i.e the input given a specific class label.

        Params:
            tokenized_doc (list[int]): The tokenized documents (corpus).
            y (np.ndarray): The labels for the data.
            k_ (int): The class label.

        Returns:
            tokenized_data: The tokenized documents belonging to the
                specified class label.
        """
        tokenized_data = [txt for txt, label in zip(tokenized_doc, y) if label == k_]
        return tokenized_data

    def _log_priors(self, *, y: np.ndarray) -> list[float]:
        """This returns the log priors of y.

        Params:
            y (np.ndarray): The labels for the data.

        Returns:
            log_probs: A list containing the log probabilities
            of the class labels.
        """
        # Get the counts; calculate the log probabilities
        # using the probabilities obtained from the counts.
        counts = np.bincount(y)
        probs = counts / len(y)
        self.priors = [p_i for p_i in probs if p_i > 0]
        log_probs = [(np.log(p_i)) for p_i in self.priors]
        return log_probs

    @staticmethod
    def _count_state_transitions(
        *, X: list[int], A_hat: np.ndarray, Pi_hat: np.ndarray
    ) -> tuple[np.ndarray]:
        """This is used to count the occurrences of transitions.
        i.e calculate the counts of A_hat and Pi_hat.

        Returns:
            (A_hat, Pi_hat)
        """

        # To calculate the Pi_hat, we need to count the number of times
        # the initial state was `i` divided by the number of state sequences.
        # Pi_hat = (count(state = i) / N)
        # A_hat: count of the number of times we transitioned from the prev state `i`
        # to the current state `j` divided by the count of the prev state `i`.
        # i.e. A_hat = ( count(state_i to state_j) / (count(state_i)) )
        # Note: Update the prev_state after each transition.
        for tokenized_doc in X:
            prev_token = None
            for curr_token in tokenized_doc:
                if prev_token is None:
                    Pi_hat[curr_token] += 1
                else:
                    A_hat[prev_token, curr_token] += 1
                # Update the prev_token
                prev_token = curr_token
        return (A_hat, Pi_hat)

    @staticmethod
    def _convert_counts_to_log_prob(
        *,
        X: Union[np.ndarray, pd.Series],
        A_hat: np.ndarray,
        Pi_hat: np.ndarray,
    ) -> tuple[np.ndarray]:
        """This is used to calculate the log of the class conditional
        probability given a specific class label. It returns a tuple
        of arrays containing the log probabilities.

        Returns:
            (log(A_hat), log(Pi_hat))
        """
        # Calculate the probabilities
        A_hat /= A_hat.sum(axis=1, keepdims=True)  # OR A_hat/ A_hat.shape[0]
        Pi_hat /= Pi_hat.sum(axis=0)
        return (np.log(A_hat), np.log(Pi_hat))

    def _calculate_log_likelihoods(self, *, x: list[int], k_: int) -> tuple[np.ndarray]:
        """This is used to extract the log probability given the log
        probabililty and class label of the tokenized document.

        Returns:
            (log(A_hat), log(Pi_hat))
        """
        log_A_hat, log_Pi_hat = self.transition_matrix[k_], self.initial_state_distr[k_]

        # Calculate the probability:
        # if it's an initial state (prev_idx is None), retrieve the probability
        # otherwise, transition to a new state and retrieve the probability using
        # the prev_idx and the curr_idx.
        prev_idx, log_prob = None, 0

        for curr_idx in x:
            if prev_idx is None:
                log_prob += log_Pi_hat[curr_idx]
            else:
                log_prob += log_A_hat[prev_idx, curr_idx]

            # Update the value (for the next iteration)
            prev_idx = curr_idx
        return log_prob

    def predict(self, X: list[int]) -> None:
        """This is used for making predictions using
        the trained model."""
        # Instantiate
        predictions = np.zeros(shape=(len(X)))

        # For each sentence/tokenized_doc, make a prediction of the class label.
        # This is done by calculating the argmax of the posteriors over all classes.
        # posterior = likelihood + log_prior
        # i.e compute the prob that an input/sentence belongs to a specific class label.
        # The argmax of the posterior returns the index (class label) with the highest probability
        # i.e if an input has a prob of [0.05, 0.001], the argmax returns 0 (index 0) which means
        # that the input belongs to class 0 since 0.05 > 0.001 and it has an index of 0.
        for idx, sentence in enumerate(X):
            # Posteriors = posterior_k_0 and posterior_k_1
            posteriors = [
                (self._calculate_log_likelihoods(x=sentence, k_=k_) + self.log_priors[k_])
                for k_ in self.K
            ]
            pred = np.argmax(posteriors)
            predictions[idx] = pred
        return predictions
