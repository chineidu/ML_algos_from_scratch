"""This module is contains the Abstract Base Class for building ML algotithms from scratch."""
from abc import ABC, abstractmethod

import numpy as np


# pylint: disable=unnecessary-pass
class Model(ABC):
    """This is an abstract class for defining an ML model/algorithm."""

    @abstractmethod
    def __repr__(self) -> str:
        """This is used for printing the model signature."""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """This is used for training the model."""
        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """This is used for making predictions using
        the trained model."""
        pass
