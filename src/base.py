"""This module is contains the Abstract Base Class for building ML algotithms from scratch."""
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    """This is an abstract class for defining an ML model/algorithm."""

    @abstractmethod
    def __repr__(self) -> None:
        """This is used for printing the model signature."""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y=np.ndarray) -> None:
        """This is used for training the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> None:
        """This is used for making predictions using
        the trained model."""
        pass