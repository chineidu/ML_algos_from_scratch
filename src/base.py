"""This module is contains the Abstract Base Class for building ML algotithms from scratch."""
from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np


# pylint: disable=unnecessary-pass
class Model(ABC):
    """This is an abstract class for defining an ML model/algorithm."""

    @abstractmethod
    def __repr__(self) -> str:
        """This is used for printing the model signature."""
        pass

    @abstractmethod
    def fit(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> None:
        """This is used for training the model."""
        return self

    @abstractmethod
    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """This is used for making predictions using
        the trained model."""
        pass
