"""This module is contains the Abstract Base Class for building ML algotithms from scratch."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import numpy.typing as npt


# pylint: disable=unnecessary-pass
class Model(ABC):
    """This is an abstract class for defining an ML model/algorithm."""

    @abstractmethod
    def __repr__(self) -> str:
        """This is used for printing the model signature."""
        pass

    @abstractmethod
    def fit(
        self, X: npt.NDArray[Union[np.int_, np.float_]], y: npt.NDArray[Union[np.int_, np.float_]]
    ) -> None:
        """This is used for training the model."""
        pass

    @abstractmethod
    def predict(
        self, X: npt.NDArray[Union[np.int_, np.float_]]
    ) -> npt.NDArray[Union[np.int_, np.float_]]:
        """This is used for making predictions using
        the trained model."""
        pass
