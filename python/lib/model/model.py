from abc import ABC, abstractmethod
from typing import Tuple

from torch import nn, Tensor


class GameNetwork(nn.Module, ABC):
    @abstractmethod
    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """ Returns value: `(B,)`, wdl: `(B,3)`, policy: `(C,S,S)`"""
        pass
