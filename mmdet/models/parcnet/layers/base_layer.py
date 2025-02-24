from torch import nn, Tensor
import argparse
from typing import Tuple


class BaseLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseLayer, self).__init__()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def forward(self, *args, **kwargs) ->  Tensor or Tuple[Tensor]:
        pass

    def profile_module(self, *args, **kwargs) -> Tuple[Tensor, float, float]:
        raise NotImplementedError

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
