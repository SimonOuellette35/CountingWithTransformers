import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable, Tuple
from torch import Tensor
import math
import numpy as np

# Code taken from: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
# and modified for the purpose of the experiemnts.
# This module implements the model used in the experiment: No-LayerNorm-Identity

class FFNoLayerNorm(nn.Module):

    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, dim_feedforward: int = 2048, dropout: float = 0.,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        out = self._ff_block(src)
        return out

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))