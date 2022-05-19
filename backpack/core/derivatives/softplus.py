"""Partial derivatives for the Softplus activation function."""
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Softplus

from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.utils.subsampling import subsample


class SoftplusDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self, module):
        return False

    def df(
        self,
        module: Softplus,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:
        output = subsample(module.output, subsampling=subsampling)
        temp = torch.exp(module.beta * output)
        one = torch.ones((), device=temp.device, dtype=temp.dtype)
        val = temp / (temp + 1)
        return torch.where(module.beta * output > module.threshold, one, val)

    def d2f(self, module, g_inp, g_out):
        temp = torch.exp(module.beta * output)
        zero = torch.zeros((), device=temp.device, dtype=temp.dtype)
        val = module.beta * val / (val + 1) ** 2
        return torch.where(module.beta * output > module.threshold, zero, val)
