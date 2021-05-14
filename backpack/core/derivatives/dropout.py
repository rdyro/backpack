from torch import eq

from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.core.derivatives.subsampling import subsample_output


class DropoutDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self):
        return True

    def df(self, module, g_inp, g_out, subsampling=None):
        scaling = 1 / (1 - module.p)
        output = subsample_output(module, subsampling=subsampling)
        mask = 1 - eq(output, 0.0).float()
        return mask * scaling
