from backpack.core.derivatives.subsampling import subsample_input
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule
from backpack.utils import conv as convUtils


class DiagGGNConvND(DiagGGNBaseModule):
    def __init__(self, derivatives, N, params=None):
        super().__init__(derivatives=derivatives, params=params)
        self.N = N

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        sqrt_ggn = backproped

        return convUtils.extract_bias_diagonal(module, sqrt_ggn, self.N, sum_batch=True)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        subsampling = ext.get_subsampling()
        input = subsample_input(module, subsampling=subsampling)

        if self.N == 2:
            X = convUtils.unfold_func(module)(input)
        else:
            X = convUtils.unfold_by_conv(input, module)

        weight_diag = convUtils.extract_weight_diagonal(
            module, X, backproped, self.N, sum_batch=True
        )
        return weight_diag


class BatchDiagGGNConvND(DiagGGNBaseModule):
    def __init__(self, derivatives, N, params=None):
        super().__init__(derivatives=derivatives, params=params)
        self.N = N

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        sqrt_ggn = backproped
        return convUtils.extract_bias_diagonal(
            module, sqrt_ggn, self.N, sum_batch=False
        )

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        subsampling = ext.get_subsampling()
        input = subsample_input(module, subsampling=subsampling)

        if self.N == 2:
            X = convUtils.unfold_func(module)(input)
        else:
            X = convUtils.unfold_by_conv(input, module)
        weight_diag = convUtils.extract_weight_diagonal(
            module, X, backproped, self.N, sum_batch=False
        )
        return weight_diag
