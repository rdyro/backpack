from torch.nn import (
    ELU,
    SELU,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    LeakyReLU,
    Linear,
    LogSigmoid,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    MSELoss,
    ReLU,
    Sigmoid,
    Tanh,
    ZeroPad2d,
)

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.secondorder.hbp import LossHessianStrategy

from . import (
    activations,
    conv1d,
    conv2d,
    conv3d,
    convtranspose1d,
    convtranspose2d,
    convtranspose3d,
    dropout,
    flatten,
    linear,
    losses,
    padding,
    pooling,
)


class DiagGGN(BackpropExtension):
    """Base class for diagonal generalized Gauss-Newton/Fisher matrix."""

    VALID_LOSS_HESSIAN_STRATEGIES = [
        LossHessianStrategy.EXACT,
        LossHessianStrategy.SAMPLING,
    ]

    def __init__(self, loss_hessian_strategy, savefield, subsampling):
        if loss_hessian_strategy not in self.VALID_LOSS_HESSIAN_STRATEGIES:
            raise ValueError(
                "Unknown hessian strategy: {}".format(loss_hessian_strategy)
                + "Valid strategies: [{}]".format(self.VALID_LOSS_HESSIAN_STRATEGIES)
            )

        self.loss_hessian_strategy = loss_hessian_strategy
        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.DiagGGNMSELoss(),
                CrossEntropyLoss: losses.DiagGGNCrossEntropyLoss(),
                Linear: linear.DiagGGNLinear(),
                MaxPool1d: pooling.DiagGGNMaxPool1d(),
                MaxPool2d: pooling.DiagGGNMaxPool2d(),
                AvgPool1d: pooling.DiagGGNAvgPool1d(),
                MaxPool3d: pooling.DiagGGNMaxPool3d(),
                AvgPool2d: pooling.DiagGGNAvgPool2d(),
                AvgPool3d: pooling.DiagGGNAvgPool3d(),
                ZeroPad2d: padding.DiagGGNZeroPad2d(),
                Conv1d: conv1d.DiagGGNConv1d(),
                Conv2d: conv2d.DiagGGNConv2d(),
                Conv3d: conv3d.DiagGGNConv3d(),
                ConvTranspose1d: convtranspose1d.DiagGGNConvTranspose1d(),
                ConvTranspose2d: convtranspose2d.DiagGGNConvTranspose2d(),
                ConvTranspose3d: convtranspose3d.DiagGGNConvTranspose3d(),
                Dropout: dropout.DiagGGNDropout(),
                Flatten: flatten.DiagGGNFlatten(),
                ReLU: activations.DiagGGNReLU(),
                Sigmoid: activations.DiagGGNSigmoid(),
                Tanh: activations.DiagGGNTanh(),
                LeakyReLU: activations.DiagGGNLeakyReLU(),
                LogSigmoid: activations.DiagGGNLogSigmoid(),
                ELU: activations.DiagGGNELU(),
                SELU: activations.DiagGGNSELU(),
            },
        )
        self._subsampling = subsampling

    def get_subsampling(self):
        """Return the indices of samples contributing to the diagonal GGN/Fisher."""
        return self._subsampling


class DiagGGNExact(DiagGGN):
    """
    Diagonal of the mini-batch (or a subset) Generalized Gauss-Newton/Fisher.
    Uses the exact Hessian of the loss w.r.t. the model output.

    Stores the output in :code:`diag_ggn_exact`,
    has the same dimensions as the gradient.

    For a faster but less precise alternative,
    see :py:meth:`backpack.extensions.DiagGGNMC`.

    Note: Using ``subsampling``
        Let the loss be given by a sum ``∑ᵢ₌₁ⁿ fᵢ``, or mean
        ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``. The diagonal GGN/Fisher is evaluated
        for every ``fᵢ`` or ``¹/ₙ fᵢ``, then summed. With
        ``subsampling``, the summation runs only over the
        specified indices, rather then the full mini-batch.

    Args:
        subsampling ([int], optional): Indices of samples in
            the mini-batch for which the GGN/Fisher diagonal
            should be computed and summed. Default value ``None``
            uses the entire mini-batch.
    """

    def __init__(self, subsampling=None):
        super().__init__(LossHessianStrategy.EXACT, "diag_ggn_exact", subsampling)


class DiagGGNMC(DiagGGN):
    """
    Diagonal of the Generalized Gauss-Newton/Fisher.
    Uses a Monte-Carlo approximation of
    the Hessian of the loss w.r.t. the model output.

    Stores the output in :code:`diag_ggn_mc`,
    has the same dimensions as the gradient.

    For a more precise but slower alternative,
    see :py:meth:`backpack.extensions.DiagGGNExact`.

    Note: Using ``subsampling``
        Let the loss be given by a sum ``∑ᵢ₌₁ⁿ fᵢ``, or mean
        ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``. The diagonal GGN/Fisher is evaluated
        for every ``fᵢ`` or ``¹/ₙ fᵢ``, then summed. With
        ``subsampling``, the summation runs only over the
        specified indices, rather then the full mini-batch.

    Args:
        subsampling ([int], optional): Indices of samples in
            the mini-batch for which the MC-approximated GGN/Fisher
            diagonal should be computed and summed. Default value
            ``None`` uses the entire mini-batch.
    """

    def __init__(self, mc_samples=1, subsampling=None):
        self._mc_samples = mc_samples
        super().__init__(LossHessianStrategy.SAMPLING, "diag_ggn_mc", subsampling)

    def get_num_mc_samples(self):
        return self._mc_samples


class BatchDiagGGN(BackpropExtension):
    """Base class for batched diagonal generalized Gauss-Newton/Fisher matrix."""

    VALID_LOSS_HESSIAN_STRATEGIES = [
        LossHessianStrategy.EXACT,
        LossHessianStrategy.SAMPLING,
    ]

    def __init__(self, loss_hessian_strategy, savefield, subsampling):
        if loss_hessian_strategy not in self.VALID_LOSS_HESSIAN_STRATEGIES:
            raise ValueError(
                "Unknown hessian strategy: {}".format(loss_hessian_strategy)
                + "Valid strategies: [{}]".format(self.VALID_LOSS_HESSIAN_STRATEGIES)
            )

        self.loss_hessian_strategy = loss_hessian_strategy
        super().__init__(
            savefield=savefield,
            module_exts={
                MSELoss: losses.DiagGGNMSELoss(),
                CrossEntropyLoss: losses.DiagGGNCrossEntropyLoss(),
                Linear: linear.BatchDiagGGNLinear(),
                MaxPool1d: pooling.DiagGGNMaxPool1d(),
                MaxPool2d: pooling.DiagGGNMaxPool2d(),
                AvgPool1d: pooling.DiagGGNAvgPool1d(),
                MaxPool3d: pooling.DiagGGNMaxPool3d(),
                AvgPool2d: pooling.DiagGGNAvgPool2d(),
                AvgPool3d: pooling.DiagGGNAvgPool3d(),
                ZeroPad2d: padding.DiagGGNZeroPad2d(),
                Conv1d: conv1d.BatchDiagGGNConv1d(),
                Conv2d: conv2d.BatchDiagGGNConv2d(),
                Conv3d: conv3d.BatchDiagGGNConv3d(),
                ConvTranspose1d: convtranspose1d.BatchDiagGGNConvTranspose1d(),
                ConvTranspose2d: convtranspose2d.BatchDiagGGNConvTranspose2d(),
                ConvTranspose3d: convtranspose3d.BatchDiagGGNConvTranspose3d(),
                Dropout: dropout.DiagGGNDropout(),
                Flatten: flatten.DiagGGNFlatten(),
                ReLU: activations.DiagGGNReLU(),
                Sigmoid: activations.DiagGGNSigmoid(),
                Tanh: activations.DiagGGNTanh(),
                LeakyReLU: activations.DiagGGNLeakyReLU(),
                LogSigmoid: activations.DiagGGNLogSigmoid(),
                ELU: activations.DiagGGNELU(),
                SELU: activations.DiagGGNSELU(),
            },
        )
        self._subsampling = subsampling

    def get_subsampling(self):
        """Return the indices of samples whose GGN/Fisher diagonals are evaluated."""
        return self._subsampling


class BatchDiagGGNExact(BatchDiagGGN):
    """
    Individual diagonal of the Generalized Gauss-Newton/Fisher.
    Uses the exact Hessian of the loss w.r.t. the model output.

    Stores the output in ``diag_ggn_exact_batch`` as a ``[N x ...]`` tensor,
    where ``N`` is the mini-batch (or subset) size and ``...`` is the shape
    of the gradient.

    Args:
        subsampling ([int], optional): Indices of samples in
            the mini-batch for which the individual GGN/Fisher diagonal
            should be computed. Default value ``None`` uses the entire
            mini-batch.
    """

    def __init__(self, subsampling=None):
        super().__init__(
            loss_hessian_strategy=LossHessianStrategy.EXACT,
            savefield="diag_ggn_exact_batch",
            subsampling=subsampling,
        )
