"""Convert problem settings."""

import copy
from test.core.derivatives.utils import get_available_devices

import torch

from backpack import extend


def make_test_problems(settings):
    problem_dicts = []

    for setting in settings:
        setting = add_missing_defaults(setting)
        devices = setting["device"]

        for dev in devices:
            problem = copy.deepcopy(setting)
            problem["device"] = dev
            problem_dicts.append(problem)

    return [ExtensionsTestProblem(**p) for p in problem_dicts]


def add_missing_defaults(setting):
    """Create extensions test problem from setting.
    Args:
        setting (dict): configuration dictionary
    Returns:
        ExtensionsTestProblem: problem with specified settings.
    """
    required = ["module_fn", "input_fn", "loss_function_fn", "target_fn"]
    optional = {
        "id_prefix": "",
        "seed": 0,
        "device": get_available_devices(),
    }

    for req in required:
        if req not in setting.keys():
            raise ValueError("Missing configuration entry for {}".format(req))

    for opt, default in optional.items():
        if opt not in setting.keys():
            setting[opt] = default

    for s in setting.keys():
        if s not in required and s not in optional.keys():
            raise ValueError("Unknown config: {}".format(s))

    return setting


class ExtensionsTestProblem:
    def __init__(
        self,
        input_fn,
        module_fn,
        loss_function_fn,
        target_fn,
        device,
        seed,
        id_prefix,
    ):
        """Collection of information required to test extensions.

        Args:
            input_fn (callable): Function returning the network input.
            module_fn (callable): Function returning the network.
            loss_function_fn (callable): Function returning the loss module.
            target_fn (callable): Function returning the labels.
            device (torch.device): Device to run on.
            seed (int): Random seed.
            id_prefix (str): Extra string added to test id.
        """
        self.module_fn = module_fn
        self.input_fn = input_fn
        self.loss_function_fn = loss_function_fn
        self.target_fn = target_fn

        self.device = device
        self.seed = seed
        self.id_prefix = id_prefix

    def set_up(self):
        torch.manual_seed(self.seed)

        self.model = self.module_fn().to(self.device)
        self.input = self.input_fn().to(self.device)
        self.target = self.target_fn().to(self.device)
        self.loss_function = self.loss_function_fn().to(self.device)

    def tear_down(self):
        del self.model, self.input, self.target, self.loss_function

    def make_id(self):
        """Needs to function without call to `set_up`."""
        prefix = (self.id_prefix + "-") if self.id_prefix != "" else ""
        return (
            prefix
            + "dev={}-in={}-model={}-loss={}".format(
                self.device,
                tuple(self.input_fn().shape),
                self.module_fn(),
                self.loss_function_fn(),
            ).replace(" ", "")
        )

    def forward_pass(self, sample_idx=None):
        """Do a forward pass. Return input, output, and loss.

        Args:
            sample_idx ([int], int or None): Indices of samples to use. ``None``
                uses all samples. Integer denotes the index of a single sample, and
                list of integers refers to a set of samples.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): First tensor is the forward
                passes' input, second tensor is the model output, i.e. the forward
                passes' evaluation of the first tensor. Third tensor is the loss.
        """
        input = self.input.clone().detach()
        target = self.target.clone().detach()

        if sample_idx is not None:
            idx = [sample_idx] if isinstance(sample_idx, int) else sample_idx
            input = input[idx]
            target = target[idx]

        output = self.model(input)
        loss = self.loss_function(output, target)

        return input, output, loss

    def extend(self):
        self.model = extend(self.model)
        self.loss_function = extend(self.loss_function)

    @staticmethod
    def get_reduction_factor(loss, unreduced_loss):
        """Return the factor used to reduce the individual losses."""
        mean_loss = unreduced_loss.flatten().mean()
        sum_loss = unreduced_loss.flatten().sum()
        if torch.allclose(mean_loss, sum_loss):
            raise RuntimeError(
                "Cannot determine reduction factor. ",
                "Results from 'mean' and 'sum' reduction are identical. ",
                f"'mean': {mean_loss}, 'sum': {sum_loss}",
            )
        if torch.allclose(loss, mean_loss):
            factor = 1.0 / unreduced_loss.numel()
        elif torch.allclose(loss, sum_loss):
            factor = 1.0
        else:
            raise RuntimeError(
                "Reductions 'mean' or 'sum' do not match with loss. ",
                f"'mean': {mean_loss}, 'sum': {sum_loss}, loss: {loss}",
            )
        return factor

    def compute_reduction_factor(self):
        """Compute and return the loss function's reduction factor in batch mode.

        For instance, if ``reduction='mean'`` is used, then the reduction factor
        is ``1 / N`` where ``N`` is the batch size. With ``reduction='sum'``, it
        is ``1``.

        Returns:
            float: Reduction factor
        """
        _, _, loss = self.forward_pass()

        batch_size = self.input.shape[0]
        loss_list = torch.zeros(batch_size, device=self.device)

        for n in range(batch_size):
            _, _, loss_n = self.forward_pass(sample_idx=n)
            loss_list[n] = loss_n

        return self.get_reduction_factor(loss, loss_list)
