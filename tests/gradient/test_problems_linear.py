import torch
from bpexts.utils import set_seeds
import bpexts.gradient.config as config
from .test_problem import TestProblem

TEST_SETTINGS = {
    "in_features": 20,
    "out_features": 10,
    "bias": True,
    "batch": 13,
    "rtol": 1e-5,
    "atol": 1e-5
}

set_seeds(0)
linearlayer = config.extend(torch.nn.Linear(
    in_features=TEST_SETTINGS["in_features"],
    out_features=TEST_SETTINGS["out_features"],
    bias=TEST_SETTINGS["bias"],
))
summationLinearLayer = config.extend(torch.nn.Linear(
    in_features=TEST_SETTINGS["out_features"],
    out_features=1,
    bias=True
))
input_size = (TEST_SETTINGS["batch"], TEST_SETTINGS["in_features"])
X = torch.randn(size=input_size)


def make_regression_problem():
    model = torch.nn.Sequential(
        linearlayer,
        summationLinearLayer
    )

    Y = torch.randn(size=(model(X).shape[0], 1))

    lossfunc = config.extend(torch.nn.MSELoss())

    return TestProblem(X, Y, model, lossfunc)


def make_classification_problem():
    class To2D(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return input.view(input.shape[0], -1)

    model = torch.nn.Sequential(
        linearlayer,
        To2D()
    )

    Y = torch.randint(high=model(X).shape[1], size=(X.shape[0],))

    lossfunc = config.extend(torch.nn.CrossEntropyLoss())

    return TestProblem(X, Y, model, lossfunc)


TEST_PROBLEMS = {
    "linear-regression": make_regression_problem(),
    "linear-classification": make_classification_problem(),
}
