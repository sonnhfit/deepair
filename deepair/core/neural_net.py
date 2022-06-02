from typing import List, Type

from torch import nn


def make_neural_net(
    input_shape: int,
    output_shape: int,
    hidden: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU
) -> List[nn.Module]:

    if len(hidden) > 0:
        modules = [nn.Linear(input_shape, hidden[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(hidden) - 1):
        modules.append(nn.Linear(hidden[idx], hidden[idx + 1]))
        modules.append(activation_fn())

    if output_shape > 0:
        last_layer = hidden[-1] if len(hidden) > 0 else input_shape
        modules.append(nn.Linear(last_layer, output_shape))

    return modules
