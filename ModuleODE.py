from abc import abstractmethod
import torch.nn as nn
from torch import Tensor


class ModuleODE(nn.Module):

    def __init__(self):
        super(ModuleODE, self).__init__()

    @abstractmethod
    def forward(self, t: Tensor, input: Tensor) -> Tensor:
        pass


class ExampleModuleODE(ModuleODE):
    def __init__(self, input_size: int):
        super(ExampleModuleODE, self).__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.Tanh(),
            nn.Linear(input_size * 2, input_size)
        )

    def forward(self, t: Tensor, input: Tensor) -> Tensor:
        return self.model(input**3)
