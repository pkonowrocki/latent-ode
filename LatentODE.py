from abc import ABC
from typing import Iterator, Tuple

import torch.nn as nn
from ModuleODE import ExampleModuleODE
from RnnCell import RnnCell
from torch import Tensor, device, randn, ones
from torch.nn import Parameter
from torchdiffeq import odeint


class LatentODE(nn.Module, ABC):

    def __init__(self,
                 latent_size: int = 4,
                 obs_size: int = 14,
                 hidden_size: int = 25,
                 output_size: int = 12,
                 device: device = device("cpu"),
                 ode_fun: nn.Module = None,
                 rnn_cell: nn.Module = None,
                 decoder: nn.Module = None,
                 match: bool = False):
        super(LatentODE, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.device = device
        self.match = match

        if rnn_cell:
            self.rnn_cell = rnn_cell
        else:
            self.rnn_cell = RnnCell(
                latent_size,
                obs_size,
                hidden_size,
                device
            )

        if decoder:
            self.decoder = decoder
        else:
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_size, 4 * self.latent_size),
                nn.ELU(),
                nn.Linear(4*self.latent_size, 2 * self.latent_size),
                nn.ELU(),
                nn.Linear(2 * self.latent_size, self.output_size)
            )

        if ode_fun:
            self.ode_fun = ode_fun
        else:
            self.ode_fun = ExampleModuleODE(self.latent_size)

        self.decoder.to(device)
        self.rnn_cell.to(device)
        self.ode_fun.to(device)

    def forward(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x: Tensor [batch_size x seq_len x obs_size]
            t: Tensor [seq_len]
        Returns:
            pred_x: Tensor [batch_size x seq_len x output_size]
            z0: Tensor [batch_size x latent_size]
            mean: Tensor [batch_size x latent_size]
            var: Tensor [batch_size x latent_size]
        """
        qz0_mean, qz0_var = self.rnn_cell(x)
        epsilon = randn(qz0_mean.size()).to(self.device)
        z0 = epsilon * qz0_var + qz0_mean
        pred_z = odeint(self.ode_fun, z0, t).permute(1, 0, 2)
        pred_x = self.decoder(pred_z)
        if self.match:
            dx = (x[:, x.shape[1]-1:x.shape[1], 3:4] - pred_x[:, x.shape[1]-1:x.shape[1], :])
            dx = dx * ones(pred_x.shape, device=self.device)
            pred_x = pred_x + dx
        return pred_x, z0, qz0_mean, qz0_var

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return list(self.ode_fun.parameters()) + list(self.decoder.parameters()) + list(self.rnn_cell.parameters())
