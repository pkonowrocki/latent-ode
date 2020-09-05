from abc import ABC
from typing import Tuple, Iterator, Optional, Union

from torch import Tensor, zeros, device, cat, tanh, randn, exp
from torch.nn import Module, Sequential, Linear, Tanh, Parameter


class RnnCell(Module, ABC):

    def __init__(self,
                 latent_size: int = 4,
                 obs_size: int = 14,
                 hidden_size: int = 25,
                 device: device = device('cpu')):
        """
        Args:
            latent_size: int   z_0 size
            obs_size: int   x size
            hidden_size: int   h size
        """
        super(RnnCell, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.device = device

        self.i2h = Sequential(
            Linear(obs_size + hidden_size, hidden_size),
            Tanh()
        ).to(device)
        self.h2o = Linear(hidden_size, latent_size * 2).to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor [batch_size x seq_len x obs_size]

        Returns:
            z0: Tensor [batch_size x latent_size]
            mean: Tensor [batch_size x latent_size]
            var: Tensor [batch_size x latent_size]
        """
        h = zeros(x.shape[0], self.hidden_size, device=self.device)
        for t in reversed(range(x.shape[1])):
            obs = x[:, t, :]
            out, h = self.forward_rnn(obs, h)
        qz0_mean, qz0_logvar = out[:, :self.latent_size], out[:, self.latent_size:]

        return qz0_mean, qz0_logvar

    # TODO: change to native pytorch implementation
    def forward_rnn(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        combined = cat((x, h), dim=1)
        h = self.i2h(combined)
        out = self.h2o(h)
        return out, h

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return list(self.h2o.parameters()) + list(self.i2h.parameters())

