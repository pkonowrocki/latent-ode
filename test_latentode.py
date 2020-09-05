import os

import torch.nn as nn
import torch.optim as optim
from torch import from_numpy, device, cuda, Tensor, no_grad

from LatentODE import LatentODE
from ModuleODE import ModuleODE
from utils import load_tensor_data, RunningAverageMeter, Visualisator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



class FuncODE(ModuleODE):
    def __init__(self, input_size: int):
        super(FuncODE, self).__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(input_size, input_size * 3),
            nn.Tanh(),
            nn.Linear(input_size * 3, input_size * 2),
            nn.Tanh(),
            nn.Linear(input_size * 2, input_size)
        )

    def forward(self, t: Tensor, input: Tensor) -> Tensor:
        return self.model(input**3)


if __name__ == '__main__':
    n_iters = 2000
    seq_len = 15
    latent_size = 32
    observation_size = 14
    output_size = 1
    hidden_state_size = 64
    device = device('cuda' if cuda.is_available() else 'cpu')
    dir = '.\latentode_test_img'
    if not os.path.isdir(dir):
        os.mkdir(dir)

    print(f'Device: {device}')

    latentODE = LatentODE(
        latent_size=latent_size,
        obs_size=observation_size,
        hidden_size=hidden_state_size,
        output_size=output_size,
        ode_fun=FuncODE(latent_size),
        device=device
    )
    optimizer = optim.Adam(latentODE.parameters())
    loss_meter = RunningAverageMeter()
    val_loss_meter = RunningAverageMeter()
    metric = nn.MSELoss()
    test_T, test_X, training_T, training_X, val_T, val_X = load_tensor_data(k=500, batch_size=50,
                                                                            seq_size=seq_len*2, validation_size=0.2,
                                                                            test_size=0.1)
    vis = Visualisator(device)
    vis.visualise_alpha(latentODE, test_T, test_X, save_to_dir=dir, show_fig=False, iter=0)

    for itr in range(1, n_iters + 1):
        # training
        optimizer.zero_grad()
        loss = 0
        for batch_idx in range(training_T.shape[0]):
            x = from_numpy(training_X[batch_idx, :, :, :]).float().to(device)
            t = from_numpy(training_T[batch_idx, :]).float().to(device)
            y_hat, _, _, _ = latentODE(x[:, :seq_len, :], t)
            y = x[:, :, 3:4]
            loss += metric(y_hat, y)

        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item()/training_T.shape[0])
        loss_meter.print()

        # validation
        with no_grad():
            val_loss = 0
            for val_idx in range(val_T.shape[0]):
                x = from_numpy(val_X[val_idx, :, :, :]).float().to(device)
                t = from_numpy(val_T[val_idx, :]).float().to(device)
                y_hat, _, _, _ = latentODE(x[:, :seq_len, :], t)
                y = x[:, :, 3:4]
                val_loss += metric(y_hat, y)
            val_loss_meter.update(val_loss.item()/val_T.shape[0])
            val_loss_meter.print()

        # visual
            if itr % 2 == 0:
                vis.visualise_alpha(latentODE, test_T, test_X,
                          save_to_dir=dir, show_fig=False, iter=itr)
                loss_meter.visualise(save_to=f'{dir}/loss.png')
                val_loss_meter.visualise(save_to=f'{dir}/val_loss.png')
