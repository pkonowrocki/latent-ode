import os

import torch.nn as nn
import torch.optim as optim
from torch import from_numpy, device, cuda, Tensor, no_grad

from LatentODE import LatentODE
from ModuleODE import ModuleODE
from utils import RunningAverageMeter, Visualisator, load_tensor_data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FuncODE(ModuleODE):
    def __init__(self, input_size: int):
        super(FuncODE, self).__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, input_size)
        )

    def forward(self, t: Tensor, input: Tensor) -> Tensor:
        return self.model(input ** 3)


if __name__ == '__main__':
    n_iters = 2000
    seq_len = 15
    latent_size = 128
    observation_size = 14
    output_size = 1
    hidden_state_size = 64
    device = device('cuda' if cuda.is_available() else 'cpu')
    dir = './latentode_test_img'
    training_dir = './latentode_training_img'
    if not os.path.isdir(dir):
        os.mkdir(dir)

    if not os.path.isdir(training_dir):
        os.mkdir(training_dir)

    print(f'Device: {device}')

    # init model
    latentODE = LatentODE(
        latent_size=latent_size,
        obs_size=observation_size,
        hidden_size=hidden_state_size,
        output_size=output_size,
        ode_fun=FuncODE(latent_size),
        device=device,
        decoder=nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, output_size)
        )
    )

    optimizer = optim.Adam(latentODE.parameters(), lr=1.5e-3)
    metric = nn.MSELoss()

    # load data
    test_T, test_X, training_T, training_X, val_T, val_X = load_tensor_data(data_folder='./data', k=500, batch_size=16,
                                                                            seq_size=seq_len * 2, validation_size=0.2,
                                                                            test_size=0.1)

    # loss loggers
    loss_meter = RunningAverageMeter(use_log_scale=True)
    val_loss_meter = RunningAverageMeter(use_log_scale=True)

    # init printers
    vis = Visualisator(device)
    vis.visualise_alpha(latentODE, test_T, test_X, save_to_dir=dir, show_fig=False, iter=0)

    train_vis = Visualisator(device, samples_num=1)
    train_vis.visualise_alpha(latentODE, [training_T[0, :]], [training_X[0, 0:1, :, :]], save_to_dir=training_dir,
                              show_fig=False, iter=0)

    # prepare data
    training_X = from_numpy(training_X).float().to(device)
    training_T = from_numpy(training_T).float().to(device)
    val_X = from_numpy(val_X).float().to(device)
    val_T = from_numpy(val_T).float().to(device)

    for itr in range(1, n_iters + 1):
        # training
        optimizer.zero_grad()
        loss = 0
        for batch_idx in range(training_T.shape[0]):
            y_hat, _, _, _ = latentODE(training_X[batch_idx, :, :seq_len, :], training_T[batch_idx, :])
            y = training_X[batch_idx, :, :, 3:4]
            loss += metric(y_hat, y)

        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item() / training_T.shape[0])
        loss_meter.print()

        # validation
        with no_grad():
            # cals validation loss
            if itr % 10 == 0:
                val_loss = 0
                for val_idx in range(val_T.shape[0]):
                    y_hat, _, _, _ = latentODE(val_X[val_idx, :, :seq_len, :], val_T[val_idx, :])
                    y = val_X[val_idx, :, :, 3:4]
                    val_loss += metric(y_hat, y)
                val_loss_meter.update(val_loss.item() / val_T.shape[0])
                val_loss_meter.print()

            # visualise training loss
            if itr % 10 == 0:
                loss_meter.visualise(show_fig=False)

            # visualise validation loss, training example, test example
            if itr % 20 == 0:
                val_loss_meter.visualise(show_fig=False)
                vis.visualise_alpha(latentODE, test_T, test_X,
                                    save_to_dir=dir, show_fig=False, iter=itr, metric=metric)
