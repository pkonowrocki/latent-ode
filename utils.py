from typing import Tuple, List, Union
from numpy import ndarray
import matplotlib.pyplot as plt
from numpy import random, log10
from torch import device, from_numpy, no_grad


class RunningAverageMeter(object):
    def __init__(self, momentum=0.5, use_log_scale=False):
        self.momentum = momentum
        self.reset()
        self.loss_history = []
        self.avg_history = []
        self.use_log_scale = use_log_scale

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        self.loss_history.append(val)
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        self.avg_history.append(self.avg)

    def visualise(self, show_fig=False, save_to=None):
        plt.figure()
        if self.use_log_scale:
            plt.yscale('log')

        plt.plot(self.loss_history, 'b', label='loss')
        plt.plot(self.avg_history, 'r', label='avg loss')
        plt.legend()
        if show_fig:
            plt.show()
        if save_to is not None:
            plt.savefig(save_to, dpi=500)
        plt.close()

    def print(self):
        print(
            f'Iter: {len(self.loss_history)}, current loss: {self.loss_history[-1]} running avg loss: {self.avg_history[-1]}'
            + (f' running change {(self.avg_history[-2] - self.avg_history[-1]) / self.avg_history[-2]}' if len(
                self.avg_history) >= 2 else ''))


class Visualisator():
    def __init__(self, device: device = device('cpu'), samples_num: int = 4, sample_size: int = 10):
        self.starts = []
        self.samples_num = samples_num
        self.sample_size = sample_size
        self.device = device
        self.col_names = ['x', 'y', 'z', 'alpha', 'beta', 'gamma',
                          'v_x', 'v_y', 'v_z', 'w_alpha', 'w_beta', 'w_gamma',
                          'input_1', 'input_2']
        self.idxs = []

    def visualise_alpha(self, model, T, X, metric=None, show_fig=False, save_to_dir=None, iter=''):
        # lazy init
        idx = 0
        while len(self.idxs) < self.samples_num:
            while X[idx].shape[1] - 2 * self.sample_size <= 0:
                idx += 1
            t = T[idx]
            x = X[idx]
            self.starts.append(random.randint(low=0, high=x.shape[1] - 2 * self.sample_size))
            self.idxs.append(idx)
            idx += 1

        # calc
        with no_grad():
            for i in range(len(self.idxs)):
                idx = self.idxs[i]
                start = self.starts[i]
                t = T[idx]
                x = X[idx]
                x_p = x[:, start:start + self.sample_size, :]
                t_p = t[start:start + 2 * self.sample_size]
                x_p = from_numpy(x_p).float().to(self.device)
                t_p = from_numpy(t_p).float().to(self.device)
                y_p, _, _, _ = model(x_p, t_p)
                loss = None

                if metric is not None:
                    loss = metric(y_p[0, :self.sample_size, 0], x_p[0, :, 0])
                x_p = x_p.cpu().numpy()
                y_p = y_p.cpu().numpy()
                t_p = t_p.cpu().numpy()

                generated_part = y_p[0, self.sample_size:, 0]
                sample_part = y_p[0, :self.sample_size, 0]

                plt.figure()
                if loss is not None:
                    plt.title(f'Loss: {loss.item()}')
                plt.plot(t, x[0, :, 3], 'y', label='true trajectory')
                plt.plot(t_p[:self.sample_size], x_p[0, :, 3], 'b', label='sample')
                plt.plot(t_p, y_p[0, :, 0], 'r', label='generated-ode')
                plt.plot(t_p[:self.sample_size], sample_part, 'g', label='sample-ode')

                plt.legend()
                if show_fig:
                    plt.show()
                if save_to_dir is not None:
                    plt.savefig(f'{save_to_dir}/{iter}_{i+1}.png', dpi=500)
                plt.close()

    def visualise_all(self, model, T, X, idx=0, show_fig=False, save_to=None):
        col_names = ['x', 'y', 'z', 'alpha', 'beta', 'gamma',
                     'v_x', 'v_y', 'v_z', 'w_alpha', 'w_beta', 'w_gamma']

        t = T[idx]
        x = X[idx]

        # n: 10
        n = 10
        if self.starts is None:
            self.starts = random.randint(low=0, high=x.shape[1] - 2 * n)
        x_10 = x[:, self.starts:self.starts + n, :]
        t_10 = t[self.starts:self.starts + 2 * n]
        x_10 = from_numpy(x_10).float().to(self.device)
        t_10 = from_numpy(t_10).float().to(self.device)

        y_10, _, _, _ = model(x_10, t_10)

        x_10 = x_10.cpu().numpy()
        y_10 = y_10.cpu().detach().numpy()
        t_10 = t_10.cpu().numpy()

        figure, axes = plt.subplots(nrows=3, ncols=4)
        for i in range(12):
            axes[i % 3, i // 3].set_title(col_names[i])
            axes[i % 3, i // 3].plot(t, x[0, :, i], 'y', label='true trajectory')
            axes[i % 3, i // 3].plot(t_10[:n], x_10[0, :, i], 'b', label='sample')
            axes[i % 3, i // 3].plot(t_10[n:], y_10[0, n:, i], 'r', label='generated-ode')
            axes[i % 3, i // 3].plot(t_10[:n], y_10[0, :n, i], 'g', label='sample-ode')

        plt.legend()
        if show_fig:
            plt.show()
        if save_to is not None:
            plt.savefig(save_to, dpi=500)
        plt.close()



def load_data(data_folder: str = None) -> Tuple[List[ndarray], List[ndarray]]:
    from os import listdir
    from numpy import loadtxt, array, expand_dims, stack
    if data_folder is None:
        data_folder = "./data"
    files = listdir(data_folder)
    X = []
    T = []

    for file in files:
        try:
            temp = loadtxt(f'{data_folder}/{file}', delimiter=',')
        except ValueError:
            print(file)
        x = expand_dims(temp[:, 1:15], 0)
        t = temp[:, 0]
        X = X + [x]
        T = T + [t]

    return T, X


def load_split_data(data_folder: str = None, test_size: int = 0.3, max_seq_len: Union[None, int] = None,
                    batch_size: int = 100, training_is_test: bool = False) -> Tuple[
    List[ndarray], List[ndarray], List[ndarray], List[ndarray]]:
    from numpy import arange, random, array
    T, X = load_data(data_folder)
    shuffled_idx = arange(len(T))
    random.shuffle(shuffled_idx)
    split_point = int((len(T) * test_size) // 1)
    training_idx = shuffled_idx[split_point:]
    test_idx = shuffled_idx[:split_point]
    test_T = [T[i] for i in test_idx]
    test_X = [X[i] for i in test_idx]
    training_T = [T[i] for i in training_idx]
    training_X = [X[i] for i in training_idx]
    if training_is_test:
        test_T = training_T
        test_X = training_X

    if max_seq_len is not None:
        temp_training_X = []
        temp_training_T = []
        for _ in range(batch_size):
            i = random.randint(low=0, high=len(training_X))
            t = training_T[i]
            x = training_X[i]
            if x.shape[1] > max_seq_len:
                start = random.randint(low=0, high=x.shape[1] - max_seq_len)
                x = x[:, start:start + max_seq_len, :]
                t = t[start:start + max_seq_len]
            temp_training_X.append(x)
            temp_training_T.append(t)
        training_X = temp_training_X
        training_T = temp_training_T

    return test_T, test_X, training_T, training_X


def load_tensor_data(k: int = 5, batch_size: int = 100, seq_size: int = 100,
                     data_folder: str = None, test_size: int = 0.1,
                     validation_size: Union[int, None] = None, validation_k: int = 5):
    from numpy import random, empty

    split_size = test_size
    if validation_size is not None:
        split_size += validation_size

    test_T, test_X, training_T, training_X = load_split_data(data_folder, split_size, batch_size=batch_size)
    val_T, val_X = None, None
    if validation_size is not None:
        validation_size = validation_size / split_size
        val_T = test_T[:int(len(test_T) * validation_size)]
        test_T = test_T[int(len(test_T) * validation_size):]
        val_X = test_X[:int(len(test_X) * validation_size)]
        test_X = test_X[int(len(test_X) * validation_size):]

    # training data
    training_data_X = empty((k, batch_size, seq_size, training_X[0].shape[2]))
    training_data_T = empty((k, seq_size))
    idx = 0
    for n in range(k):
        while training_T[idx].shape[0] < seq_size:
            idx += 1
            idx %= len(training_X)
        start = random.randint(low=0, high=training_T[idx].shape[0] - seq_size)
        for i in range(batch_size):
            while training_X[idx].shape[1] < start + seq_size:
                idx += 1
                idx %= len(training_X)
            training_data_X[n, i, :, :] = training_X[idx][0, start:start + seq_size, :]
            training_data_T[n, :] = training_T[idx][start:start + seq_size]
            idx += 1
            idx %= len(training_X)

    # training data
    validation_data_X = empty((validation_k, batch_size, seq_size, training_X[0].shape[2]))
    validation_data_T = empty((validation_k, seq_size))
    idx = 0
    for n in range(validation_k):
        while val_T[idx].shape[0] < seq_size:
            idx += 1
            idx %= len(val_X)
        start = random.randint(low=0, high=val_T[idx].shape[0] - seq_size)
        for i in range(batch_size):
            while val_X[idx].shape[1] < start + seq_size:
                idx += 1
                idx %= len(val_X)
            validation_data_X[n, i, :, :] = val_X[idx][0, start:start + seq_size, :]
            validation_data_T[n, :] = val_T[idx][start:start + seq_size]
            idx += 1
            idx %= len(val_X)

    return test_T, test_X, training_data_T, training_data_X, validation_data_T, validation_data_X
