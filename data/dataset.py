import numpy as np
import pandas as pd
import torch

from utils import config_logging

logging = config_logging()


def load_data_from_file(file_dir: str, col_index: list) -> np.ndarray:
    return pd.read_csv(file_dir).to_numpy()[:, col_index]


class MyDataset:
    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        bptt: int,
        is_GPU: bool,
    ):
        self.X = X
        self.Y = Y
        self.N_trajectories = X.shape[0]
        self.bptt = bptt
        self.is_GPU = is_GPU
        self.X = self._segment_traj(self.X, self.bptt)
        self.Y = self._segment_traj(self.Y, self.bptt)
        logging("The shape of Segmented X is: ", self.X.shape)
        logging("The shape of Segmented Y is: ", self.Y.shape)

    def _segment_traj(self, tensor: torch.Tensor, bptt: int) -> torch.Tensor:
        assert len(tensor.shape) == 3
        N = tensor.shape[1]
        tensor_list = []
        for i in range(max(1, N - bptt + 1)):
            segment_tensor = tensor[:, i : min(N, i + bptt), :].unsqueeze(1)
            tensor_list.append(segment_tensor)
        if self.is_GPU:
            return torch.cat(tensor_list, dim=1).cuda()
        else:
            return torch.cat(tensor_list, dim=1)

    def get_train_dev_test(
        self,
        dev_rate: float,
        test_rate: float,
    ) -> list[torch.Tensor]:
        assert self.X.shape[1] == self.Y.shape[1]
        N_segments = self.X.shape[1]
        N_dev = int(N_segments * dev_rate)
        N_test = int(N_segments * test_rate)
        N_train = N_segments - N_dev - N_test

        X_train = self.X[:, :N_train, :, :]
        X_dev = self.X[:, N_train : N_train + N_dev, :, :]
        X_test = self.X[:, N_train + N_dev : -1, :, :]

        Y_train = self.Y[:, 1 : N_train + 1, :, :]
        Y_dev = self.Y[:, N_train + 1 : N_train + N_dev + 1, :, :]
        Y_test = self.Y[:, N_train + N_dev + 1 :, :, :]
        return [X_train, X_dev, X_test, Y_train, Y_dev, Y_test]
