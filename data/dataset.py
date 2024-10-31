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
        dev_rate: float,
        test_rate: float,
    ):
        self.X = X
        self.Y = Y
        self.N_trajectories = X.shape[0]
        self.bptt = bptt
        self.is_GPU = is_GPU
        self.dev_rate = dev_rate
        self.test_rate = test_rate

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

    def seg_train_dev_test(self, pred_type: str = "next") -> list[torch.Tensor]:
        X_seg = self._segment_traj(self.X, self.bptt)
        Y_seg = self._segment_traj(self.Y, self.bptt)

        logging("The shape of Segmented X is: ", self.X.shape)
        logging("The shape of Segmented Y is: ", self.Y.shape)

        assert X_seg.shape[1] == Y_seg.shape[1]
        N_segments = X_seg.shape[1]
        N_dev, N_train = self.calculate_dev_test(N_segments)

        X_train = X_seg[:, :N_train, :, :]
        X_dev = X_seg[:, N_train : N_train + N_dev, :, :]
        X_test = X_seg[:, N_train + N_dev : -1, :, :]

        if pred_type == "next":
            Y_train = Y_seg[:, 1 : N_train + 1, :, :]
            Y_dev = Y_seg[:, N_train + 1 : N_train + N_dev + 1, :, :]
            Y_test = Y_seg[:, N_train + N_dev + 1 :, :, :]

        elif pred_type == "current":
            Y_train = Y_seg[:, :N_train, :, :]
            Y_dev = Y_seg[:, N_train : N_train + N_dev, :, :]
            Y_test = Y_seg[:, N_train + N_dev : -1, :, :]
        return [X_train, X_dev, X_test, Y_train, Y_dev, Y_test]

    def divide_train_dev_set_without_seg(
        self, pred_type: str = "next"
    ) -> list[torch.Tensor]:
        N_samples = self.X.shape[1]
        assert self.X.shape[1] == self.Y.shape[1]
        N_dev, N_train = self.calculate_dev_test(N_samples)

        X_train = self.X[:, :N_train, :].unsqueeze(1)
        X_dev = self.X[:, N_train : N_train + N_dev, :].unsqueeze(1)
        X_test = self.X[:, N_train + N_dev : -1, :].unsqueeze(1)

        if pred_type == "next":
            Y_train = self.Y[:, 1 : N_train + 1, :].unsqueeze(1)
            Y_dev = self.Y[:, N_train + 1 : N_train + N_dev + 1, :].unsqueeze(1)
            Y_test = self.Y[:, N_train + N_dev + 1 :, :].unsqueeze(1)

        if pred_type == "current":
            Y_train = self.Y[:, :N_train, :].unsqueeze(1)
            Y_dev = self.Y[:, N_train : N_train + N_dev, :].unsqueeze(1)
            Y_test = self.Y[:, N_train + N_dev : -1, :].unsqueeze(1)

        return [X_train, X_dev, X_test, Y_train, Y_dev, Y_test]

    def calculate_dev_test(self, N):
        # N_segments = X_seg.shape[1]
        N_dev = int(N * self.dev_rate)
        N_test = int(N * self.test_rate)
        N_train = N - N_dev - N_test
        return N_dev, N_train
