import time

import numpy as np
import torch

from model import MODEL


def prediction_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert (
        prediction.shape == target.shape
    ), "Incompatible truth and prediction for calculating prediction error"

    se = torch.sum(torch.pow((prediction - target), 2), axis=1).unsqueeze(1)  #
    rse = se / torch.sum(torch.pow(target - torch.mean(target, axis=1).unsqueeze(1), 2))
    rrse = torch.mean(torch.sqrt(rse))
    return rrse


class EpochTrainer(object):
    def __init__(
        self,
        model: MODEL,
        optimizer,
        X: torch.Tensor,
        Y: torch.Tensor,
        batch_size: int,
        is_GPU: bool,
    ):
        self.model = model
        self.optimizer = optimizer
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.is_GPU = is_GPU
        self.train_inds: list = list(
            range(self.X.shape[1])
        )  # index list for all segments
        self.state0 = None
        self.loss = model.loss
        self.all_states = None

        print(
            "Initialized epoch trainer: segmented size (trajectories, segments, bptt, in) for X",
            self.X.size(),
            "and (trajectories, segments, bptt, out) for Y",
            self.Y.size(),
        )
        print("for batch size", self.batch_size)

    def set_all_states(self):
        with torch.no_grad():
            all_states = self.model(self.X, state0=None)[1]
            self.all_states = all_states.data

    def __call__(self):
        np.random.shuffle(self.train_inds)

        epoch_loss = 0.0
        cum_bs = 0

        self.model.train()
        self.model.zero_grad()

        Y_pred, _ = self.model(input=self.X[:, 1, :, :].unsqueeze(1), state0=None)
        loss = self.model.loss(Y_pred, self.Y[:, 1, :, :].unsqueeze(1))
        loss.backward()
        self.optimizer.step()

        self.set_all_states()

        for i in range(
            int(np.ceil(len(self.train_inds) / self.batch_size))
        ):  # 对整个数据集的遍历（仅一个 epoch）
            iter_inds = self.train_inds[
                i * self.batch_size : min(
                    (i + 1) * self.batch_size, len(self.train_inds)
                )
            ]
            bs = len(iter_inds)
            cum_bs += bs
            state0 = self.all_states[:, iter_inds, :, :]
            X_train = self.X[
                :, iter_inds, :, :
            ]  # (traj_nums, batch_size, bptt_nums, features_nums) # 把每个 batchsize 拆出来
            Y_target = self.Y[:, iter_inds, :, :]

            Y_pred, _ = self.model(
                X_train, state0
            )  # 连着 state 输入模型，state0 是最新的输出状态，(traj_nums, batch_size, 1, features_nums)
            # 输出的尺寸：Y_pred()

            self.model.train()
            self.model.zero_grad()
            loss = self.model.loss(Y_pred, Y_target)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * bs

        epoch_loss /= cum_bs

        return epoch_loss


if __name__ == "__main__":
    target = torch.randn(2, 30, 20, 18)
    prediction = torch.randn(2, 30, 20, 18)
    se = prediction_error(prediction, target)
    print(se)
