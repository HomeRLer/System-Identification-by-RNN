import math

import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, k_in: int, k_out: int, k_state: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 这里的 input 只有一个时间步
        self.k_in = k_in
        self.k_out = k_out
        self.k_state = k_state

        self.Linear_xh = nn.Linear(k_in, k_state, bias=False)
        self.Linear_hidden_hh = nn.Linear(k_state, k_state)
        self.Linear_output_hy = nn.Linear(k_state, k_out)
        self.tanh = nn.Tanh()

        self.init_param()

    def init_param(self):
        # stdv = 1.0 / math.sqrt(self.k_state)
        for weight in self.parameters():
            nn.init.normal_(weight, mean=0, std=1.0)

    def forward(self, input: torch.Tensor, state: torch.Tensor):
        state_new = self.tanh(self.Linear_xh(input) + self.Linear_hidden_hh(state))
        y_new = self.Linear_output_hy(state_new)
        return y_new, state_new


class MODEL(nn.Module):
    def __init__(
        self,
        k_in: int,
        k_out: int,
        k_state: int,
        cell: str,
        is_GPU: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k_in = k_in
        self.k_out = k_out
        self.k_state = k_state
        self.model_dict = {"RNNCell": RNNCell}
        self.loss = nn.MSELoss()
        self.state0 = nn.Parameter(
            torch.zeros(
                k_state,
            ),
            requires_grad=True,
        )
        self.cell = self.model_dict[cell](k_in, k_out, k_state)
        self.is_GPU = is_GPU
        if self.is_GPU:
            self.cell.cuda()

    def forward(
        self, input: torch.Tensor, state0: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state0 is None:
            state = torch.zeros(input.shape[0], input.shape[1], 1, self.k_state)
            if self.is_GPU:
                state = state.cuda()
        else:
            state = state0
        # state0: (traj_nums, segments_nums, 1, k_state)
        # x0: (traj_nums, segments_nums, bptt, k_in)
        outputs = []

        for i in range(input.shape[2]):
            x0 = input[:, :, i, :].unsqueeze(2)
            output, state = self.cell(x0, state)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=2)
        return outputs, state


# if __name__=="__main__":
