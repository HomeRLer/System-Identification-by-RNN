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
