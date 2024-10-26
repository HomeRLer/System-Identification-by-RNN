import math

import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, k_in: int, k_out: int, k_state: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_in = k_in
        self.k_out = k_out
        self.k_state = k_state

        self.Linear_xh = nn.Linear(k_in, k_state, bias=False)
        self.Linear_hidden_hh = nn.Linear(k_state, k_state)
        self.Linear_output_hy = nn.Linear(k_state, k_out)
        self.tanh = nn.Tanh()

        self.init_param()

    def init_param(self):
        stdv = 1.0 / math.sqrt(self.k_state)
        for weight in self.parameters():
            nn.init.normal_(weight, mean=0, std=stdv)

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
        cell: nn.Module,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k_in = k_in
        self.k_out = k_out
        self.k_state = k_state
        self.cell = cell(k_in, k_out, k_state)
        self.loss = nn.MSELoss()
        self.state0 = nn.Parameter(
            torch.zeros(
                k_state,
            ),
            requires_grad=True,
        )

    def forward(self, input: torch.Tensor, state0: torch.Tensor):
        state = (
            self.state0.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(input.shape[0], input.shape[1], 1, self.k_state)
            if state0 is None
            else state0
        )
        outputs = []
        states = []

        for i in range(input.shape[2]):
            x0 = input[:, :, i, :].unsqueeze(2)
            output, state = self.cell(x0, state)
            outputs.append(output)
            states.append(state)

        outputs = torch.cat(outputs, dim=2)
        states = torch.cat(states, dim=2)
        return outputs, states
