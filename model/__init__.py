import torch
import torch.nn as nn
import yaml

from model.cell import RNNCell
from robot import Robot
from utils import quaternion_to_euler


class MODEL(nn.Module):
    def __init__(
        self,
        k_in: int,
        k_out: int,
        k_state: int,
        cell: str = "RNNCell",
        is_GPU: bool = True,
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
        self, all_input: torch.Tensor, state0: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input = all_input[..., 4:]  # get rid of the quaternion
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

        # UUV Phisycal Dynamics Equations
        M_A_prim = torch.diag_embed(outputs[..., :6])
        tau_prim = outputs[..., 6:12]
        D_V_prim = outputs[..., 12:18]

        f = open("hyper_para.yaml")
        hyper_paras = yaml.safe_load(f)
        rb_mass = torch.diag(torch.tensor(hyper_paras["rigid_body_mass"]))
        if self.is_GPU:
            rb_mass = rb_mass.cuda()

        quaternion = all_input[..., 0:4]  # load the quaternion
        quaternion = quaternion[..., [3, 0, 1, 2]]

        euler = quaternion_to_euler(quaternion)

        body_vel = input[..., :6]

        UUV_robot = Robot(M_A_prim, rb_mass, body_vel, D_V_prim, tau_prim, euler)
        acc_pred = UUV_robot.dynamics_forward()
        
        # sample_time = 0.1
        # Y_pred_v = acc_pred * sample_time + body_vel
        return acc_pred, state


# if __name__=="__main__":
