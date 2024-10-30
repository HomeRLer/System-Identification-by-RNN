import torch


class Robot(object):
    def __init__(
        self,
        a_mass: torch.Tensor,
        rb_mass: torch.Tensor,
        body_vel: torch.Tensor,
        D_V_prim: torch.Tensor,
        tau_prim: torch.Tensor,
        euler: torch.Tensor,
        is_GPU: bool = True,
    ):
        # f = open("hyper_para.yaml")
        # hyper_paras = yaml.safe_load(f)

        self.rb_mass = rb_mass
        self.a_mass = a_mass
        self.shape = self.a_mass.shape[:3]
        self.body_vel = body_vel
        self.D_V_prim = D_V_prim
        self.tau_prim = tau_prim
        self.euler = euler
        self.is_GPU = is_GPU

    def calculate_corilis(self, mass: torch.Tensor, body_vels: torch.Tensor):
        ab = mass @ body_vels.unsqueeze(-1)
        ab = ab.squeeze(-1)
        coriolis = torch.zeros(*self.shape, 6)
        if self.is_GPU:
            coriolis = coriolis.cuda()
        # coriolis.squeeze_(dim=1)
        coriolis[..., 0:3] = -torch.cross(ab[..., 0:3], body_vels[..., 3:6], dim=-1)
        coriolis[..., 3:6] = -(
            torch.cross(ab[..., 0:3], body_vels[..., 0:3], dim=-1)
            + torch.cross(ab[..., 3:6], body_vels[..., 3:6], dim=-1)
        )

        return coriolis

    def calculate_buoyancy(self, rpy):
        buoyancy = torch.zeros(*self.shape, 6)
        if self.is_GPU:
            buoyancy = buoyancy.cuda()
        # buoyancy.squeeze_(dim=1)
        buoyancyForce = 9.8 * self.rb_mass[..., 0, 0] * 1.01
        dis = 0.005
        buoyancy[..., 0] = buoyancyForce * torch.sin(rpy[..., 1])
        buoyancy[..., 1] = (
            -buoyancyForce * torch.sin(rpy[..., 0]) * torch.cos(rpy[..., 1])
        )
        buoyancy[..., 2] = (
            -buoyancyForce * torch.cos(rpy[..., 0]) * torch.cos(rpy[..., 1])
        )
        buoyancy[..., 3] = (
            -dis * buoyancyForce * torch.cos(rpy[..., 1]) * torch.sin(rpy[..., 0])
        )
        buoyancy[..., 4] = -dis * buoyancyForce * torch.sin(rpy[..., 1])
        return buoyancy

    def dynamics_forward(self):
        added_coriolis_V = self.calculate_corilis(
            self.a_mass, self.body_vel
        )  # calculate: C_A(V)V
        rb_coriolis_V = self.calculate_corilis(
            self.rb_mass, self.body_vel
        )  # calculate: C_{RB}(V)V
        D_times_V = (
            torch.diag_embed(self.D_V_prim) @ self.body_vel.unsqueeze(-1)
        ).squeeze(-1)  # calculate: D(V)V
        buoyancy = self.calculate_buoyancy(self.euler)  # calculate: g_{RB}(\eta)

        acc_pred = torch.diagonal(
            torch.inverse(self.rb_mass.expand(*self.shape, 6, 6) + torch.abs(self.a_mass)),
            dim1=-2,
            dim2=-1,
        ) * (self.tau_prim - added_coriolis_V - D_times_V - rb_coriolis_V - buoyancy)
        return acc_pred
