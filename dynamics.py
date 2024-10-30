
import torch

def calculate_corilis(self, body_vels):
    ab = self.added_mass_matrix @ body_vels.unsqueeze(2)
    ab =ab.squeeze(2)
    coriolis = torch.zeros(*self.shape, 6, device=self.device)
    coriolis.squeeze_(dim=1)
    coriolis[:, 0:3] = - torch.cross(ab[:, 0:3], body_vels[:, 3:6], dim=1)
    coriolis[:, 3:6] = - (torch.cross(ab[:, 0:3], body_vels[:, 0:3], dim=1) + torch.cross(ab[:, 3:6], body_vels[:, 3:6], dim=1))
    
    return coriolis


def calculate_buoyancy(self, rpy):
    buoyancy = torch.zeros(*self.shape, 6, device=self.device)
    buoyancy.squeeze_(dim=1)
    buoyancyForce = 9.8 * self.masses[:,0,0] * 1.01
    dis = 0.005
    buoyancy[:, 0] = buoyancyForce * torch.sin(rpy[:,1])
    buoyancy[:, 1] = -buoyancyForce * torch.sin(rpy[:,0]) * torch.cos(rpy[:,1])
    buoyancy[:, 2] = -buoyancyForce * torch.cos(rpy[:,0]) * torch.cos(rpy[:,1])
    buoyancy[:, 3] = - dis * buoyancyForce * torch.cos(rpy[:,1]) * torch.sin(rpy[:,0])
    buoyancy[:, 4] = - dis * buoyancyForce * torch.sin(rpy[:,1])
    
    return buoyancy

# if __name__=="__main__":
#     quaternion = torch.Tensor([0.011276,0.00081,0.259857,0.965581])
#     euler = quaternion_to_euler(quaternion)
#     print(euler)