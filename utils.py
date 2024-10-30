import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.utils import SimpleLogger


def config_logging(output_dir: str = "output") -> SimpleLogger:
    log_dir = os.path.join(output_dir, "log.txt")
    if os.path.isfile(log_dir):
        with open(log_dir, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            completed = "Finished" in content
            # if completed:
            #     sys.exit()
            # else:
            #     shutil.rmtree(output_dir, ignore_errors=True)

    logging = SimpleLogger(log_dir)
    return logging


def show_data(target: torch.Tensor, pred: torch.Tensor, folder: str, tag, title=""):
    data_show_num = 300
    for i in range(target.shape[0]):
        plt.figure(i)
        target_np = target[i, :, :, :].squeeze(0).cpu().numpy()
        pred_np = pred[i, :, :, :].squeeze(0).cpu().numpy()
        # maxv = np.max(target_np)
        # minv = np.min(target_np)
        # view = maxv - minv
        t = np.array(list(range(data_show_num)))

        k_out = target_np.shape[1]
        for j in range(k_out):
            ax_j = plt.subplot(k_out, 1, j + 1)
            ax_j.plot(t, target_np[:data_show_num, j], "g-")
            ax_j.scatter(t, pred_np[:data_show_num, j], s=2, color="r")
            if j == 0:
                plt.title(title)
        os.makedirs("%s/%s" % (folder, tag), exist_ok=True)
        plt.savefig("%s/%s/trajectory_%d.png" % (folder, tag, i))
        plt.close("all")

def quaternion_to_euler(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=quaternion.dim() - 1)

    euler_angles: torch.Tensor = torch.stack(
        (
            torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            torch.asin(2.0 * (w * y - z * x)),
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        dim=-1,
    )

    return euler_angles
