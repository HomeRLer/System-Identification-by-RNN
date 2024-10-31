import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import data.dataset as dt
import model
from utils import config_logging, normalize_np

f = open("hyper_para.yaml")
para = yaml.safe_load(f)

k_out = para["k_out"]
k_state = para["k_state"]
output_folder = "output_eval"
logging = config_logging(output_folder)

file_dir = "net_eval\eval_dataset.csv"
X_quaternion: np.ndarray = dt.load_data_from_file(
    file_dir, list(range(4, 8))
)  # quaternion
X1_np: np.ndarray = dt.load_data_from_file(file_dir, list(range(8, 14)))  # velocity
X2_np: np.ndarray = dt.load_data_from_file(file_dir, list(range(26, 34)))  # PWM
X1_np = normalize_np(X1_np)
X2_np = normalize_np(X2_np)
X_np: np.ndarray = np.hstack((X_quaternion, X1_np, X2_np))

Y_np: np.ndarray = dt.load_data_from_file(file_dir, list(range(8, 14)))

Y_np = normalize_np(Y_np)

X_tensor = (
    torch.tensor(X_np, dtype=torch.float).unsqueeze(0).unsqueeze(0)
)  # 匹配维度，造孽啊我非得写个四维
Y_tensor = torch.tensor(Y_np, dtype=torch.float).unsqueeze(0).unsqueeze(0)  # 匹配维度
logging("The shape of X is: ", X_tensor.shape, "\nThe shape of Y is: ", Y_tensor.shape)
is_GPU = torch.cuda.is_available()

model_path = os.path.join("save_model", "best_dev_model.pt")

net: model.MODEL = torch.load(model_path)
logging("Loaded model from checkpoint: best_dev_model.pt")

if is_GPU:
    net = net.cuda()
    X_tensor = X_tensor.cuda()
    Y_tensor = Y_tensor.cuda()

error_list = []
start_index = 0
steps = 30

# first epoch for prediction
Y_pred_acc, h_pred = net(X_tensor[:, :, 0, :].unsqueeze(-2), state0=None)
Y_pred_v_next = Y_pred_acc * 0.1 + X_tensor[:, :, 0, 4:10]
Y_target = Y_tensor[:, :, 0, :].unsqueeze(-2)
error = torch.mean(torch.sum(torch.pow((-Y_pred_v_next + Y_target), 2))).item()
error_list.append(error)
Y_pred_v = torch.cat((X_tensor[:, :, 0, 4:10].unsqueeze(-2), Y_pred_v_next), axis=-2)

for i in range(start_index, start_index + steps):
    start_time = time.time()
    Q_Y_pred_v_pwm = torch.cat(
        (
            X_tensor[:, :, : (i + 2), 0:4],
            Y_pred_v,
            X_tensor[:, :, : (i + 2), 10:18],
        ),
        dim=-1,
    )
    Y_pred_acc, h_pred = net(Q_Y_pred_v_pwm, state0=h_pred)
    time_usage = time.time() - start_time
    Y_pred_v_next = (
        Y_pred_v[:, :, -1, :].unsqueeze(-2)
        + Y_pred_acc[:, :, -1, :].unsqueeze(-2) * 0.1
    )
    error = torch.mean(
        torch.sum(
            torch.pow(-Y_pred_v_next + Y_tensor[:, :, i + 2, :].unsqueeze(-2), 2)
        ),
        axis=-1,
    ).item()
    error_list.append(error)
    Y_pred_v = torch.cat((Y_pred_v, Y_pred_v_next), axis=-2)
    logging(i, "th prediction, time usage: %.3f" % time_usage)


error_np = np.array(error_list)
plt.figure(1)
plt.plot(error_np)
plt.savefig(os.path.join(output_folder, "Error_accumulation_plot_%d_steps.png" % steps))
plt.show()
