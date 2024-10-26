import os

import numpy as np
import torch

from data.dataset import MyDataset, load_data_from_file
from model import MODEL, RNNCell
from utils import config_logging

bptt = 20
train_rate = 0.7
dev_rate = 0.15
test_rate = 0.15
k_state = 20  # 隐藏状态数
## Load data from file
logging = config_logging()
is_GPU = torch.cuda.is_available()
logging("Use GPU?", is_GPU)

folder_dir = "datasets_test"
file_list = os.listdir(folder_dir)  # 读取数据集文件夹内每个轨迹的数据

logging("There are %d trajectories in this dataset folder." % len(file_list))

X_traj_list = []
Y_traj_list = []


def normalize_np(A: np.ndarray):
    return (A - np.max(A, axis=0)) / (np.max(A, axis=0) - np.min(A, axis=0))


for file in file_list:
    file_dir = os.path.join(folder_dir, file)

    X1_np: np.ndarray = load_data_from_file(file_dir, list(range(1, 14)))
    X2_np: np.ndarray = load_data_from_file(file_dir, list(range(26, 34)))
    X_np: np.ndarray = np.hstack((X1_np, X2_np))

    Y_np: np.ndarray = load_data_from_file(file_dir, list(range(11, 14)))

    X_np = normalize_np(X_np)
    Y_np = normalize_np(Y_np)

    X_tensor = torch.tensor(X_np, dtype=torch.float).unsqueeze(0)  # 方便之后拼接
    Y_tensor = torch.tensor(Y_np, dtype=torch.float).unsqueeze(0)

    X_traj_list.append(X_tensor)
    Y_traj_list.append(Y_tensor)

X = torch.cat(X_traj_list, dim=0)
Y = torch.cat(Y_traj_list, dim=0)

k_in = X.shape[2]
k_out = Y.shape[2]

if is_GPU:
    X = X.cuda()
    Y = Y.cuda()

logging(
    "The Features of X is %d. The sample numbers of X is %d" % (X.shape[1], X.shape[2]),
    "\nThe Features of Y is %d. The sample numbers of Y is %d"
    % (Y.shape[1], Y.shape[2]),
    "\nThe shape of X is: ",
    X.shape,
    "\nThe shape of Y is: ",
    Y.shape,
)

dataset = MyDataset(X, Y, bptt=bptt, is_GPU=is_GPU)
X_train, X_dev, X_test, Y_train, Y_dev, Y_test = dataset.get_train_dev_test(
    dev_rate, test_rate
)

logging(
    "The shape of X train set is: ",
    X_train.shape,
    "\nThe shape of Y train set is: ",
    Y_train.shape,
    "\nThe shape of Y dev set is: ",
    Y_dev.shape,
    "\nThe shape of X test set is: ",
    X_test.shape,
    "\nThe shape of Y test set is: ",
    Y_test.shape,
)
net = MODEL(k_in, k_out, k_state, cell=RNNCell)
if is_GPU:
    net = net.cuda()
outputs, states = net(X_train, state0=None)
