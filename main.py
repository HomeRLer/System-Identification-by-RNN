import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data.dataset import MyDataset, load_data_from_file
from model import MODEL
from trainer import EpochTrainer, prediction_error
from utils import config_logging, normalize_np, show_data

# load hyper parameters from "hyper_para.yaml"
f = open("hyper_para.yaml")
hyper_paras = yaml.safe_load(f)

bptt = hyper_paras["bptt"]
k_state = hyper_paras["k_state"]
k_out = hyper_paras["k_out"]
# k_out = 18  # for UUV system identification, MA_prim=output[...,:6], tau_prim = output[...,6:12], E_v_prim = output[:,12:18]
train_rate = hyper_paras["train_rate"]
dev_rate = hyper_paras["dev_rate"]
test_rate = hyper_paras["test_rate"]
learning_rate = hyper_paras["learning_rate"]
epoch_nums = hyper_paras["epoch_nums"]
eval_epochs = hyper_paras["eval_epochs"]
batch_size = hyper_paras["batch_size"]
output_folder = hyper_paras["output_folder"]
dataset_folder = hyper_paras["dataset_folder"]
cell = hyper_paras["cell"]
max_epochs_no_decrease = hyper_paras["max_epochs_no_decrease"]

logging = config_logging(output_folder)
is_GPU = torch.cuda.is_available()
logging("Use GPU?", is_GPU)

file_list = os.listdir(dataset_folder)  # 读取数据集文件夹内每个轨迹的数据

logging("There are %d trajectories in this dataset folder." % len(file_list))

X_traj_list = []
Y_traj_list = []


for file in file_list:
    file_dir = os.path.join(dataset_folder, file)

    X_quaternion: np.ndarray = load_data_from_file(
        file_dir, list(range(4, 8))
    )  # quaternion
    X1_np: np.ndarray = load_data_from_file(file_dir, list(range(8, 14)))  # velocity
    X2_np: np.ndarray = load_data_from_file(file_dir, list(range(26, 34)))  # PWM
    X1_np = normalize_np(X1_np)
    X2_np = normalize_np(X2_np)
    X_np: np.ndarray = np.hstack((X_quaternion, X1_np, X2_np))

    Y_np: np.ndarray = load_data_from_file(file_dir, list(range(8, 14)))

    Y_np = normalize_np(Y_np)

    X_tensor = torch.tensor(X_np, dtype=torch.float).unsqueeze(0)  # 方便之后拼接
    Y_tensor = torch.tensor(Y_np, dtype=torch.float).unsqueeze(0)

    X_traj_list.append(X_tensor)
    Y_traj_list.append(Y_tensor)

X = torch.cat(X_traj_list, dim=0)
Y = torch.cat(Y_traj_list, dim=0)

k_in = X[..., 4:].shape[2]

if is_GPU:
    X = X.cuda()
    Y = Y.cuda()

logging(
    "The Features of X is %d. The sample numbers of X is %d" % (X.shape[1], X.shape[2]),
    "\nThe Features of Y is %d. The sample numbers of Y is %d"
    % (Y.shape[1], Y.shape[2]),
    "\nThe shape of X is: ",
    X[..., 4:].shape,
    "\nThe shape of Y is: ",
    Y.shape,
)

dataset = MyDataset(
    X, Y, bptt=bptt, is_GPU=is_GPU, dev_rate=dev_rate, test_rate=test_rate
)
X_train_seg, X_dev_seg, X_test_seg, Y_train_seg, Y_dev_seg, Y_test_seg = (
    dataset.seg_train_dev_test()
)
X_train, X_dev, X_test, Y_train, Y_dev, Y_test = (
    dataset.divide_train_dev_set_without_seg()
)

logging(
    "The shape of X train set is: ",
    X_train_seg[..., 4:].shape,
    "\nThe shape of Y train set is: ",
    Y_train_seg.shape,
    "\nThe shape of X dev set is: ",
    X_dev_seg[..., 4:].shape,
    "\nThe shape of Y dev set is: ",
    Y_dev_seg.shape,
    "\nThe shape of X test set is: ",
    X_test_seg[..., 4:].shape,
    "\nThe shape of Y test set is: ",
    Y_test_seg.shape,
)
# cell = "RNNCell"
net = MODEL(k_in, k_out, k_state, cell=cell, is_GPU=is_GPU)
if is_GPU:
    net = net.cuda()
state_test = torch.randn(
    k_state,
).cuda()
# Y_pred, state = net(X_train_seg, state_test)
# print(Y_pred.shape)
# print(state.shape)
params_num = sum([np.prod(p.size()) for p in net.parameters()])

logging("Model cell %s contains %d trainable params" % (cell, params_num))

for n, p in net.named_parameters():
    p_params_nums = np.prod(p.size())
    logging("\t", n, "\t", p_params_nums, "(cuda: ", str(p.is_cuda), ")")

# # 如果有已保存的模型，加载它并继续训练
# model_path = os.path.join(output_folder, "best_dev_model.pt")
# if os.path.exists(model_path):
#     net = torch.load(model_path)
#     logging("Loaded model from checkpoint: best_dev_model.pt")
# else:
#     logging("No saved model found, training from scratch.")

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0)

trainer = EpochTrainer(
    net,
    optimizer,
    X_train_seg,  # Including quaternion!
    Y_train_seg,
    batch_size=batch_size,
    is_GPU=True,
)

t00 = time.time()
best_dev_error = 1.0e5
best_dev_epoch = 0
error_test = -1

# max_epochs_no_decrease = 1000
train_loss = []
for epoch in range(1, epoch_nums + 1):
    mse_train = trainer()
    train_loss.append(mse_train)
    plt.figure(1)
    train_loss_np = np.array(train_loss)
    plt.plot(train_loss_np, "b-")
    plt.savefig("%s/train_loss_plot.png" % (output_folder))
    plt.close("all")
    if epoch % eval_epochs == 0:
        with torch.no_grad():
            net.eval()
            Ytrain_pred, htrain_pred = net(X_train, None)
            error_train = prediction_error(Ytrain_pred, Y_train)

            Ydev_pred, hdev_pred = net(X_dev, htrain_pred)
            mse_dev = net.loss(Ydev_pred, Y_dev).item()
            error_dev = prediction_error(Ydev_pred, Y_dev)

            logging(
                "epoch %04d | loss %.3f (train), %.4f (dev) | error %.4f (train), %.4f (dev) | tt %.2fmin"
                % (
                    epoch,
                    mse_train,
                    mse_dev,
                    error_train,
                    error_dev,
                    (time.time() - t00) / 60.0,
                )
            )

            show_data(
                Y_train,
                Ytrain_pred,
                output_folder,
                "current_train_results",
                title="train results (train error %.3f) at iter %d"
                % (error_train, epoch),
            )
            show_data(
                Y_dev,
                Ydev_pred,
                output_folder,
                "current_devresults",
                title="dev results (dev error %.3f) at iter %d" % (error_dev, epoch),
            )
            if error_dev < best_dev_error:
                best_dev_error = error_dev
                best_dev_epoch = epoch

                Ytest_pred, _ = net(X_test, state0=hdev_pred)
                error_test = prediction_error(Ytest_pred, Y_test)
                logging("New best dev error %.3f" % best_dev_error)

                show_data(
                    Y_dev,
                    Ydev_pred,
                    output_folder,
                    "best_dev_results",
                    title="dev results (dev error %.3f) at iter %d"
                    % (error_dev, epoch),
                )
                show_data(
                    Y_test,
                    Ytest_pred,
                    output_folder,
                    "best_test_results",
                    title="test results (test error %.3f) at iter %d"
                    % (error_test, epoch),
                )
                torch.save(net, os.path.join(output_folder, "best_dev_model.pt"))
            elif epoch - best_dev_epoch > max_epochs_no_decrease:
                logging(
                    "Development error did not decrease over %d epochs -- quitting."
                    % max_epochs_no_decrease
                )
                plt.figure(1)
                train_loss = np.array(train_loss)
                plt.plot(train_loss, "b-")
                plt.savefig("%s/train_loss_plot.png" % (output_folder))
                plt.close("all")
                break
logging(
    "Finished: best dev error",
    best_dev_error,
    "at epoch",
    best_dev_epoch,
    "with corresp. test error",
    error_test,
)
