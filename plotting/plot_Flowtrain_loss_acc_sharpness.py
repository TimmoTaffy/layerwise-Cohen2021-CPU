import torch
import matplotlib.pyplot as plt
from os import environ

dataset = "cifar10-5k"
arch = "fc-tanh"
loss = "mse"
flow_tick = 1.0
flow_eig_freq = 1

flow_directory = f"{environ['RESULTS'] }/{dataset}/{arch}/seed_0/{loss}/flow/tick_{flow_tick}"

flow_train_loss = torch.load(f"{flow_directory}/train_loss_final")
flow_train_acc = torch.load(f"{flow_directory}/train_acc_final")
flow_sharpness = torch.load(f"{flow_directory}/eigs_final")[:, 0]

plt.figure(figsize=(5, 5), dpi=100)

plt.subplot(3, 1, 1)
plt.plot(torch.arange(len(flow_train_loss)) * flow_tick, flow_train_loss)
plt.title("train loss")

plt.subplot(3, 1, 2)
plt.plot(torch.arange(len(flow_train_acc)) * flow_tick, flow_train_acc)
plt.title("train accuracy")

plt.subplot(3, 1, 3)
plt.scatter(torch.arange(len(flow_sharpness)) * flow_tick * flow_eig_freq, flow_sharpness, s=5)
plt.title("sharpness")
plt.xlabel("time")

plt.tight_layout()
out_path = f"{flow_directory}/Flow_train_loss_acc_sharpness.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")