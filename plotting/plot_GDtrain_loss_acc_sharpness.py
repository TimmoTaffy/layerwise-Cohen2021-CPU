import torch
import matplotlib.pyplot as plt
from os import environ

dataset = "cifar10-5k"
arch = "fc-tanh"
loss = "mse"
gd_lr = 0.04
gd_eig_freq = 25

gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"

gd_train_loss = torch.load(f"{gd_directory}/train_loss_final")
gd_train_acc = torch.load(f"{gd_directory}/train_acc_final")
gd_sharpness = torch.load(f"{gd_directory}/eigs_final")[:,0]

plt.figure(figsize=(5, 5), dpi=100)

plt.subplot(3, 1, 1)
plt.plot(gd_train_loss)
plt.title("train loss")

plt.subplot(3, 1, 2)
plt.plot(gd_train_acc)
plt.title("train accuracy")

plt.subplot(3, 1, 3)
plt.scatter(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, s=5)
plt.axhline(2. / gd_lr, linestyle='dotted')
plt.title("sharpness")
plt.xlabel("iteration")

plt.tight_layout()
out_path = f"{gd_directory}/GD_train_loss_acc_sharpness.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")