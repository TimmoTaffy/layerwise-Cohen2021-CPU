import torch
import matplotlib.pyplot as plt
from os import environ

dataset = "cifar10-5k"
arch = "fc-tanh"
loss = "mse"
gd_lr = 0.01
flow_tick = 1.0

gd_iterate_freq = 100
flow_iterate_freq = 1

gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
flow_directory = f"{environ['RESULTS'] }/{dataset}/{arch}/seed_0/{loss}/flow/tick_{flow_tick}"

# the GD iterates are saved every "gd_lr * gd_iterate_freq" units of time.
# the GF iterates are saved every "flow_tick * flow_iterate_freq" units of time.
# to directly compare the trajectories, these two quantities should be equal.
assert gd_lr * gd_iterate_freq == flow_tick * flow_iterate_freq

gd_iterates = torch.load(f"{gd_directory}/iterates_final")
flow_iterates = torch.load(f"{flow_directory}/iterates_final")
length = min(len(gd_iterates), len(flow_iterates))
gd_sharpness = torch.load(f"{gd_directory}/eigs_final")[:,0]

times = torch.arange(length) * flow_tick * flow_iterate_freq
distance = (gd_iterates[:length, :] - flow_iterates[:length, :]).norm(dim=1)

# the time at which the gradient descent sharpness first crosses the threshold (2  / gd_lr)
cross_threshold_time = (gd_sharpness > (2 / gd_lr)).nonzero()[0][0] * gd_lr * gd_iterate_freq

plt.figure(figsize=(5, 2), dpi=100)
plt.scatter(times, distance)
plt.axvline(cross_threshold_time, linestyle='dotted')
plt.ylim((0, plt.ylim()[1]))
plt.title("distance betwen gradient descent and gradient flow")
plt.xlabel("time")
plt.ylabel("distance")

plt.tight_layout()
out_path = f"{gd_directory}/GD_GF_distance.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")