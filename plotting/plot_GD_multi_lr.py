import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import matplotlib.pyplot as plt

# --------- configurable inputs ---------
dataset = "cifar10-5k"
arch = "fc-tanh"
loss = "mse"
seed = 0

# LRs to compare (as requested)
lrs: List[float] = [0.005, 0.01, 0.02, 0.04, 0.08]

# eig_freq is auto-inferred per run; this is only a fallback if inference is impossible
default_eig_freq = 20

# Where to read results; fall back to repo's results/ if env var not set
repo_root = Path(__file__).resolve().parents[1]
results_root = os.environ.get("RESULTS", str(repo_root / "results"))
# Save combined figures to my_results as requested
figures_dir = repo_root / "my_results"
figures_dir.mkdir(parents=True, exist_ok=True)


def lr_dir(lr: float) -> Path:
    return Path(results_root) / dataset / arch / f"seed_{seed}" / loss / "gd" / f"lr_{lr}"


def try_load_tensor(path: Path):
    try:
        return torch.load(str(path))
    except Exception:
        return None


def mask_nan(t: torch.Tensor) -> torch.Tensor:
    if t is None:
        return None
    if not torch.is_tensor(t):
        return None
    if t.ndim == 0:
        return t
    mask = ~torch.isnan(t)
    if mask.ndim == 1:
        return t[mask]
    return t


def infer_eig_freq(run_dir: Path, fallback: Optional[int] = None) -> int:
    """Infer eig_freq from the saved arrays if possible.

    Uses the identity: num_eigs ≈ floor(last_step / eig_freq) + 1,
    with last_step = len(train_loss) - 1. Solve for eig_freq ≈ (len(train_loss)-1)/(len(eigs)-1).
    """
    train_loss = try_load_tensor(run_dir / "train_loss_final")
    eigs = try_load_tensor(run_dir / "eigs_final")
    try:
        if train_loss is not None and eigs is not None and len(eigs) >= 2:
            steps = int(len(train_loss) - 1)
            ne = int(len(eigs) - 1)
            if ne > 0 and steps > 0:
                est = max(1, int(round(steps / ne)))
                return est
    except Exception:
        pass
    # fall back to provided default or 20
    return fallback if fallback is not None else 20


def integer_eig_freq_from_lr(lr: float, tol: float = 1e-8) -> Optional[int]:
    """Return int(1/lr) if 1/lr is (near) an integer; otherwise None.

    Uses a small tolerance to handle floating point rounding.
    """
    if lr <= 0:
        return None
    val = 1.0 / lr
    n = int(round(val))
    if abs(val - n) < tol and n >= 1:
        return n
    return None


# -------------- plot train loss --------------
plt.figure(figsize=(6.5, 4.0), dpi=120)
colors = plt.cm.tab10.colors

# Infer eig_freq per learning rate once and report for transparency
inferred_ef: Dict[float, int] = {}
for lr in lrs:
    d = lr_dir(lr)
    ef_from_lr = integer_eig_freq_from_lr(lr)
    if ef_from_lr is not None:
        ef = ef_from_lr
        method = "1/lr"
    else:
        ef = infer_eig_freq(d, fallback=default_eig_freq)
        method = "inferred"
    inferred_ef[lr] = ef
    print(f"[info] eig_freq for lr={lr}: {ef} (method={method})")

available_lrs_loss = []
for i, lr in enumerate(lrs):
    d = lr_dir(lr)
    train_loss = try_load_tensor(d / "train_loss_final")
    if train_loss is None:
        print(f"[skip] missing: {d}/train_loss_final")
        continue
    train_loss = mask_nan(train_loss)
    if train_loss is None or len(train_loss) == 0:
        print(f"[skip] empty/NaN: {d}/train_loss_final")
        continue
    xs = torch.arange(len(train_loss))
    plt.plot(xs, train_loss, label=f"lr={lr}", color=colors[i % len(colors)])
    available_lrs_loss.append(lr)

plt.title(f"GD train loss — {dataset}, {arch}, {loss}")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend()
plt.grid(alpha=0.2)

loss_out = figures_dir / f"GD_loss_multi_lr_{dataset}_{arch}_{loss}_seed{seed}.png"
plt.tight_layout()
plt.savefig(loss_out, dpi=150)
print(f"[saved] {loss_out}")


# -------------- plot sharpness (top eigenvalue) by iteration --------------
plt.figure(figsize=(6.5, 4.0), dpi=120)
available_lrs_eigs = []
for i, lr in enumerate(lrs):
    d = lr_dir(lr)
    eigs = try_load_tensor(d / "eigs_final")
    if eigs is None:
        print(f"[skip] missing: {d}/eigs_final")
        continue
    if eigs.ndim == 1:
        sharp = eigs
    else:
        sharp = eigs[:, 0]
    sharp = mask_nan(sharp)
    if sharp is None or len(sharp) == 0:
        print(f"[skip] empty/NaN: {d}/eigs_final")
        continue
    ef = inferred_ef[lr]
    xs = torch.arange(len(sharp)) * ef
    color = colors[i % len(colors)]
    plt.plot(xs, sharp, label=f"lr={lr} (ef={ef})", color=color)
    # Add horizontal dashed stability line at 2 / lr, same color, thin linewidth, no legend entry
    plt.axhline(2.0 / lr, linestyle='dotted', linewidth=0.8, color=color)
    available_lrs_eigs.append(lr)

plt.title(f"GD sharpness (by iteration) — {dataset}, {arch}, {loss}")
plt.xlabel("iteration")
plt.ylabel("sharpness")
plt.legend()
plt.grid(alpha=0.2)

sharp_out = figures_dir / f"GD_sharpness_multi_lr_{dataset}_{arch}_{loss}_seed{seed}.png"
plt.tight_layout()
plt.savefig(sharp_out, dpi=150)
print(f"[saved] {sharp_out}")


# -------------- plot sharpness by time (iteration × lr) --------------
plt.figure(figsize=(6.5, 4.0), dpi=120)
for i, lr in enumerate(lrs):
    d = lr_dir(lr)
    eigs = try_load_tensor(d / "eigs_final")
    if eigs is None:
        continue
    sharp = eigs if eigs.ndim == 1 else eigs[:, 0]
    sharp = mask_nan(sharp)
    if sharp is None or len(sharp) == 0:
        continue
    ef = inferred_ef[lr]
    times = torch.arange(len(sharp)) * ef * lr
    color = colors[i % len(colors)]
    plt.plot(times, sharp, label=f"lr={lr} (ef={ef})", color=color)
    # stability threshold line remains at 2/lr (same y), independent of time scaling
    plt.axhline(2.0 / lr, linestyle='dotted', linewidth=0.8, color=color)

plt.title(f"GD sharpness (by time) — {dataset}, {arch}, {loss}")
plt.xlabel("time = iteration × η")
plt.ylabel("sharpness")
plt.legend()
plt.grid(alpha=0.2)

sharp_time_out = figures_dir / f"GD_sharpness_by_time_multi_lr_{dataset}_{arch}_{loss}_seed{seed}.png"
plt.tight_layout()
plt.savefig(sharp_time_out, dpi=150)
print(f"[saved] {sharp_time_out}")