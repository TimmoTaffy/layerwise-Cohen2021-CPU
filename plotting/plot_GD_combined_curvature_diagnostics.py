from pathlib import Path
import os

import matplotlib.pyplot as plt
import torch

DATASET = "cifar10-5k"
ARCH = "fc-tanh"
LOSS = "mse"
GD_LR = 0.08
SEED = 0
CURVATURE_FREQ = 50
EIG_FREQ = 50

def _resolve_results_dir() -> Path:
    """Return the path holding results for the configured run."""
    root = os.environ.get("RESULTS", os.path.expanduser("~/results"))
    return Path(root) / DATASET / ARCH / f"seed_{SEED}" / LOSS / "gd" / f"lr_{GD_LR}"

def _load_tensor(directory: Path, name: str) -> torch.Tensor:
    """Convenience loader that always returns a CPU tensor."""
    return torch.load(directory / name).detach().cpu()

def main() -> None:
    gd_directory = _resolve_results_dir()

    train_loss = _load_tensor(gd_directory, "train_loss_final")
    layer_eigs = _load_tensor(gd_directory, "layer_eigs_final")
    layer_gaps = _load_tensor(gd_directory, "layer_eig_gaps_final")
    directional_curv = _load_tensor(gd_directory, "directional_curv_final")
    layer_directional = _load_tensor(gd_directory, "layer_directional_curv_final")

    loss_steps = torch.arange(train_loss.numel())
    curvature_steps = torch.arange(layer_eigs.shape[0]) * CURVATURE_FREQ

    try:
        eigs = _load_tensor(gd_directory, "eigs_final")
        if eigs.numel() == 0:
            raise ValueError("empty eigenvalue log")
        sharpness = eigs[:, 0]
        step_size = EIG_FREQ if EIG_FREQ > 0 else 1
        sharpness_steps = torch.arange(sharpness.shape[0]) * step_size
    except (FileNotFoundError, ValueError):
        # Fallback: take the maximum layer-wise curvature as a proxy for sharpness.
        sharpness = layer_eigs.max(dim=1).values
        sharpness_steps = curvature_steps.clone()

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=120)

    # Train loss
    axes[0, 0].plot(loss_steps.numpy(), train_loss.numpy())
    axes[0, 0].set_title("Gradient Descent Train Loss")
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].set_ylabel("loss")

    # Sharpness (or fallback)
    axes[0, 1].plot(sharpness_steps.numpy(), sharpness.numpy())
    stability_line = 2.0 / GD_LR if GD_LR > 0 else None
    if stability_line is not None:
        axes[0, 1].axhline(stability_line, linestyle="dotted", color="tab:gray", linewidth=1)
    axes[0, 1].set_title("Gradient Descent Sharpness")
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel("sharpness")

    # Layer-wise max curvature
    for idx in range(layer_eigs.shape[1]):
        axes[1, 0].plot(curvature_steps.numpy(), layer_eigs[:, idx].numpy(), label=f"layer {idx + 1}")
    axes[1, 0].set_title("Layer-wise Curvature")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel("curvature")
    axes[1, 0].legend(loc="best", fontsize="small")

    # Layer-wise spectral gap
    for idx in range(layer_gaps.shape[1]):
        axes[1, 1].plot(curvature_steps.numpy(), layer_gaps[:, idx].numpy(), label=f"layer {idx + 1}")
    axes[1, 1].set_title("Layer-wise Spectral Gap")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].set_ylabel("spectral gap")
    axes[1, 1].legend(loc="best", fontsize="small")

    # Directional curvature
    axes[2, 0].plot(curvature_steps.numpy(), directional_curv.numpy())
    axes[2, 0].set_title("Directional Curvature")
    axes[2, 0].set_xlabel("iteration")
    axes[2, 0].set_ylabel("curvature")

    # Layer-wise directional curvature
    for idx in range(layer_directional.shape[1]):
        axes[2, 1].plot(curvature_steps.numpy(), layer_directional[:, idx].numpy(), label=f"layer {idx + 1}")
    axes[2, 1].set_title("Layer-wise Directional Curvature")
    axes[2, 1].set_xlabel("iteration")
    axes[2, 1].set_ylabel("directional curvature")
    axes[2, 1].legend(loc="best", fontsize="small")

    fig.tight_layout()
    out_path = gd_directory / "gd_combined_metrics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()