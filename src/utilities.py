from typing import List, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
import os

# the default value for "physical batch size", which is the largest batch size that we try to put on the GPU
DEFAULT_PHYS_BS = 1000


def get_device() -> torch.device:
    """Select the best available device.

    Priority: CUDA > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gd_directory(dataset: str, lr: float, arch_id: str, seed: int, opt: str, loss: str, beta: float = None):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ.get("RESULTS", os.path.expanduser("~/results"))
    directory = f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/{opt}/"
    if opt == "gd":
        return f"{directory}/lr_{lr}"
    elif opt == "polyak" or opt == "nesterov":
        return f"{directory}/lr_{lr}_beta_{beta}"


def get_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, tick: float):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ.get("RESULTS", os.path.expanduser("~/results"))
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/flow/tick_{tick}"


def get_modified_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, gd_lr: float, tick: float):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ.get("RESULTS", os.path.expanduser("~/results"))
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/modified_flow_lr_{gd_lr}/tick_{tick}"


def get_gd_optimizer(parameters, opt: str, lr: float, momentum: float) -> Optimizer:
    if opt == "gd":
        return SGD(parameters, lr=lr)
    elif opt == "polyak":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=False)
    elif opt == "nesterov":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=True)


def save_files(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}")


def save_files_final(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}_final")


def iterate_dataset(dataset: Dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data on the active device."""
    device = get_device()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X, batch_y) in loader:
        yield batch_X.to(device), batch_y.to(device)


def compute_losses(network: nn.Module, loss_functions: List[nn.Module], dataset: Dataset,
                   batch_size: int = DEFAULT_PHYS_BS):
    """Compute loss over a dataset."""
    L = len(loss_functions)
    losses = [0. for l in range(L)]
    with torch.no_grad():
        for (X, y) in iterate_dataset(dataset, batch_size):
            preds = network(X)
            for l, loss_fn in enumerate(loss_functions):
                losses[l] += loss_fn(preds, y) / len(dataset)
    return losses


def get_loss_and_acc(loss: str):
    """Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction. """
    if loss == "mse":
        return SquaredLoss(), SquaredAccuracy()
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction='sum'), AccuracyCE()
    raise NotImplementedError(f"no such loss function: {loss}")


def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset: Dataset, vector: Tensor, physical_batch_size: int = DEFAULT_PHYS_BS, P: Tensor = None):
    """Compute a Hessian-vector product.
    
    If the optional preconditioner P is not set to None, return P^{-1/2} H P^{-1/2} v rather than H v.
    """
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    device = get_device()
    hvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    if P is not None:
        P_device = P.to(device)
        vector = vector / P_device.sqrt()
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    if P is not None:
        hvp = hvp / P_device.sqrt()
    return hvp


def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        # Create vector on the active device; matrix_vector is expected to handle device internally
        gpu_vec = torch.tensor(vec, dtype=torch.float, device=get_device())
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                            neigs=6, physical_batch_size=1000, P=None):
    """ Compute the leading Hessian eigenvalues.

    If preconditioner P is not set to None, return top eigenvalue of P^{-1/2} H P^{-1/2} rather than H.
    """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size, P=P).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals


@torch.no_grad()
def _build_layer_slices(network: torch.nn.Module, combine_by_prefix: bool = False) -> Tuple[slice, ...]:
    """Return slices mapping learnable tensors (optionally grouped by module prefix) to flat indices."""
    idx = 0
    slices: List[slice] = []
    if not combine_by_prefix:
        for param in network.parameters():
            if not param.requires_grad:
                continue
            next_idx = idx + param.numel()
            slices.append(slice(idx, next_idx))
            idx = next_idx
        return tuple(slices)

    prefix_to_range: dict[str, Tuple[int, int]] = {}
    for name, param in network.named_parameters():
        if not param.requires_grad:
            continue
        prefix = name.rsplit(".", 1)[0] if "." in name else name
        start, end = prefix_to_range.get(prefix, (None, None))
        if start is None:
            start = idx
        idx += param.numel()
        end = idx
        prefix_to_range[prefix] = (start, end)

    ordered_prefixes = sorted(prefix_to_range.items(), key=lambda item: item[1][0])
    return tuple(slice(start, end) for _, (start, end) in ordered_prefixes)


def get_layer_slice_count(network: torch.nn.Module, combine_by_prefix: bool = False) -> int:
    """Return how many layer slices would be produced under the chosen grouping."""
    return len(_build_layer_slices(network, combine_by_prefix=combine_by_prefix))


def _hvp(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
         physical_batch_size: int, v_flat: Tensor) -> Tensor:
    """Return the Hessian-vector product (flattened) evaluated at the current network parameters."""
    params = tuple(p for p in network.parameters() if p.requires_grad)
    if not params:
        return torch.zeros(0, device=get_device())

    device = params[0].device
    v_flat = v_flat.to(device)
    hv_flat = torch.zeros_like(v_flat, device=device)
    n_samples = float(len(dataset))

    for xb, yb in iterate_dataset(dataset, physical_batch_size):
        xb = xb.to(device)
        yb = yb.to(device)

        loss_batch = loss_fn(network(xb), yb) / n_samples
        grads = torch.autograd.grad(loss_batch, params, create_graph=True)
        grad_flat = parameters_to_vector(grads)

        grad_dot_v = torch.dot(grad_flat, v_flat)
        hv_parts = torch.autograd.grad(grad_dot_v, params, retain_graph=False)
        hv_flat += parameters_to_vector(tuple(part.detach() for part in hv_parts))

    return hv_flat


def get_layerwise_curvature(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                            physical_batch_size: int, n_power_iter: int = 20,
                            tol: float = 1e-5, seed: int = 0,
                            combine_by_prefix: bool = False) -> Tuple[Tensor, Tensor]:
    """Approximate (lambda_max, lambda_second) for each layer via deflated power iterations."""
    params = tuple(p for p in network.parameters() if p.requires_grad)
    if not params:
        empty = torch.zeros(0)
        return empty, empty

    device = params[0].device
    generator = torch.Generator()
    generator.manual_seed(seed)

    layer_slices = _build_layer_slices(network, combine_by_prefix=combine_by_prefix)
    flat_template = parameters_to_vector(params).detach().to(device)
    max_vals = torch.zeros(len(layer_slices), device=device)
    second_vals = torch.zeros(len(layer_slices), device=device)
    workspace = flat_template.new_zeros(flat_template.shape)

    for idx, layer_slice in enumerate(layer_slices):
        layer_dim = layer_slice.stop - layer_slice.start
        if layer_dim == 0:
            continue

        u = torch.randn(layer_dim, generator=generator, dtype=flat_template.dtype).to(device)
        u = u / (u.norm() + 1e-12)

        lambda_prev = 0.0
        lambda_curr = 0.0

        for _ in range(n_power_iter):
            workspace.zero_()
            workspace[layer_slice] = u
            hv_full = _hvp(network, loss_fn, dataset, physical_batch_size, workspace)
            w = hv_full[layer_slice]
            norm_w = w.norm()
            if norm_w <= 1e-12:
                lambda_curr = 0.0
                break

            u = w / norm_w
            lambda_tensor = torch.dot(u, w)
            lambda_curr = lambda_tensor.item() if torch.isfinite(lambda_tensor) else 0.0

            if abs(lambda_curr - lambda_prev) <= tol * (abs(lambda_prev) + 1e-12):
                break
            lambda_prev = lambda_curr

        max_val = lambda_curr
        second_val = max_val
        has_second = False
        if layer_dim > 1:
            v = torch.randn(layer_dim, generator=generator, dtype=flat_template.dtype).to(device)
            v = v - torch.dot(v, u) * u
            v_norm = v.norm()
            if v_norm > 1e-12:
                v = v / v_norm
                lambda_prev = 0.0
                lambda_curr = 0.0

                for _ in range(n_power_iter):
                    workspace.zero_()
                    workspace[layer_slice] = v
                    hv_full = _hvp(network, loss_fn, dataset, physical_batch_size, workspace)
                    w = hv_full[layer_slice]
                    w = w - torch.dot(w, u) * u
                    norm_w = w.norm()
                    if norm_w <= 1e-12:
                        lambda_curr = 0.0
                        break

                    v = w / norm_w
                    lambda_tensor = torch.dot(v, w)
                    lambda_curr = lambda_tensor.item() if torch.isfinite(lambda_tensor) else 0.0

                    if abs(lambda_curr - lambda_prev) <= tol * (abs(lambda_prev) + 1e-12):
                        break
                    lambda_prev = lambda_curr

                second_val = lambda_curr
                has_second = True

        if has_second and second_val > max_val:
            max_val, second_val = second_val, max_val

        max_vals[idx] = max_val
        second_vals[idx] = second_val if has_second else max_val

    return max_vals.detach().cpu(), second_vals.detach().cpu()


def get_layerwise_max_curvature(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                                physical_batch_size: int, n_power_iter: int = 20,
                                tol: float = 1e-5, seed: int = 0,
                                combine_by_prefix: bool = False) -> Tensor:
    """Approximate the top curvature (largest eigenvalue) of each layer via power iteration."""
    max_vals, _ = get_layerwise_curvature(network, loss_fn, dataset, physical_batch_size,
                                          n_power_iter=n_power_iter, tol=tol, seed=seed,
                                          combine_by_prefix=combine_by_prefix)
    return max_vals


def get_directional_curvature(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                              physical_batch_size: int, direction: Tensor,
                              tol: float = 1e-12,
                              combine_by_prefix: bool = False) -> Tuple[Tensor, Tensor]:
    """Return curvature along a given direction and its per-layer decomposition."""
    params = tuple(p for p in network.parameters() if p.requires_grad)
    if not params:
        zero = torch.zeros(1, dtype=torch.float32)
        return zero, torch.zeros(0, dtype=torch.float32)

    device = params[0].device
    dtype = params[0].dtype
    direction = direction.to(device=device, dtype=dtype)

    if direction.numel() == 0:
        zero = torch.zeros(1, device=device, dtype=dtype)
        return zero.cpu(), torch.zeros(0, dtype=dtype)

    hv_flat = _hvp(network, loss_fn, dataset, physical_batch_size, direction).detach()

    denom = torch.dot(direction, direction)
    if denom.abs() <= tol:
        scalar = torch.zeros(1, device=device, dtype=dtype)
    else:
        scalar = torch.dot(direction, hv_flat) / denom

    layer_slices = _build_layer_slices(network, combine_by_prefix=combine_by_prefix)
    layer_vals = torch.zeros(len(layer_slices), device=device, dtype=dtype)

    for idx, layer_slice in enumerate(layer_slices):
        sub_dir = direction[layer_slice]
        sub_denom = torch.dot(sub_dir, sub_dir)
        if sub_denom.abs() <= tol:
            layer_vals[idx] = 0.0
            continue
        layer_vals[idx] = torch.dot(sub_dir, hv_flat[layer_slice]) / sub_denom

    return scalar.detach().cpu(), layer_vals.detach().cpu()


def compute_gradient(network: nn.Module, loss_fn: nn.Module,
                     dataset: Dataset, physical_batch_size: int = DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at the current network parameters. """
    p = len(parameters_to_vector(network.parameters()))
    device = get_device()
    average_gradient = torch.zeros(p, device=device)
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        batch_loss = loss_fn(network(X), y) / len(dataset)
        batch_gradient = parameters_to_vector(torch.autograd.grad(batch_loss, inputs=network.parameters()))
        average_gradient += batch_gradient
    return average_gradient


class AtParams(object):
    """ Within a with block, install a new set of parameters into a network.

    Usage:

        # suppose the network has parameter vector old_params
        with AtParams(network, new_params):
            # now network has parameter vector new_params
            do_stuff()
        # now the network once again has parameter vector new_params
    """

    def __init__(self, network: nn.Module, new_params: Tensor):
        self.network = network
        self.new_params = new_params

    def __enter__(self):
        self.stash = parameters_to_vector(self.network.parameters())
        vector_to_parameters(self.new_params, self.network.parameters())

    def __exit__(self, type, value, traceback):
        vector_to_parameters(self.stash, self.network.parameters())


def compute_gradient_at_theta(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                              theta: torch.Tensor, batch_size=DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at arbitrary network parameters "theta".  """
    with AtParams(network, theta):
        return compute_gradient(network, loss_fn, dataset, physical_batch_size=batch_size)


class SquaredLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return 0.5 * ((input - target) ** 2).sum()


class SquaredAccuracy(nn.Module):
    def __init__(self):
        super(SquaredAccuracy, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target.argmax(1)).float().sum()


class AccuracyCE(nn.Module):
    def __init__(self):
        super(AccuracyCE, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target).float().sum()


class VoidLoss(nn.Module):
    def forward(self, X, Y):
        return 0

