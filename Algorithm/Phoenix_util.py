import matplotlib
matplotlib.use('Agg')
import copy
import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np


def reset_kernels_for_task(
    model,
    reset_ratio,
    seed,
    layer_scope="all",
    scale_factor=1.0,
    init_method="uniform",
    at_least_one=True,
    current_iter=None,
    conv_transition_period=None,
):
    """Apply a deterministic FedPhoenix-style reset and return its task trace.

    The original training routine intentionally draws a fresh random reset for
    every local copy.  A motivation experiment needs to replay *the same* reset
    task on several clients, otherwise client and reset randomness are
    confounded.  This helper therefore uses private RNG objects and records the
    reset kernel indices without changing the global random state.

    ``layer_scope`` can be ``all``, ``early``, ``middle`` or ``late``.  The
    default ``all`` mirrors a FedPhoenix round in which all convolutional layers
    are still active.  Layer scopes are diagnostic variants for checking
    whether client rankings change with reset depth.
    """
    if not 0 <= reset_ratio <= 1:
        raise ValueError("reset_ratio must be between 0 and 1")
    if layer_scope not in {"all", "early", "middle", "late"}:
        raise ValueError(
            "layer_scope must be one of: all, early, middle, late"
        )
    if init_method not in {
        "uniform",
        "kaiming_uniform",
        "kaiming_normal",
        "ori_normal",
    }:
        raise ValueError("Unsupported initialization method")

    conv_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    if not conv_layers:
        raise ValueError("The model has no convolutional layers to reset")

    if (current_iter is None) != (conv_transition_period is None):
        raise ValueError(
            "current_iter and conv_transition_period must be provided together"
        )
    if current_iter is not None:
        total_conv = len(conv_layers)
        conv_layers = [
            (name, layer)
            for depth, (name, layer) in enumerate(conv_layers)
            if current_iter
            < (depth + 1) * (float(conv_transition_period) / total_conv)
        ]
        if not conv_layers:
            return {
                "seed": int(seed),
                "layer_scope": layer_scope,
                "requested_reset_ratio": float(reset_ratio),
                "init_method": init_method,
                "current_iter": int(current_iter),
                "conv_transition_period": float(conv_transition_period),
                "layers": [],
                "num_reset_layers": 0,
                "num_reset_kernels": 0,
            }

    if layer_scope == "all":
        selected_layers = conv_layers
    else:
        # Split depth into three non-empty-friendly bands.  With a very shallow
        # CNN, early and late naturally select its first and last conv layers.
        depth_ids = np.array_split(np.arange(len(conv_layers)), 3)
        scope_index = {"early": 0, "middle": 1, "late": 2}[layer_scope]
        selected_ids = depth_ids[scope_index]
        if len(selected_ids) == 0:
            selected_ids = np.array(
                [round(scope_index * (len(conv_layers) - 1) / 2)], dtype=int
            )
        selected_layers = [conv_layers[int(idx)] for idx in selected_ids]

    python_rng = random.Random(int(seed))
    torch_rng = torch.Generator(device="cpu")
    torch_rng.manual_seed(int(seed))
    trace = {
        "seed": int(seed),
        "layer_scope": layer_scope,
        "requested_reset_ratio": float(reset_ratio),
        "init_method": init_method,
        "current_iter": int(current_iter) if current_iter is not None else None,
        "conv_transition_period": (
            float(conv_transition_period)
            if conv_transition_period is not None
            else None
        ),
        "layers": [],
    }

    with torch.no_grad():
        for name, layer in selected_layers:
            if layer.weight.device.type != "cpu":
                raise ValueError(
                    "reset_kernels_for_task expects a CPU model so that a task "
                    "is exactly reproducible across devices"
                )

            num_kernels = int(layer.weight.shape[0])
            num_reset = int(num_kernels * reset_ratio)
            if reset_ratio > 0 and at_least_one:
                num_reset = max(1, num_reset)
            num_reset = min(num_kernels, num_reset)
            reset_indices = sorted(python_rng.sample(range(num_kernels), num_reset))

            mean = layer.weight.data.mean().item()
            std = layer.weight.data.std().item()
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)

            for idx in reset_indices:
                shape = layer.weight[idx].shape
                dtype = layer.weight.dtype
                if init_method == "uniform":
                    sample = torch.rand(shape, generator=torch_rng, dtype=dtype)
                    new_kernel = sample * (2 * std * scale_factor)
                    new_kernel += mean - std * scale_factor
                elif init_method == "kaiming_uniform":
                    bound = np.sqrt(6.0 / fan_in) * scale_factor
                    sample = torch.rand(shape, generator=torch_rng, dtype=dtype)
                    new_kernel = sample * (2 * bound) - bound
                elif init_method == "kaiming_normal":
                    reset_std = np.sqrt(2.0 / fan_in) * scale_factor
                    new_kernel = (
                        torch.randn(shape, generator=torch_rng, dtype=dtype)
                        * reset_std
                    )
                else:  # ori_normal
                    new_kernel = (
                        torch.randn(shape, generator=torch_rng, dtype=dtype)
                        * std
                        * scale_factor
                        + mean
                    )
                layer.weight[idx].copy_(new_kernel)

            trace["layers"].append(
                {
                    "name": name,
                    "num_kernels": num_kernels,
                    "reset_indices": reset_indices,
                    "actual_reset_ratio": (
                        float(num_reset / num_kernels) if num_kernels else 0.0
                    ),
                }
            )

    trace["num_reset_layers"] = len(trace["layers"])
    trace["num_reset_kernels"] = sum(
        len(layer_trace["reset_indices"]) for layer_trace in trace["layers"]
    )
    return trace

def get_reset_probability(current_round, m, p, initial_mu):
    """计算当前轮次的Reset概率"""
    if current_round <= m:
        return initial_mu
    elif current_round <= m + p:
        # 线性递减
        progress = (current_round - m) / p
        return initial_mu * (1 - progress)
    else:
        return 0.0

    

def zero_out_model_params(model, percent_set_zero_base):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
        
            percent_set_zero = percent_set_zero_base*  1 
        elif isinstance(layer, nn.Linear):
            # 全连接层设定较高的置零比例
            percent_set_zero = percent_set_zero_base *1
    
        else:

            percent_set_zero = percent_set_zero_base*0
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if hasattr(layer, 'weight'):
                param = layer.weight
                total_params = param.numel()
                num_zero_params = int(total_params * percent_set_zero)
                mask = torch.ones_like(param)
                zero_indices = np.random.choice(total_params, num_zero_params, replace=False)
                mask.view(-1)[zero_indices] = 0
                param.data.mul_(mask)
    return model







def get_test_image(dataset, seed=42):
    """
    Get a test image from the dataset
    Args:
        dataset: CIFAR10 dataset
        seed: random seed
    Returns:
        original image and normalized image
    """
    torch.manual_seed(seed)
    idx = torch.randint(len(dataset), (1,)).item()
    img, label = dataset[idx]
    
    # Denormalize the image for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
    orig_img = img * std + mean
    
    return orig_img, img, label




def reset_kernels_and_neurons_stair(model, current_iter, conv_transition_period,conv_reset_ratio=0.03125, fc_reset_ratio=0, scale_factor=1, init_method='ori_normal'):
    """
    Stop resetting convolutional kernels and fully connected layer neurons in a stepwise manner from shallow to deep layers within the specified number of iterations
    
    Args:
    model: PyTorch model
    current_iter: current training iteration
    conv_transition_period: total number of iterations required to complete resetting all convolutional layers
    conv_reset_ratio: reset ratio for convolutional layers
    fc_reset_ratio: reset ratio for fully connected layers
    scale_factor: scaling factor
    init_method: initialization method, can be 'uniform', 'kaiming_uniform', 'kaiming_normal'

    Returns:
        reset_count: number of layers that have been reset
    """
    if not (0 <= conv_reset_ratio <= 1 and 0 <= fc_reset_ratio <= 1):
        raise ValueError("Reset ratio must be between 0 and 1")

    # Separately get convolutional layers and fully connected layers
    conv_layers = []
    fc_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
        elif isinstance(module, nn.Linear):
            fc_layers.append((name, module))

    reset_count = 0
    stopped_count = 0  # Used to count the number of layers that have stopped resetting

    # Process convolutional layers
    total_conv = len(conv_layers)
    for depth, (name, layer) in enumerate(conv_layers):
        # Calculate the iteration threshold for this layer to stop resetting
        stop_iter = (depth + 1) * (conv_transition_period / total_conv)

        # If the current iteration has exceeded the stop threshold for this layer, skip this layer
        if current_iter >= stop_iter:
            stopped_count += 1  # Count the number of layers that have stopped resetting
            continue

        current_mean = layer.weight.data.mean().item()
        current_std = layer.weight.data.std().item()

        num_kernels = layer.weight.shape[0]
        num_reset = int(num_kernels * conv_reset_ratio)

        if num_reset > 0:
            reset_indices = random.sample(range(num_kernels), num_reset)
            
            for idx in reset_indices:
                if init_method == 'uniform':
                    new_kernel = torch.zeros_like(layer.weight[idx]).uniform_(
                        current_mean - current_std * scale_factor, 
                        current_mean + current_std * scale_factor
                    )
                elif init_method == 'kaiming_uniform':
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = np.sqrt(6.0 / fan_in) * scale_factor
                    new_kernel = torch.zeros_like(layer.weight[idx]).uniform_(-bound, bound)
                elif init_method == 'kaiming_normal':
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    std = np.sqrt(2.0 / fan_in) * scale_factor
                    new_kernel = torch.randn_like(layer.weight[idx]) * std
                elif init_method == 'ori_normal':
                    new_kernel = torch.randn_like(layer.weight[idx]) * current_std * scale_factor + current_mean
                else:
                    raise ValueError("Unsupported initialization method")
                
                layer.weight.data[idx] = new_kernel
            
            reset_count += 1


            
    return reset_count, stopped_count
