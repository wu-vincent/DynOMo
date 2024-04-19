import os

import numpy as np
import random
import torch


def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.
        
        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")


def params2cpu(params):
    res = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().contiguous().numpy()
        else:
            res[k] = v
    return res


def save_params(output_params, output_dir):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **to_save)


def save_params_ckpt(output_params, output_variables, output_dir):
    for name, param in zip(["params", "variables"], [output_params, output_variables]):
        # Convert to CPU Numpy Arrays
        to_save = params2cpu(param)
        # Save the Parameters containing the Gaussian Trajectories
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving parameters to: {output_dir}")
        save_path = os.path.join(output_dir, f"temp_{name}.npz")
        np.savez(save_path, **to_save)


def load_params_ckpt(output_dir):
    loaded_params = list()
    for name in ["params", "variables"]:
        print(f"Loading parameters from: {output_dir}")
        save_path = os.path.join(output_dir, f"temp_{name}.npz")
        params = np.load(save_path)
        if name == 'params':
            params = {k: torch.nn.Parameter(torch.from_numpy(v).cuda().float().contiguous().requires_grad_(True)) for k, v in params.items()}
        else:
            params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
        loaded_params.append(params)
    return loaded_params


def save_seq_params(all_params, output_dir):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **params_to_save)


def save_seq_params_ckpt(all_params, output_dir,time_idx):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **params_to_save)