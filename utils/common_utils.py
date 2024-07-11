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


def save_params(output_params, output_dir, time_idx=0, end_frame=0, keep_all=False):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    if keep_all:
        save_path = os.path.join(output_dir, "params_{:04d}_{:04d}.npz".format(time_idx, end_frame))
    else:
        save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **to_save)


def save_params_ckpt(output_params, output_variables, output_dir, time_idx, keep_all=False):
    output_variables['last_time_idx'] = torch.tensor(time_idx)
    for name, param in zip(["params", "variables"], [output_params, output_variables]):
        # Convert to CPU Numpy Arrays
        to_save = params2cpu(param)
        # Save the Parameters containing the Gaussian Trajectories
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving parameters to: {output_dir}")
        if keep_all:
            save_path = os.path.join(output_dir, f"temp_{name}_{time_idx}.npz")
        else:
            save_path = os.path.join(output_dir, f"temp_{name}.npz")
        np.savez(save_path, **to_save)


def load_params_ckpt(output_dir, device):
    loaded_params = list()
    for name in ["params", "variables"]:
        print(f"Loading parameters from: {output_dir}")
        save_path = os.path.join(output_dir, f"temp_{name}.npz")
        params = np.load(save_path, allow_pickle=True)
        if name == 'params':
            params = {k: torch.nn.Parameter(torch.from_numpy(v).to(device).float().contiguous().requires_grad_(True)) for k, v in params.items()}
        else:
            _params = dict()
            for k, v in params.items():
                if (v != np.array(None)).all():
                    _params[k] = torch.from_numpy(v).to(device)
                else:
                    _params[k] = v
            params = _params
            # params = {k: torch.from_numpy(v).to(device) for k, v in params.items() if v is not None}
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