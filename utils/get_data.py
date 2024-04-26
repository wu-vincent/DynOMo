from datasets.gradslam_datasets import (
    load_dataset_config,
    ReplicaDataset,
    DynoSplatamDataset,
    SyntheticDynoSplatamDataset,
    PointOdysseeDynoSplatamDataset,
    DavisDynoSplatamDataset,
    JonoDynoSplatamDataset,
    datautils
)
import pickle
import numpy as np
import os
import torch
from utils.camera_helpers import as_intrinsics_matrix
import numpy as np


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["dynosplatam"]:
        return DynoSplatamDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["synthetic"]:
        return SyntheticDynoSplatamDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["pointodyssee"]:
        return PointOdysseeDynoSplatamDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["davis"]:
        return DavisDynoSplatamDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["jono_data"]:
        return JonoDynoSplatamDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")

def get_gradslam_data_cfg(dataset_config):
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    
    return gradslam_data_cfg


def get_data(config):
    dataset_config = config["data"]
    gradslam_data_cfg = get_gradslam_data_cfg(dataset_config)

    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=dataset_config["sequence"],
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=config["primary_device"],
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
        load_embeddings=dataset_config["load_embeddings"])

    return dataset


def get_cam_data(config, orig_image_size=False):
    config_dict = get_gradslam_data_cfg(config["data"])

    if orig_image_size:
        desired_image_height = config_dict["camera_params"]["image_height"]
        desired_image_width = config_dict["camera_params"]["image_width"]
    else:
        desired_image_height = config['data']['desired_image_height']
        desired_image_width = config['data']['desired_image_width']

    orig_height = config_dict["camera_params"]["image_height"]
    orig_width = config_dict["camera_params"]["image_width"]
    fx = config_dict["camera_params"]["fx"]
    fy = config_dict["camera_params"]["fy"]
    cx = config_dict["camera_params"]["cx"]
    cy = config_dict["camera_params"]["cy"]

    height_downsample_ratio = float(desired_image_height) / orig_height
    width_downsample_ratio = float(desired_image_width) / orig_width

    K = as_intrinsics_matrix([fx, fy, cx, cy])
    K = torch.from_numpy(K)
    K = datautils.scale_intrinsics(K, height_downsample_ratio, width_downsample_ratio)
    intrinsics = torch.eye(4).to(K)
    intrinsics[:3, :3] = K

    pose = torch.eye(4).float()

    return intrinsics, pose, desired_image_height, desired_image_width


def load_davis_all(in_torch=False):
    with open('data/tapvid_davis/tapvid_davis.pkl', 'rb') as jf:
        gt = pickle.load(jf)
    if in_torch:
        gt = {seq: {k: torch.from_numpy(v) for k, v in data.items()} for seq, data in gt.items()}
    return gt

def load_davis(sequence, in_torch=False):
    data = load_davis_all(in_torch)[sequence] # N x T x 2
    return data


def load_jono_all(cam_id, in_torch=False):
    path = 'data/data/annotations/traj_tap_vid_format_gs_{:04d}.pickle'.format(int(cam_id))
    with open(path, 'rb') as pf:
        gt = pickle.load(pf)
    if in_torch:
        gt = {seq: {k: torch.from_numpy(v) for k, v in data.items()} for seq, data in gt.items()}
    return gt


def load_jono(sequence, in_torch=False):
    cam_id = os.path.basename(sequence)
    sequence = os.path.dirname(os.path.dirname(sequence))
    data = load_jono_all(cam_id, in_torch)[sequence]
    return data


def get_gt_traj(config, in_torch=False):
    config_dict = get_gradslam_data_cfg(config["data"])
    if config_dict["dataset_name"].lower() in ["davis"]:
        return load_davis(config["data"]["sequence"], in_torch)
    elif config_dict["dataset_name"].lower() in ["jono_data"]:
        return  load_jono(config["data"]["sequence"], in_torch)


def load_scene_data(config, results_dir, device="cuda:0"):
    params = dict(np.load(f"{results_dir}/params.npz"))
    params = {k: torch.tensor(v).to(device).float() for k, v in params.items()}
    return params, params['moving'], params['timestep'], params['intrinsics'], params['w2c']


def just_get_start_pix(config, in_torch=True, normalized=False, h=None, w=None, rounded=True):
    data = get_gt_traj(config, in_torch)
    if data['points'].sum() == 0:
        start_pix = data['points_projected'][:, 0, :]
    else:
        start_pix = data['points'][:, 0, :]
        # visible = data['occluded'][:, 0] == 0
        # start_pix = start_pix[visible]

    if not normalized:
        if w is None:
            print("please give h and w if start pixels should be unnormalized!!")
            quit()
        start_pix[:, 0] = (start_pix[:, 0] * w) - 1
        start_pix[:, 1] = (start_pix[:, 1] * h) - 1
        start_pix = torch.round(start_pix).long()
    return start_pix