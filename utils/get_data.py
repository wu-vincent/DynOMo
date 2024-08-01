from datasets.gradslam_datasets import (
    load_dataset_config,
    DavisDynoSplatamDataset,
    JonoDynoSplatamDataset,
    RGBDynoSplatamDataset,
    IphoneDynoSplatamDataset,
    datautils
)
import pickle
import numpy as np
import os
import torch
from utils.camera_helpers import as_intrinsics_matrix
import numpy as np
import imageio



def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["davis"]:
        return DavisDynoSplatamDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["jono_data"]:
        return JonoDynoSplatamDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["rgb_stacking"]:
        return RGBDynoSplatamDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["iphone"]:
        return IphoneDynoSplatamDataset(config_dict, basedir, sequence, **kwargs)
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
        load_embeddings=dataset_config["load_embeddings"],
        embedding_dim=dataset_config["embedding_dim"],
        get_pc_jono=dataset_config["get_pc_jono"],
        jono_depth=dataset_config["jono_depth"],
        feats_224=dataset_config['feats_224'],
        do_transform=dataset_config['do_transform'] if 'do_transform' in dataset_config.keys() else False,
        novel_view_mode=dataset_config['novel_view_mode'])

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
    with open('/scratch/jseidens/data/tapvid_davis/tapvid_davis.pkl', 'rb') as jf:
        gt = pickle.load(jf)
    if in_torch:
        gt = {seq: {k: torch.from_numpy(v) for k, v in data.items()} for seq, data in gt.items()}
    return gt

def load_davis(sequence, in_torch=False):
    data = load_davis_all(in_torch)[sequence] # N x T x 2
    return data


def load_jono_all(cam_id, in_torch=False):
    path = '/scratch/jseidens/data/data/annotations/traj_tap_vid_format_gs_{:04d}.pickle'.format(int(cam_id))
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


def load_rgb_all(in_torch=False):
    with open('/data3/jseidens/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl', 'rb') as jf:
        gt = pickle.load(jf)
    if in_torch:
        gt = [{k: torch.from_numpy(v) for k, v in data.items()} for data in gt]
    return gt

def load_rgb(sequence, in_torch=False):
    data = load_rgb_all(in_torch)[int(sequence)] # N x T x 2
    return data


def load_iphone(config, in_torch=True):
    dataset = get_data(config=config)
    color_paths = dataset.color_paths
    rgbs = list()
    for path in color_paths:
        rgbs.append(np.asarray(imageio.imread(path), dtype=float))
    data = dict()
    data['video'] = torch.stack([torch.from_numpy(r) for r in rgbs])
    return data

def get_gt_traj(config, in_torch=False):
    config_dict = get_gradslam_data_cfg(config["data"])
    if config_dict["dataset_name"].lower() in ["davis"]:
        return load_davis(config["data"]["sequence"], in_torch)
    elif config_dict["dataset_name"].lower() in ["jono_data"]:
        return  load_jono(config["data"]["sequence"], in_torch)
    elif config_dict["dataset_name"].lower() in ["rgb_stacking"]:
        return  load_rgb(config["data"]["sequence"], in_torch)
    elif config_dict["dataset_name"].lower() in ["iphone"]:
        return  load_iphone(config, in_torch)


def load_scene_data(config, results_dir, device="cuda:0", file=None):
    if file is None:
        params = dict(np.load(f"{results_dir}/params.npz", allow_pickle=True))
    else:
        params = dict(np.load(file, allow_pickle=True))
    _params = dict()
    for k, v in params.items():
        if (v != np.array(None)).all():
            _params[k] = torch.tensor(v).to(device).float()
        else:
            _params[k] = None
    params = _params
    if "timestep" in params.keys():
        return params, params['timestep'], params['intrinsics'], params['w2c']
    else:
        params['means3D'] = params['means3D'].permute(1, 2, 0)[:, :, 2:]
        params['unnorm_rotations'] = params['unnorm_rotations'].permute(1, 2, 0)[:, :, 2:]
        params['bg'] = params['seg_colors'][:, 2]
        params['rgb_colors'] = params['rgb_colors'].permute(1, 2, 0)[:, :, 0]
        params['timestep'] = torch.zeros(params['means3D'].shape[0])
        cam_rots = np.tile([1, 0, 0, 0], (1, 1))
        cam_rots = torch.from_numpy(np.tile(cam_rots[:, :, None], (1, 1, params['means3D'].shape[2])))
        params['cam_unnorm_rots'] = cam_rots.to(params['means3D'].device).float()
        params['cam_trans'] = torch.from_numpy(np.zeros((1, 3, params['means3D'].shape[2]))).to(params['means3D'].device).float()

        return params, None, None, None

def just_get_start_pix(config, in_torch=True, normalized=False, h=None, w=None, rounded=True):
    data = get_gt_traj(config, in_torch)
    data['occluded'] = data['occluded'].bool()
    data['points'] = data['points'][~data['occluded'][:, 0]]
    data['occluded'] = data['occluded'][~data['occluded'][:, 0]]
    
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
        if 'jono' in config['data']["gradslam_data_cfg"]:
            start_pix[:, 0] = (start_pix[:, 0] * w)
            start_pix[:, 1] = (start_pix[:, 1] * h)
        else:
            start_pix[:, 0] = (start_pix[:, 0] * w) - 1
            start_pix[:, 1] = (start_pix[:, 1] * h) - 1

        if rounded:
            start_pix = torch.round(start_pix).long()

    return start_pix