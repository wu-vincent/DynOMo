from src.datasets.datasets import (
    load_dataset_config,
    DavisDataset,
    PanopticSportsDataset,
    IphoneDataset,
    datautils
)
import pickle
import numpy as np
import os
import torch
from src.utils.camera_helpers import as_intrinsics_matrix
import numpy as np
import imageio
import glob
import json
from itertools import product
import cv2
from src.utils.gaussian_utils import normalize_points
import copy


def get_dataset(
    config_dict,
    basedir,
    sequence,
    depth_type=None,
    cam_type=None,
    factor=2,
    do_scale=False,
    **kwargs):
    if config_dict["dataset_name"].lower() in ["davis"]:
        return DavisDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["panoptic_sport"]:
        return PanopticSportsDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["iphone"]:
        return IphoneDataset(config_dict, basedir, sequence, depth_type, cam_type, factor, do_scale, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_data(config):
    dataset_config = config["data"]
    gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        relative_pose=True,
        device=config["primary_device"],
        basedir=dataset_config["basedir"],
        sequence=dataset_config["sequence"],
        every_x_frame=dataset_config["every_x_frame"],
        start=dataset_config["start"],
        end=dataset_config["end"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        load_embeddings=dataset_config["load_embeddings"],
        depth_type=dataset_config['depth_type'],
        cam_type=dataset_config['cam_type'] if 'cam_type' in dataset_config.keys() else None,
        embedding_dim=dataset_config["embedding_dim"],
        start_from_complete_pc=dataset_config["start_from_complete_pc"],
        novel_view_mode=dataset_config['novel_view_mode'],
        factor=2 if 'factor' not in dataset_config.keys() else dataset_config['factor'],
        do_scale=False if 'do_scale' not in dataset_config.keys() else dataset_config['do_scale'],
        online_emb=dataset_config['online_emb'],
        online_depth=dataset_config['online_depth']
        )

    return dataset


def get_cam_data(config, orig_image_size=False):
    config_dict = load_dataset_config(config['data']["gradslam_data_cfg"])

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


def load_davis_all(in_torch=False, gt_traj_data=''):
    with open(gt_traj_data, 'rb') as jf:
        gt = pickle.load(jf)
    if in_torch:
        gt = {seq: {k: torch.from_numpy(v) for k, v in data.items()} for seq, data in gt.items()}
    return gt


def load_davis(sequence, gt_traj_data='', in_torch=False):
    data = load_davis_all(in_torch, gt_traj_data)[sequence] # N x T x 2
    return data


def load_panoptic_sports_all(cam_id, in_torch=False, basedir=''):
    with open(os.path.join(basedir, 'annotations/traj_tap_vid_format_gs_{:04d}.pickle'.format(int(cam_id))), 'rb') as pf:
        gt = pickle.load(pf)
    if in_torch:
        gt = {seq: {k: torch.from_numpy(v) for k, v in data.items()} for seq, data in gt.items()}
    return gt


def load_panoptic_sports(sequence, in_torch=False, basedir=''):
    cam_id = os.path.basename(sequence)
    sequence = os.path.dirname(os.path.dirname(sequence))
    data = load_panoptic_sports_all(cam_id, in_torch, basedir)[sequence]
    return data


def load_iphone(config, in_torch=True, device="cuda:0"):
    config = copy.deepcopy(config)
    config['data']['factor'] = 1
    config['data']['online_depth'] = None
    config['data']['online_emb'] = None
    config['data']['cam_type'] = 'refined'
    config['data']['depth_type'] = 'aligned_depth_anything_colmap'
    config['primary_device'] = device
    dataset = get_data(config=config)
    color_paths = dataset.color_paths
    dataset.load_depth(dataset.depth_paths[0], 0)

    # get rgb
    video = [np.asarray(imageio.imread(path), dtype=float) for path in color_paths]
    video = torch.stack([torch.from_numpy(r) for r in video])[:, :, :, :-1]

    # get depth
    train_depths = np.array([np.load(p.replace('rgb', 'depth').replace('png', 'npy')) for p in dataset.color_paths])
    if dataset.do_scale:
        train_depths = train_depths / dataset.scale
    # train_depths = dataset.depths

    # get poses
    dataset.load_poses(cam=0)
    train_Ks, train_w2cs = dataset.intrinsics, dataset.w2cs

    # scale poses
    scale = np.load(os.path.join(dataset.input_folder, "flow3d_preprocessed/colmap/scale.npy")).item()
    train_c2ws = np.linalg.inv(train_w2cs)
    train_c2ws[:, :3, -1] *= scale
    train_w2cs = np.linalg.inv(train_c2ws)
    
    # get keypoints
    keypoint_paths = sorted(glob.glob(os.path.join(dataset.input_folder, "keypoint/2x/train/0_*.json")))
    keypoints_2d = []
    for keypoint_path in keypoint_paths:
        with open(keypoint_path) as f:
            keypoints_2d.append(json.load(f))
    keypoints_2d = np.array(keypoints_2d)
    keypoints_2d[..., :2] *= 2.0
    time_ids = np.array(
        [int(os.path.basename(p).split("_")[1].split(".")[0]) for p in keypoint_paths]
    )
    time_pairs = np.array(list(product(time_ids, repeat=2)))
    index_pairs = np.array(list(product(range(len(time_ids)), repeat=2)))
    keypoints_3d = []
    for i, kps_2d in zip(time_ids, keypoints_2d):
        K = train_Ks[i]
        w2c = train_w2cs[i]
        depth = train_depths[i]
        is_kp_visible = kps_2d[:, 2] == 1
        is_depth_valid = (
            cv2.remap(
                (depth != 0).astype(np.float32),
                kps_2d[None, :, :2].astype(np.float32),
                None,  # type: ignore
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )[0]
            == 1
        )
        kp_depths = cv2.remap(
            depth,  # type: ignore
            kps_2d[None, :, :2].astype(np.float32),
            None,  # type: ignore
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )[0]
        kps_3d = (
            np.einsum(
                "ij,pj->pi",
                np.linalg.inv(K),
                np.pad(kps_2d[:, :2], ((0, 0), (0, 1)), constant_values=1),
            )
            * kp_depths[:, None]
        )
        kps_3d = np.einsum(
            "ij,pj->pi",
            np.linalg.inv(w2c)[:3],
            np.pad(kps_3d, ((0, 0), (0, 1)), constant_values=1),
        )
        kps_3d = np.concatenate(
            [kps_3d, (is_kp_visible & is_depth_valid)[:, None]], axis=1
        )
        kps_3d[kps_3d[:, -1] != 1] = 0.0
        keypoints_3d.append(kps_3d)
    
    keypoints_3d = np.array(keypoints_3d)

    data_dict = {
        "video": video,
        "train_depths": train_depths,
        "train_Ks": train_Ks,
        "train_w2cs": train_w2cs,
        "points": keypoints_2d[:, :, :-1],
        "occluded": ~(keypoints_2d[:, :, -1].astype(np.bool_)),
        "trajs": keypoints_3d[:, :, :-1],
        "occluded_trajs": ~(keypoints_3d[:, :, -1].astype(np.bool_)),
        "time_ids": time_ids,
        "time_pairs": time_pairs,
        "index_pairs": index_pairs,
    }
    if in_torch:
        for k, v in data_dict.items():
            if not isinstance(v, torch.Tensor):
                data_dict[k] = torch.from_numpy(v)
    data_dict['points'] = data_dict['points'].permute(1, 0, 2)
    data_dict['points'] = normalize_points(data_dict['points'], h=video.shape[1], w=video.shape[2])
    data_dict['trajs'] = data_dict['trajs'].permute(1, 0, 2)
    data_dict['occluded'] = data_dict['occluded'].permute(1, 0)
    data_dict['occluded_trajs'] = data_dict['occluded_trajs'].permute(1, 0)

    return data_dict


def get_gt_traj(config, in_torch=False, device='cuda:0'):
    config_dict = load_dataset_config(config['data']["gradslam_data_cfg"])
    if config_dict["dataset_name"].lower() in ["davis"]:
        return load_davis(config["data"]["sequence"], config["data"]["gt_traj_data"], in_torch)
    elif config_dict["dataset_name"].lower() in ["panoptic_sport"]:
        return  load_panoptic_sports(config["data"]["sequence"], in_torch, config['data']['basedir'])
    elif config_dict["dataset_name"].lower() in ["iphone"]:
        return  load_iphone(config, in_torch, device)


def load_scene_data(config=None, results_dir=None, device="cuda:0", file=None):
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
