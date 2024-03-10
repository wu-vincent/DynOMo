
import torch
import numpy as np
import open3d as o3d
import time
from copy import deepcopy
import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader
import cv2
import random

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)
from utils.recon_helpers import setup_camera
from utils.colormap import colormap
try:
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer
except:
    print('Rasterizer not installed locally, will use precomputed img and depth.')
from utils.common_utils import seed_everything
from datasets.gradslam_datasets import datautils, load_dataset_config
from utils.slam_helpers import (
    transformed_params2rendervar,
    transformed_params2depthplussilhouette,
    quat_mult,
    mask_timestamp,
    build_rotation,
    transform_to_frame,
    )
from utils.slam_external import normalize_quat


RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'depth'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'centers'  # 'color', 'depth' or 'centers'

# ADDITIONAL_LINES = None  # None, 'trajectories' or 'rotations'
ADDITIONAL_LINES = 'trajectories'  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'rotations'  # None, 'trajectories' or 'rotations'

FORCE_LOOP = False  # False or True
# FORCE_LOOP = True  # False or True

w, h = 640, 480
near, far = 0.01, 100.0
view_scale = 1.0
fps = 1
traj_frac = 10 # 4% of points
traj_length = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).to(device).float()
pix_ones = torch.ones(h * w, 1).to(device).float()


def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k


def get_extrinsics(y_angle=0., center_dist=2.4, cam_height=1.3):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    return w2c


def get_intrinsics(camera_params, dataset_params):
    distortion = (
            np.array(camera_params["distortion"])
            if "distortion" in camera_params
            else None
        )
    K = as_intrinsics_matrix(
        [camera_params['fx'], camera_params['fy'], camera_params['cx'], camera_params['cy']])
    
    if distortion is not None:
        # undistortion is only applied on color image, not depth!
        color = cv2.undistort(color, K, distortion)

    color = torch.from_numpy(color)
    K = torch.from_numpy(K)
    depth = _preprocess_depth(
        depth,
        dataset_params['desired_width'],
        dataset_params['desired_height'],
        dataset_params['png_depth_scale'])
    depth = torch.from_numpy(depth)
    height_downsample_ratio = float(dataset_params['desired_height']) / camera_params['image_height']
    width_downsample_ratio = float(dataset_params['desired_width']) / camera_params['image_width']
    K = datautils.scale_intrinsics(K, height_downsample_ratio, width_downsample_ratio)
    intrinsics = torch.eye(4).to(K)
    intrinsics[:3, :3] = K


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def _preprocess_depth(depth: np.ndarray, desired_width, desired_height, png_depth_scale, channels_first=False):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (desired_width, desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        if channels_first:
            depth = datautils.channels_first(depth)
        return depth / png_depth_scale


def load_scene_data(config, results_dir):
    params = dict(np.load(f"{results_dir}/params.npz"))
    params = {k: torch.tensor(v).to(device).float() for k, v in params.items()}
    return params, params['moving'], params['timestep'], params['intrinsics'], params['w2c']


def make_lineset(all_pts, cols, num_lines):
    linesets = []
    for pts in all_pts:
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def calculate_trajectories(params, is_mov):
    idxs = random.sample(list(range(params['means3D'][is_mov.bool()].shape[0])), traj_frac)
    in_pts = params['means3D'][is_mov.bool()][idxs, :, :].contiguous().float().cpu().numpy()
    num_lines = traj_frac
    cols = np.repeat(colormap[np.arange(num_lines) % len(colormap)][None], traj_length, 0).reshape(-1, 3)
    out_pts = []
    for t in range(in_pts.shape[2])[traj_length:]:
        out_pts.append(np.array(in_pts[:, :, t - traj_length:t + 1]).transpose(2, 0, 1).reshape(-1, 3))
    
    return make_lineset(out_pts, cols, num_lines)


def calculate_rot_vec(scene_data, is_mov):
    in_pts = params['means3D'][is_mov.bool()][:traj_frac, :, :].contiguous().float().cpu().numpy()
    in_rotation = params['unnorm_rotations'][is_mov.bool()]
    for t in range(in_rotation.shape[2]):
        in_rotation[:, :, t] = normalize_quat(in_rotation[:, :, t])
    in_rotation = params['unnorm_rotations'][is_mov.bool()][:traj_frac, :, :].contiguous().float().cpu().numpy()
    num_lines = traj_frac
    cols = colormap[np.arange(num_lines) % len(colormap)]
    inv_init_q = deepcopy(in_rotation[:, : 0])
    inv_init_q[:, 1:] = -1 * inv_init_q[:, 1:]
    inv_init_q = inv_init_q / (inv_init_q ** 2).sum(-1)[:, None]
    init_vec = np.array([-0.1, 0, 0])
    out_pts = []
    for t in range(in_pts.shape[2]):
        cam_rel_qs = quat_mult(in_rotation[t], inv_init_q)
        rot = build_rotation(cam_rel_qs).cpu().numpy()
        vec = (rot @ init_vec[None, :, None]).squeeze()
        out_pts.append(np.concatenate((in_pts[:, :, t] + vec, in_pts[:, :, t]), 0))
    return make_lineset(out_pts, cols, num_lines)


def render(w2c, k, params, moving_mask, first_occurance, iter_time_idx):
    cam = setup_camera(w, h, k, w2c, near, far)
    transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                                gaussians_grad=False,
                                                camera_grad=False)
    
    rendervar = transformed_params2rendervar(params, transformed_gaussians, iter_time_idx)
    rendervar, _ = mask_timestamp(rendervar, iter_time_idx, first_occurance, moving_mask)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, w2c,
                                                            transformed_gaussians, iter_time_idx)
    depth_sil_rendervar, _ = mask_timestamp(depth_sil_rendervar, iter_time_idx, first_occurance, moving_mask)
    with torch.no_grad():
        # image
        im, _, _, = Renderer(raster_settings=cam)(**rendervar)
        # Depth & Silhouette Rendering
        depth_sil, _, _, = Renderer(raster_settings=cam)(**depth_sil_rendervar)
        return im, depth_sil[0, :, :].unsqueeze(0)


def rgbd2pcd(im, depth, w2c, k, show_depth=False, project_to_cam_w_scale=None):
    d_near = 1.5
    d_far = 6
    invk = torch.inverse(torch.tensor(k).to(device).float())
    c2w = torch.inverse(torch.tensor(w2c).to(device).float())
    radial_depth = depth[0].reshape(-1)
    def_rays = (invk @ def_pix.T).T
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    pts_cam = def_radial_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]
    if project_to_cam_w_scale is not None:
        pts_cam = project_to_cam_w_scale * pts_cam / z_depth[:, None]
    pts4 = torch.concat((pts_cam, pix_ones), 1)
    pts = (c2w @ pts4.T).T[:, :3]
    if show_depth:
        cols = ((z_depth - d_near) / (d_far - d_near))[:, None].repeat(1, 3)
    else:
        cols = torch.permute(im, (1, 2, 0)).reshape(-1, 3)
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols

def visualize(config, results_dir):
    print(config.keys())
    params, moving_mask, first_occurance, k, w2c = load_scene_data(config, results_dir)
    dataset_cfg = load_dataset_config(config["data"]["gradslam_data_cfg"])
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=int(dataset_cfg['camera_params']['image_width'] * view_scale),
        height=int(dataset_cfg['camera_params']['image_height'] * view_scale),
        visible=True)

    if os.path.isfile(f"{results_dir}/rendered_for_traj_vis/0_im.npz"):
        print(np.load(f"{results_dir}/rendered_for_traj_vis/0_im.npz"))
        im = torch.from_numpy(np.load(f"{results_dir}/rendered_for_traj_vis/0_im.npz")['arr_0']).to(device)
        depth = torch.from_numpy(np.load(f"{results_dir}/rendered_for_traj_vis/0_depth.npz")['arr_0']).to(device)
    else:
        im, depth = render(w2c, k, params, moving_mask, first_occurance, iter_time_idx=0)
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, show_depth=(RENDER_MODE == 'depth'))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    linesets = None
    lines = None
    if ADDITIONAL_LINES is not None:
        if ADDITIONAL_LINES == 'trajectories':
            linesets = calculate_trajectories(params, moving_mask)
        elif ADDITIONAL_LINES == 'rotations':
            linesets = calculate_rot_vec(params, moving_mask)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    view_k = k * view_scale
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    cparams.extrinsic = w2c.cpu().numpy()
    cparams.intrinsic.intrinsic_matrix = view_k.cpu().numpy()
    cparams.intrinsic.height = int(dataset_cfg['camera_params']['image_height'] * view_scale)
    cparams.intrinsic.width = int(dataset_cfg['camera_params']['image_width'] * view_scale)
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = view_scale
    render_options.line_width = 50
    render_options.light_on = False

    start_time = time.time()
    num_timesteps = params['means3D'].shape[2]
    while True:
        passed_time = time.time() - start_time
        passed_frames = passed_time * fps
        if ADDITIONAL_LINES == 'trajectories':
            iter_time_idx = int(passed_frames % (num_timesteps - traj_length)) + traj_length  # Skip t that don't have full traj.
        else:
            iter_time_idx = int(passed_frames % num_timesteps)

        if FORCE_LOOP:
            num_loops = 1.4
            y_angle = 360*iter_time_idx*num_loops / num_timesteps
            w2c = get_extrinsics(y_angle)
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            cam_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        else:  # Interactive control
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / view_scale
            k[2, 2] = 1
            w2c = cam_params.extrinsic

        if RENDER_MODE == 'centers':
            pts = o3d.utility.Vector3dVector(params['means3D'][:, :, iter_time_idx].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(params['rgb_colors'].contiguous().double().cpu().numpy())
        else:
            if os.path.isfile(f"{results_dir}/rendered_for_traj_vis/{iter_time_idx}_im.npz"):
                im = torch.from_numpy(np.load(f"{results_dir}/rendered_for_traj_vis/{iter_time_idx}_im.npz")['arr_0']).to(device)
                depth = torch.from_numpy(np.load(f"{results_dir}/rendered_for_traj_vis/{iter_time_idx}_depth.npz")['arr_0']).to(device)
            else:
                im, depth = render(w2c, k, params, moving_mask, first_occurance, iter_time_idx)
            pts, cols = rgbd2pcd(im, depth, w2c, k, show_depth=(RENDER_MODE == 'depth'))
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if ADDITIONAL_LINES is not None:
            if ADDITIONAL_LINES == 'trajectories':
                lt = iter_time_idx - traj_length
            else:
                lt = iter_time_idx
            lines.points = linesets[lt].points
            lines.colors = linesets[lt].colors
            lines.lines = linesets[lt].lines
            vis.update_geometry(lines)

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()
    del view_control
    del vis
    del render_options


def just_render(config, results_dir):
    os.makedirs(f"{results_dir}/rendered_for_traj_vis", exist_ok=True)
    params, moving_mask, first_occurance, k, w2c = load_scene_data(config, results_dir)

    num_timesteps = params['means3D'].shape[2]
    for iter_time_idx in range(num_timesteps):
        im, depth = render(w2c, k, params, moving_mask.bool(), first_occurance, iter_time_idx)
        np.savez(f"{results_dir}/rendered_for_traj_vis/{iter_time_idx}_im.npz", im.cpu().numpy())
        np.savez(f"{results_dir}/rendered_for_traj_vis/{iter_time_idx}_depth.npz", depth.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    try:
        visualize(experiment.config, results_dir)
    except:
        just_render(experiment.config, results_dir)
