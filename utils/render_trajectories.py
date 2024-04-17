
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
import matplotlib.pyplot as plt

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
from utils.get_data import load_scene_data


RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'depth'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'centers'  # 'color', 'depth' or 'centers'

# ADDITIONAL_LINES = None  # None, 'trajectories' or 'rotations'
ADDITIONAL_LINES = 'trajectories'  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'rotations'  # None, 'trajectories' or 'rotations'

FORCE_LOOP = False  # False or True
# FORCE_LOOP = True  # False or True

near, far = 0.01, 100.0
view_scale = 1.0
fps = 1
traj_frac = 10 # 4% of points
traj_length = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def render(w2c, k, params, moving_mask, first_occurance, iter_time_idx, h, w):
    cam = setup_camera(w, h, k, w2c, near, far)
    transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                                gaussians_grad=False,
                                                camera_grad=False)
    
    rendervar, _ = transformed_params2rendervar(
        params, transformed_gaussians, iter_time_idx, first_occurance)
    depth_sil_rendervar, _ = transformed_params2depthplussilhouette(
        params, w2c, transformed_gaussians, iter_time_idx, first_occurance)
    with torch.no_grad():
        # transform 3D means to 2D (from Gaussian rasterizer code)
        points_xy = cam.projmatrix.squeeze().T.matmul(torch.cat(
            [rendervar['means3D'], torch.ones(rendervar['means3D'].shape[0], 1).cuda()], dim=1).T)
        points_xy = points_xy / points_xy[3, :]
        points_xy = points_xy[:2].T
        points_xy[:, 0] = ((points_xy[:, 0]+1)*w-1) * 0.5
        points_xy[:, 1] = ((points_xy[:, 1]+1)*h-1) * 0.5

        # render image
        im, _, _, _, _ = Renderer(raster_settings=cam)(**rendervar)

        # Depth & Silhouette Rendering
        depth_sil, _, _, _, _ = Renderer(raster_settings=cam)(**depth_sil_rendervar)
        return im, depth_sil[0, :, :].unsqueeze(0), points_xy


def just_render(config, results_dir, len_traj=20):
    h, w = config["data"]["desired_image_height"], config["data"]["desired_image_width"]
    os.makedirs(f"{results_dir}/rendered_for_traj_vis", exist_ok=True)
    params, moving_mask, first_occurance, k, w2c = load_scene_data(config, results_dir)
    moving_mask = moving_mask.bool()
    moving_mask = None
    num_timesteps = params['means3D'].shape[2]
    num_points_t0 = params['means3D'][params['timestep']==0].shape[0]
    trajectories_to_plot_idx = random.sample(list(range(num_points_t0)), 200) #list(range(0, num_points_t0, int(num_points_t0/200))) 
    points_xy_list = list()
    rendered_imgs = list()
    imgs = list()
    for iter_time_idx in range(num_timesteps):
        im, depth, points_xy = render(w2c, k, params, moving_mask, first_occurance, iter_time_idx, h, w)
        rendered_imgs.append(im)
        points_xy_list.append(points_xy[:num_points_t0])

    points_xy_list = len_traj * [points_xy_list[0]] + points_xy_list
    points_xy_list = torch.stack(points_xy_list, dim=0)
    sampled_points_xy_list = points_xy_list[:, trajectories_to_plot_idx].cpu().numpy()

    os.makedirs(os.path.join(results_dir, 'eval', 'traj_vis'), exist_ok=True)
    colormap = plt.get_cmap('hsv')
    for time, im in enumerate(rendered_imgs):

        viz_gt_im = (im*255).detach().cpu().long().permute(1, 2, 0).numpy().astype(np.uint8)

        fig, ax = plt.subplots()
        ax.imshow(viz_gt_im)
        from matplotlib.collections import LineCollection
        for i in range(sampled_points_xy_list.shape[1]):
            x = np.clip(sampled_points_xy_list[time:time+len_traj, i, 0], a_min=0, a_max=w-1)
            y = np.clip(sampled_points_xy_list[time:time+len_traj, i, 1], a_min=0, a_max=h-1)
            color_len = np.arange(len_traj-1)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a continuous norm to map from data points to colors
            norm = plt.Normalize(color_len.min(), color_len.max())
            lc = LineCollection(segments, cmap='hsv', norm=norm)
            # Set the values used for colormapping
            lc.set_array(color_len)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            plt.scatter(x[-1], y[-1], c='white', marker='D', edgecolors='black', s=2, linewidths=0.5)
            # fig.colorbar(line, ax=ax)

            # plt.plot(x, y, color='orange') #, marker="D")

        # Add the patch to the Axes
        plt.axis('off')
        plt.savefig(os.path.join(results_dir, 'eval', 'traj_vis', "gs_{:04d}.png".format(time)), bbox_inches='tight', pad_inches = 0)
