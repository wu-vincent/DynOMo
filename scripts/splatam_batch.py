import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import copy

from utils.get_data import get_data, just_get_start_pix, load_scene_data
from utils.common_utils import seed_everything, save_params_ckpt, save_params, load_params_ckpt
from utils.eval_helpers import report_loss, report_progress, eval, eval_during
from utils.camera_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2depthplussilhouette,
    transform_to_frame,
    l1_loss_v1,
    quat_mult,
    l2_loss_v2,
    get_hook,
    dyno_losses,
    get_renderings,
    get_hook_bg,
    matrix_to_quaternion,
    compute_visibility
)
from utils.forward_backward_field import MotionPredictor
from utils.attention import DotAttentionLayer
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify, normalize_quat, remove_points
from utils.neighbor_search import calculate_neighbors_seg, calculate_neighbors_between_pc, o3d_knn, torch_3d_knn, calculate_neighbors_seg_after_init
from utils.eval_traj import find_closest_to_start_pixels

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from torch_scatter import scatter_mean, scatter_add
import open3d as o3d
import imageio
import torchvision
from torchvision.transforms.functional import InterpolationMode
import json
import glob

# Make deterministic
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from utils.render_trajectories import just_render
from utils.eval_traj import eval_traj, vis_grid_trajs, eval_traj_window, vis_grid_trajs_window
from datasets.gradslam_datasets import datautils

from collections import OrderedDict
from typing import Dict, Callable

import gc

from torch.utils.data import DataLoader
from scripts.splatam import RBDG_SLAMMER


class RBDG_SLAMMER_WINDOW(RBDG_SLAMMER):
    def __init__(self, config):
        self.min_cam_loss = 0
        torch.set_num_threads(config["num_threads"])
        self.hook_list = list()
        self.support_trajs_trans = None
        self.device = config['primary_device']
        device = torch.device(self.device)
        torch.cuda.set_device(device)

        if config['data']['get_pc_jono']:
            config['data']['load_embeddings'] = False
            config['add_gaussians']['add_new_gaussians'] = False

        self.config = config
        self.config_orig = copy.deepcopy(config)
        # Print Config
        print("Loaded Config:")
        if "use_depth_loss_thres" not in config['tracking_obj']:
            config['tracking_obj']['use_depth_loss_thres'] = False
            config['tracking_obj']['depth_loss_thres'] = 100000
        if "visualize_tracking_loss" not in config['tracking_obj']:
            config['tracking_obj']['visualize_tracking_loss'] = False
        if "gaussian_distribution" not in config:
            config['gaussian_distribution'] = "isotropic"
        print(f"{config}")

        # Create Output Directories
        self.output_dir = os.path.join(config["workdir"], config["run_name"])
        self.eval_dir = os.path.join(self.output_dir, "eval")
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # Init WandB
        if config['use_wandb']:
            self.wandb_time_step = 0
            self.wandb_obj_tracking_step = 0
            self.wandb_init_next_step = 0
            self.wandb_run = wandb.init(project=config['wandb']['project'],
                                group=config['wandb']['group'],
                                name=config['wandb']['name'],
                                config=config)
        
        self.psnr_list = list()
        self.rmse_list = list()
        self.l1_list = list()
        self.ssim_list = list()
        self.lpips_list = list()
        self.eval_pca = None

    def rgbd_slam(self):
        self.motion_mlp = MotionPredictor(device=self.device) if self.config['motion_mlp'] else None
        # Initialize list to keep track of Keyframes
        keyframe_list = []
        keyframe_time_indices = []
        
        # Init Variables to keep track of ground truth poses and runtimes
        gt_w2c_all_frames = []
        self.tracking_obj_iter_time_sum = 0
        self.tracking_obj_iter_time_count = 0
        self.tracking_obj_frame_time_sum = 0
        self.tracking_obj_frame_time_count = 0
        self.tracking_cam_iter_time_sum = 0
        self.tracking_cam_iter_time_count = 0
        self.tracking_cam_frame_time_sum = 0
        self.tracking_cam_frame_time_count = 0
        self.supervision_flow = list()

        first_frame_w2c, _ = self.get_data()
        self.len_dataset = self.num_frames
        if len(sorted(glob.glob(os.path.join(self.output_dir, "params_*.npz")))) > 0 and self.config_orig['checkpoint']:
            start_frame = int(sorted(
                glob.glob(os.path.join(self.output_dir, "params_*.npz")))[-1].split('_')[-1].split('.')[0])
        else:
            for file in glob.glob(os.path.join(self.output_dir, "params_*.npz")):
                os.remove(file)
            start_frame = 0
        set_cam = True
        self.batch = 0
        while start_frame < self.len_dataset - 1:
            end_frame = min(start_frame + self.config['time_window'], self.len_dataset)
            self.config = copy.deepcopy(self.config_orig)
            self.config['data']['start'] = start_frame
            self.config['data']['end'] = end_frame
            self.config['data']['num_frames'] = self.config['time_window']

            first_frame_w2c, _ = self.get_data()
            if set_cam:
                self.first_frame_w2c = first_frame_w2c
                cam_data = {
                    'cam': self.cam,
                    'intrinsics': self.intrinsics, 
                    'w2c': self.first_frame_w2c
                    }
                set_cam = False
            self.variables['prev_means2d'], self.variables['prev_weight'], self.variables['prev_visible'] = None, None, None
                        
            # Iterate over Scan
            print(start_frame, len(self.dataset), self.num_frames, self.len_dataset, self.config['data']['end'])
            for time_idx in tqdm(range(0, self.num_frames)):
                curr_data, _ = self.make_data_dict(time_idx, gt_w2c_all_frames, cam_data)
                start = time.time()
                succ_cam_track = \
                    self.optimize_time(
                        time_idx,
                        curr_data)
                
                if not succ_cam_track:
                    break

                if time_idx == 0:
                    self.variables["scale_0"] = self.params["log_scales"]
                    self.variables["embeddings_0"] = self.params["embeddings"]
                
            # Add Camera Parameters to Save them
            self.update_params_for_saving(keyframe_time_indices)
            self.remove_unfilled(start_frame+time_idx)
            # Save Parameters
            save_params(self.params, self.output_dir, start_frame, start_frame+time_idx, keep_all=True)
            start_frame += time_idx
            print(start_frame)
            self.batch += 1

        self.log_time_stats()

        if self.config['eval_during']:
            self.log_eval_during()
        '''else:
            # Evaluate Final Parameters
            with torch.no_grad():
                eval(
                    self.dataset,
                    self.params,
                    self.num_frames,
                    self.eval_dir,
                    sil_thres=self.config['add_gaussians']['sil_thres_gaussians'],
                    wandb_run=self.wandb_run if self.config['use_wandb'] else None,
                    wandb_save_qual=self.config['wandb']['eval_save_qual'],
                    eval_every=self.config['eval_every'],
                    variables=self.variables,
                    mov_thresh=self.config['mov_thresh'],
                    save_pc=self.config['viz']['save_pc'],
                    save_videos=self.config['viz']['save_videos'],
                    get_embeddings=self.config['data']['load_embeddings'],
                    vis_gt=self.config['viz']['vis_gt'],
                    save_depth=self.config['viz']['vis_all'],
                    save_rendered_embeddings=self.config['viz']['vis_all'],)'''
        
        # eval traj
        with torch.no_grad():
            metrics = eval_traj_window(
                self.config_orig,
                cam=self.cam,
                results_dir=self.eval_dir, 
                vis_trajs=self.config['viz']['vis_tracked'])
            with open(os.path.join(self.eval_dir, 'traj_metrics.txt'), 'w') as f:
                f.write(f"Trajectory metrics: {metrics}")
        print("Trajectory metrics: ",  metrics)

        # Close WandB Run
        if self.config['use_wandb']:
            self.wandb_run.log(metrics)
            wandb.finish()
        
        if self.config['viz']['vis_grid']:
            vis_grid_trajs_window(
                self.config_orig,
                results_dir=self.eval_dir,
                orig_image_size=True)
        
        if self.config['save_checkpoints']:
            ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
            if os.path.isfile(os.path.join(ckpt_output_dir, "temp_params.npz")):
                os.remove(os.path.join(ckpt_output_dir, "temp_params.npz"))
                os.remove(os.path.join(ckpt_output_dir, "temp_variables.npz"))

    def remove_unfilled(self, end_frame):
        for k, v in self.params.items():
            if isinstance(v, torch.Tensor):
                if v.shape[-1] == self.len_dataset:
                    print(self.params[k].shape, end_frame)
                    shape_prev = self.params[k].shape
                    self.params[k] = v[..., :end_frame+1]
                    if len(self.params[k].shape) < len(shape_prev):
                        self.params[k] = self.params[k].unsqueeze(-1)
                    print(self.params[k].shape)
                    print()

    def optimize_time(
            self,
            time_idx,
            curr_data):

        # track cam
        if self.config['tracking_cam']['num_iters'] != 0 and time_idx > 0:
            self.track_cam(
                time_idx,
                curr_data)

            if self.min_cam_loss > 1.0 and self.config['time_window'] > self.len_dataset:
                return False
            
        # Densification
        if (time_idx+1) % self.config['add_every'] == 0 and time_idx > 0:
            self.densify(time_idx, curr_data)

        if self.config['tracking_obj']['num_iters'] != 0:
            optimizer = self.track_objects(
                time_idx,
                curr_data)
        
        if (time_idx < self.num_frames-1) and self.config['init_next']['num_iters'] != 0 and (self.config['mov_init_by'] == 'sparse_flow' or self.config['mov_init_by'] == 'im_loss' or self.config['mov_init_by'] == 'sparse_flow_simple' or self.config['mov_init_by'] == 'rendered_flow') :
            with torch.no_grad():
                self.params['means3D'][:, :, time_idx+1] = self.params['means3D'][:, :, time_idx]
                self.params['unnorm_rotations'][:, :, time_idx+1] = self.params['unnorm_rotations'][:, :, time_idx]
            _ = self.track_objects(
                time_idx+1,
                curr_data,
                init_next=True)
        
        # remove floating points based on depth
        if self.config['remove_gaussians']['remove']:
            self.remove_gaussians_with_depth(curr_data, time_idx, optimizer)
        
        # Increment WandB Time Step
        if self.config['use_wandb']:
            self.wandb_time_step += 1

        torch.cuda.empty_cache()
        gc.collect()

        if (time_idx < self.num_frames-1) and \
                (self.config['neighbors_init'] == 'post' or \
                    ('first_post' in self.config['neighbors_init'] and time_idx == 0)):
            with torch.no_grad():
                self.variables, to_remove = calculate_neighbors_seg_after_init(
                    self.params,
                    self.variables,
                    time_idx,
                    num_knn=int(20/self.config['stride']),
                    dist_to_use=self.config['dist_to_use'],
                    primary_device=self.device,
                    exp_weight=self.config['exp_weight'])

        if self.config['deactivate_gaussians']['drift']:
            self.deactivate_gaussians_drift(curr_data, time_idx, optimizer)

        if (time_idx < self.num_frames-1):
            # Initialize Gaussian poses for the next frame in params
            self.params, self.variables = self.forward_propagate_gaussians(
                time_idx,
                self.params,
                self.variables,
                self.config['mov_init_by'],
                self.config['mov_static_init'],
                self.config['determine_mov'],
                support_trajs_trans=self.support_trajs_trans,
                make_bg_static=self.config['make_bg_static'])
            
            self.forward_propagate_camera(self.params, time_idx, forward_prop=self.config['tracking_cam']['forward_prop'])
            
        if self.config['re_init_scale'] and (time_idx < self.num_frames-1):
            self.re_initialize_scale(time_idx)

        for k, p in self.params.items():
            p._backward_hooks: Dict[int, Callable] = OrderedDict()

        return True

    def densify(self, time_idx, curr_data):
        if self.config['add_gaussians']['add_new_gaussians']:
            # Add new Gaussians to the scene based on the Silhouette
            pre_num_pts = self.params['means3D'].shape[0]
            self.add_new_gaussians(curr_data, 
                                    self.config['add_gaussians']['sil_thres_gaussians'],
                                    self.config['add_gaussians']['depth_error_factor'],
                                    time_idx,
                                    self.config['mean_sq_dist_method'],
                                    self.config['gaussian_distribution'],
                                    self.params, 
                                    self.variables)

            post_num_pts = self.params['means3D'].shape[0]
            if self.config['use_wandb']:
                self.wandb_run.log({"Adding/Number of Gaussians": post_num_pts-pre_num_pts,
                                "Adding/step": self.wandb_time_step})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--just_eval", default=0, type=int, help="if only eval")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()
    experiment.config['just_eval'] = args.just_eval

    if experiment.config['just_eval']:
        experiment.config['use_wandb'] = False
        experiment.config['checkpoint'] = True

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )

    rgbd_slammer = RBDG_SLAMMER_WINDOW(experiment.config)
    if experiment.config['just_eval']:
        rgbd_slammer.eval()
    else:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))
        rgbd_slammer.rgbd_slam()


