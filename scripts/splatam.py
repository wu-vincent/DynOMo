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
from utils.eval_helpers import report_loss, eval, eval_during, make_vid
from utils.camera_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2depthplussilhouette,
    transformed_params2rendervar,
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
from utils.forward_backward_field import MotionPredictor, BaseTransformations, TransformationMLP
from utils.attention import DotAttentionLayer
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify, normalize_quat, remove_points
from utils.neighbor_search import calculate_neighbors_seg, torch_3d_knn, calculate_neighbors_seg_after_init
from utils.eval_traj import find_closest_to_start_pixels

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from torch_scatter import scatter_mean, scatter_add
import open3d as o3d
import imageio
import torchvision
from torchvision.transforms.functional import InterpolationMode
import json

# Make deterministic
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from utils.eval_traj import eval_traj, vis_grid_trajs
from datasets.gradslam_datasets import datautils

from collections import OrderedDict
from typing import Dict, Callable

import gc

from torch.utils.data import DataLoader
from sklearn.decomposition import PCA


class RBDG_SLAMMER():
    def __init__(self, config):
        self.batch = 0
        self.min_cam_loss = 0
        torch.set_num_threads(config["num_threads"])
        self.hook_list = list()
        self.support_trajs_trans = None
        self.device = config['primary_device']
        device = torch.device(self.device)
        torch.cuda.set_device(device)
        torch.cuda.seed()
        self.l1_loss = torch.nn.L1Loss()

        if config['data']['get_pc_jono']:
            # config['data']['load_embeddings'] = False
            config['add_gaussians']['add_new_gaussians'] = False

        self.config = config
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
            self.wandb_init_refine = 0
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
    
    def get_trans(self, scale, h, w):
        h, w = int(h*scale), int(w*scale)
        trans_nearest = torchvision.transforms.Resize(
                (h, w), InterpolationMode.NEAREST)
        trans_bilinear = torchvision.transforms.Resize(
                (h, w), InterpolationMode.BILINEAR)
        return trans_nearest, trans_bilinear
    
    def resize_for_init(self, color, depth, instseg=None, embeddings=None, mask=None, x_grid=None, y_grid=None, bg=None, intrinsics=None, reverse=False):
        if self.config['init_scale'] != 1:
            if reverse:
                scale = 1/self.config['init_scale']
            else:
                scale = self.config['init_scale']
            
            trans_nearest, trans_bilinear = self.get_trans(scale, color.shape[1], color.shape[2])
              
            return_vals = [trans_bilinear(color), trans_nearest(depth)]

            if instseg is not None:
                return_vals = return_vals + [trans_nearest(instseg)]
            else:
                return_vals = return_vals + [None]

            if embeddings is not None:
                return_vals = return_vals + [trans_bilinear(embeddings)]
            else:
                return_vals = return_vals + [None]
            
            if mask is not None:
                return_vals = return_vals + [trans_nearest(mask)]
            else:
                return_vals = return_vals + [None]

            if x_grid is not None:
                return_vals = return_vals + [trans_bilinear(x_grid.unsqueeze(0)).squeeze(), trans_bilinear(y_grid.unsqueeze(0)).squeeze()]
            else:
                return_vals = return_vals + [None, None]
            
            if bg is not None:
                return_vals = return_vals + [trans_nearest(bg)]
            else:
                return_vals = return_vals + [None]

            if intrinsics is not None:
                intrinsics = datautils.scale_intrinsics(
                    intrinsics,
                    self.config['init_scale'],
                    self.config['init_scale']
                )
            return_vals = return_vals + [intrinsics]
            
            return return_vals
        
        else:
            return color, depth, instseg, embeddings, mask, x_grid, y_grid, bg, intrinsics
    
    def pre_process_depth(self, depth, color):
        if self.config['zeodepth']:
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=(int(depth.shape[1]/4), int(depth.shape[2]/4)), mode='nearest')
            depth = torch.nn.functional.interpolate(
                depth, size=(color.shape[1], color.shape[2]), mode='nearest').squeeze(1)
        return depth
        
    def get_pointcloud(
            self,
            color,
            depth,
            intrinsics,
            w2c,
            transform_pts=True, 
            mask=None,
            mean_sq_dist_method="projective",
            instseg=None,
            embeddings=None,
            time_idx=0,
            support_trajs=None,
            bg=None,
            start_pix=None):
        
        depth = self.pre_process_depth(depth, color)

        # TODO
        # Try +0.5 to get middle of pixel
        # x_grid += 0.5
        # y_grid += 0.5

        # downscale 
        color, depth, instseg, embeddings, _, _, _, bg, intrinsics = \
            self.resize_for_init(color, depth, instseg, embeddings, bg=bg, intrinsics=intrinsics)
        if mask is None:
            mask = (depth > 0)
        else:
            mask = mask & (depth > 0)
        
        if bg is not None:
            bg = bg[:, ::self.config['stride'], ::self.config['stride']]
            bg = bg.reshape(-1, 1)

        # Compute indices of pixels
        width, height = color.shape[2], color.shape[1]
        x_grid, y_grid = torch.meshgrid(torch.arange(width).to(self.device).float(), 
                                        torch.arange(height).to(self.device).float(),
                                        indexing='xy')

        x_grid = x_grid[::self.config['stride'], ::self.config['stride']]
        y_grid = y_grid[::self.config['stride'], ::self.config['stride']]
        mask = mask[:, ::self.config['stride'], ::self.config['stride']]

        # get pixel grid into 3D
        mask = mask.reshape(-1)
        xx = (x_grid - intrinsics[0][2])/intrinsics[0][0]
        yy = (y_grid - intrinsics[1][2])/intrinsics[1][1]
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        depth_z = depth[0][::self.config['stride'], ::self.config['stride']].reshape(-1)

        # Initialize point cloud
        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        if transform_pts:
            pix_ones = torch.ones(pts_cam.shape[0], 1).to(self.device).float()
            pts4 = torch.cat((pts_cam, pix_ones), dim=1)
            c2w = torch.inverse(w2c)
            pts = (c2w @ pts4.T).T[:, :3]
        else:
            pts = pts_cam

        # Compute mean squared distance for initializing the scale of the Gaussians
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            # scale_gaussian = depth_z / ((intrinsics[0][0]/self.config['init_scale'] + intrinsics[1][1]/self.config['init_scale'])/2)
            scale_gaussian = depth_z / ((intrinsics[0][0] + intrinsics[1][1])/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
        
        # Colorize point cloud
        cols = torch.permute(color, (1, 2, 0)) # (C, H, W) -> (H, W, C) -> (H * W, C)
        cols = cols[::self.config['stride'], ::self.config['stride']].reshape(-1, 3)
        point_cld = torch.cat((pts, cols), -1)
        if instseg is not None:
            instseg = torch.permute(instseg, (1, 2, 0))
            instseg = instseg[::self.config['stride'], ::self.config['stride']].reshape(-1, 1) # (C, H, W) -> (H, W, C) -> (H * W, C)
            point_cld = torch.cat((point_cld, instseg), -1)
        if embeddings is not None:
            channels = embeddings.shape[0]
            embeddings = torch.permute(embeddings, (1, 2, 0))
            embeddings = embeddings[::self.config['stride'], ::self.config['stride']].reshape(-1, channels) # (C, H, W) -> (H, W, C) -> (H * W, C)
            point_cld = torch.cat((point_cld, embeddings), -1)
                
        if self.config['remove_outliers_l2'] < 10:
            one_pix_diff = (
                torch.sqrt(torch.pow(self.config['remove_outliers_l2']/intrinsics[0][0], 2)) * depth_z + 
                torch.sqrt(torch.pow(0*self.config['remove_outliers_l2']/intrinsics[1][1], 2)) * depth_z)/2
            dist, _ = torch_3d_knn(point_cld[:, :3].contiguous().float(), num_knn=5)
            dist = dist[:, 1:]
            dist_mask = dist.mean(-1).clip(min=0.0000001) < one_pix_diff

           # mask everything
            pts = pts[dist_mask]
            point_cld = point_cld[dist_mask]
            mask = mask[dist_mask]
            instseg = instseg[dist_mask]
            mean3_sq_dist = mean3_sq_dist[dist_mask]
            bg = bg[dist_mask]

        # filter small segments
        if instseg is not None:
            uniques, counts = point_cld[:, 6].unique(return_counts=True)
            big_segs = torch.isin(point_cld[:, 6], uniques[counts>self.config['filter_small_segments']])

        if instseg is not None:
            mask = mask & big_segs

        # mask background points and incoming points
        if mask is not None:
            point_cld = point_cld[mask]
            mean3_sq_dist = mean3_sq_dist[mask]
            if bg is not None:
                bg = bg[mask]
        
        if self.config['just_fg']:
            point_cld = point_cld[~bg.squeeze()]
            mean3_sq_dist = mean3_sq_dist[~bg.squeeze()]
            bg = bg[~bg.squeeze()]
        
        if self.config['add_gaussians']['only_bg'] and time_idx > 0:
            point_cld = point_cld[bg.squeeze()]
            mean3_sq_dist = mean3_sq_dist[bg.squeeze()]
            bg = bg[bg.squeeze()]

        return point_cld, mean3_sq_dist, bg

    def get_pointcloud_jono(
            self,
            intrinsics,
            mean_sq_dist_method="projective",
            start_pix=None,
            initial_pc=False):

        if initial_pc:
            point_cld = torch.from_numpy(np.load(os.path.join(
                self.config['data']['basedir'],
                os.path.dirname(os.path.dirname(self.config['data']['sequence'])),
                'init_pt_cld.npz'))['data']).to(self.device)
        elif self.config['data']['load_embeddings']:
            params = np.load(os.path.join(
                '../Dynamic3DGaussians/experiments/output_first_embeddings/embeddings/',
                os.path.dirname(os.path.dirname(self.config['data']['sequence'])),
                'params.npz'))
            point_cld = torch.from_numpy(params['means3D']).to(self.device)
            color = torch.from_numpy(params['rgb_colors']).to(self.device)
            seg_colors = torch.from_numpy(params['seg_colors']).to(self.device)[:, 2].unsqueeze(-1)
            embeddings = torch.from_numpy(params['embeddings']).to(self.device)
            embeddings = torch.from_numpy(self.dataset.embedding_downscale.transform(embeddings.cpu().numpy())).to(self.device)
            point_cld = torch.cat([point_cld, color, seg_colors, embeddings], dim=-1)
        else:
            params = np.load(os.path.join(
                '../Dynamic3DGaussians/experiments/output_first/exp1/',
                os.path.dirname(os.path.dirname(self.config['data']['sequence'])),
                'params.npz'))
            point_cld = torch.from_numpy(params['means3D']).to(self.device)
            color = torch.from_numpy(params['rgb_colors']).to(self.device)
            seg_colors = torch.from_numpy(params['seg_colors']).to(self.device)[:, 2].unsqueeze(-1)
            point_cld = torch.cat([point_cld, color, seg_colors], dim=-1)

        if not self.config['data']['do_transform']:
            meta_path = os.path.join(
                self.config['data']['basedir'],
                os.path.dirname(os.path.dirname(self.config['data']['sequence'])),
                'meta.json')
            cam_id = int(os.path.basename(self.config['data']['sequence']))

            with open(meta_path, 'r') as jf:
                data = json.load(jf)
            idx = data['cam_id'][0].index(cam_id)
            w2c = torch.tensor(data['w2c'][0][idx]).to(self.device).float()

            pts = point_cld[:, :3].float()
            pts_ones = torch.ones(pts.shape[0], 1).to(self.device).float()
            pts4 = torch.cat((pts, pts_ones), dim=1)
            transformed_pts = (w2c @ pts4.T).T[:, :3]
            point_cld[:, :3] = transformed_pts
            transformed_pts = transformed_pts[point_cld[:, 2] > 1]
            point_cld = point_cld[point_cld[:, 2] > 1]
        else:
            transformed_pts = point_cld[:, :3]
        
        dist, _ = torch_3d_knn(point_cld[:, :3].contiguous().float(), num_knn=4)
        dist = dist[:, 1:]
        mean3_sq_dist = dist.mean(-1).clip(min=0.0000001)
        point_cld = point_cld[mean3_sq_dist<0.01]
        transformed_pts = transformed_pts[mean3_sq_dist<0.01]
        mean3_sq_dist = mean3_sq_dist[mean3_sq_dist<0.01]

        # scale_gaussian = transformed_pts[transformed_pts[:, 2] > 1][:, 2] / ((intrinsics[0][0] + intrinsics[1][1])/2)
        # mean3_sq_dist = (2*scale_gaussian)**2

        point_cld[:, -1] = point_cld[:, -1]
        bg = point_cld[:, -1].unsqueeze(1)

        return point_cld, mean3_sq_dist, bg

    def initialize_params(
            self,
            init_pt_cld,
            num_frames,
            mean3_sq_dist,
            gaussian_distribution,
            bg,
            width=1):
        
        num_pts = init_pt_cld.shape[0]

        logit_opacities = torch.ones((num_pts, 1), dtype=torch.float, device=self.device)
        if gaussian_distribution == "isotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
        elif gaussian_distribution == "anisotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
        else:
            raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")

        unnorm_rotations = torch.zeros((num_pts, 4, self.num_frames), dtype=torch.float32).to(self.device)
        unnorm_rotations[:, 0, :] = 1
        means3D = torch.zeros((num_pts, 3, self.num_frames), dtype=torch.float32).to(self.device)
        means3D[:, :, 0] = init_pt_cld[:, :3]

        params = {
                'means3D': means3D[:, :, 0],
                'rgb_colors': init_pt_cld[:, 3:6],
                'unnorm_rotations': unnorm_rotations[:, :, 0],
                'logit_opacities': logit_opacities,
                'log_scales': log_scales,
                'instseg': init_pt_cld[:, 6].to(self.device).long()
            }
    
        # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
        cam_rots = np.tile([1, 0, 0, 0], (1, 1))
        cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
        params['cam_unnorm_rots'] = cam_rots
        params['cam_trans'] = np.zeros((1, 3, num_frames))
        # params['cam_dfx_dfy_dcx_dcy'] = np.zeros((1, 4))

        if self.dataset.load_embeddings:
            params['embeddings'] = init_pt_cld[:, 7:].to(self.device).float()

        if bg is not None:
            params['bg'] = bg

        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).to(self.device).float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.to(self.device).float().contiguous().requires_grad_(True))
        
        all_log_scales = torch.zeros(params['means3D'].shape[0], self.num_frames).to(self.device).float()

        variables = {
            'max_2D_radius': torch.zeros(means3D.shape[0]).to(self.device).float(),
            'means2D_gradient_accum': torch.zeros(means3D.shape[0], dtype=torch.float32).to(self.device),
            'denom': torch.zeros(params['means3D'].shape[0]).to(self.device).float(),
            'timestep': torch.zeros(params['means3D'].shape[0]).to(self.device).float(),
            'visibility': torch.zeros(params['means3D'].shape[0], self.num_frames).to(self.device).float(),
            'rgb_colors': torch.zeros(params['means3D'].shape[0], 3, self.num_frames).to(self.device).float(),
            'log_scales': all_log_scales if gaussian_distribution == "isotropic" else torch.tile(all_log_scales[..., None], (1, 1, 3)).permute(0, 2, 1),
            "to_deactivate": torch.ones(params['means3D'].shape[0]).to(self.device).float() * 100000,
            "means3D": means3D,
            "unnorm_rotations": unnorm_rotations}
        
        instseg_mask = params["instseg"].long()
        if not self.config['use_seg_for_nn']:
            instseg_mask = torch.ones_like(instseg_mask).long().to(instseg_mask.device)

        return params, variables
        
    def initialize_optimizer(self, params, lrs_dict, tracking=True):
        lrs = lrs_dict
        param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
        if self.motion_mlp is not None:
            param_groups += [{'params': self.motion_mlp.parameters(), 'lr': self.config['motion_lr'], 'name': 'motion_mlp'}]
        if self.base_transformations is not None:
            param_groups += [
                {'params': self.base_transformations.parameters(), 'lr': self.config['motion_lr'], 'name': 'base_transforms'}]

        if tracking:
            return torch.optim.Adam(param_groups)
        else:
            return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    def get_loss_cam(self,
                      params,
                      variables,
                      curr_data,
                      iter_time_idx,
                      config=None,
                      filter_gaussians=False,
                      use_gt_mask=False,
                      use_depth_error=False):

        if not self.config['just_fg'] and filter_gaussians:
            _params = dict()
            bg = (params['bg'].detach().clone() > 0.5).squeeze()
            num_gaussians = params['means3D'].shape[0]
            for k, v in params.items():
                if v.shape[0] == num_gaussians:
                    _params[k] = v[bg]
                else:
                    _params[k] = v
            _variables = dict()
            for k, v in variables.items():
                if k not in ['scene_radius', 'last_time_idx'] and v is not None and v.shape[0] == num_gaussians:
                    _variables[k] = v[bg]
                else:
                    _variables[k] = v
        else:
            _params = params
            _variables = variables
 
        # Initialize Loss Dictionary
        losses = {}
        _variables, im, _, depth, instseg, mask, _, _, _, time_mask, _, _, embeddings, bg, _, _ = \
            get_renderings(
                _params,
                _variables,
                iter_time_idx,
                curr_data,
                config,
                track_cam=True,
                get_seg=True,
                get_embeddings=self.config['data']['load_embeddings'],
                remove_close=self.config['remove_close'])
        
        if not self.config['just_fg']:
            bg_mask = bg.detach().clone() > 0.5
            bg_mask_gt = curr_data['bg'].detach().clone() > 0.5
            mask = (bg_mask & mask.detach()).squeeze()
            mask_gt = (bg_mask_gt & mask.detach()).squeeze()
            if use_gt_mask:
                mask = mask_gt
        else:
            fg_mask = ~(bg.detach().clone() > 0.5)
            mask = fg_mask & mask.detach()
        
        if use_depth_error:
            depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
            error_depth = (depth > curr_data['depth']) * (depth_error > 25*depth_error.median())
            mask = mask | error_depth

        # Depth loss
        if config['use_l1']:
            losses['depth'] = l1_loss_v1(
                curr_data['depth'].squeeze(),
                depth.squeeze(),
                mask=mask,
                reduction='mean')
        
        # RGB Loss
        losses['im'] = l1_loss_v1(
                    curr_data['im'].permute(1, 2, 0),
                    im.permute(1, 2, 0),
                    mask=mask,
                    reduction='mean')

        # EMBEDDING LOSS
        if self.dataset.load_embeddings:
            embeddings_gt = curr_data['embeddings']
            if self.config['norm_embeddings']:
                embeddings_gt = torch.nn.functional.normalize(curr_data['embeddings'], p=2, dim=0)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=0)
            losses['embeddings'] = l1_loss_v1(
                embeddings_gt.permute(1, 2, 0),
                embeddings.permute(1, 2, 0),
                mask=mask,
                reduction='mean')
        
        if config['loss_weights']['instseg']:
                losses['instseg'] = l1_loss_v1(
                    instseg.squeeze(),
                    curr_data['instseg'].squeeze(),
                    mask=mask.squeeze())

        weighted_losses = {k: v * config['loss_weights'][k] for k, v in losses.items()}
        loss = sum(weighted_losses.values())
        
        weighted_losses['loss'] = loss

        return loss, weighted_losses, variables

    def save_pca_downscaled(self, features, save_dir, time_idx):
        features = features.permute(1, 2, 0).detach().cpu().numpy()
        shape = features.shape
        if shape[2] != 3:
            pca = PCA(n_components=3)
            pca.fit(features.reshape(-1, shape[2]))
            features = pca.transform(
                features.reshape(-1, shape[2]))
            features = features.reshape(
                (shape[0], shape[1], 3))
        vmax, vmin = features.max(), features.min()
        normalized_features = np.clip((features - vmin) / (vmax - vmin + 1e-10), 0, 1)
        normalized_features_colormap = (normalized_features * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(save_dir, "gs_{:04d}.png".format(time_idx)), normalized_features_colormap)
        if time_idx == self.num_frames - 1: 
            make_vid(save_dir)

    def get_loss_dyno(self,
                      curr_data,
                      iter_time_idx,
                      num_iters,
                      iter=0,
                      config=None,
                      init_next=False,
                      early_stop_eval=False,
                      last=False):
 

        # Initialize Loss Dictionary
        losses = {}
        # get renderings for current time
        self.variables, im, radius, depth, instseg, mask, transformed_gaussians, visible, weight, time_mask, _, _, embeddings, bg, _, coefficients = \
            get_renderings(
                self.params,
                self.variables,
                iter_time_idx,
                curr_data,
                config,
                get_seg=True,
                disable_grads=init_next,
                get_embeddings=self.config['data']['load_embeddings'],
                motion_mlp=self.motion_mlp if iter_time_idx != 0 else None,
                base_transformations=self.base_transformations if iter_time_idx != 0 else None,
                remove_close=self.config['remove_close'])

        if coefficients is not None:
            coefficients = torch.abs(coefficients)
            max_val = coefficients.max(dim=1).values.squeeze()
            coefficients_loss_sparse = (coefficients.squeeze() / max_val.unsqueeze(1)).mean()
            coefficients_loss_low = coefficients.mean()
            # coefficients_similar = torch.abs(coefficients[variables['self_indices']].squeeze() - coefficients[variables['neighbor_indices']].squeeze()).mean()
            losses['coeff'] = coefficients_loss_low + coefficients_loss_sparse # + coefficients_similar

        mask = mask.detach()
        if self.config['just_fg']:
            fg_mask = ~(bg.detach().clone() > 0.5)
            mask = fg_mask & mask.detach()

        # Depth loss
        if config['use_l1']:
            if True: # not config['ssmi_all_mods']:
                losses['depth'] = l1_loss_v1(curr_data['depth'], depth, mask, reduction='mean')
            else:
                losses['depth'] = (iter)/num_iters * l1_loss_v1(depth, curr_data['depth']) + (num_iters-iter)/num_iters * (
                    1.0 - calc_ssim(depth, curr_data['depth']))
        
        # RGB Loss
        if init_next or config['use_sil_for_loss'] or config['ignore_outlier_depth_loss']:
            color_mask = torch.tile(mask, (3, 1, 1))
            color_mask = color_mask.detach()
            losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].mean()
        elif not config['calc_ssmi']:
            losses['im'] = torch.abs(curr_data['im'] - im).mean()
        else:
            losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

        # EMBEDDING LOSS
        if self.dataset.load_embeddings and config['loss_weights']['embeddings'] != 0:
            embeddings_gt = curr_data['embeddings']
            if self.config['norm_embeddings']:
                embeddings_gt = torch.nn.functional.normalize(curr_data['embeddings'], p=2, dim=0)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=0)

            if not config['ssmi_all_mods']:
                losses['embeddings'] = l2_loss_v2(
                    embeddings_gt.permute(1, 2, 0),
                    embeddings.permute(1, 2, 0),
                    mask.squeeze(),
                    reduction='mean')
            else:
                losses['embeddings'] = (iter)/num_iters * l1_loss_v1(embeddings, embeddings_gt) + (num_iters-iter)/num_iters * (
                    1.0 - calc_ssim(embeddings, embeddings_gt))
        # self.save_pca_downscaled(embeddings, '', iter_time_idx)

        if config['loss_weights']['instseg'] != 0:
            if not config['ssmi_all_mods']:
                losses['instseg'] = l1_loss_v1(instseg.squeeze(), curr_data['instseg'].float().squeeze(), mask=mask.squeeze())
            else:
                losses['instseg'] = (iter)/num_iters * l1_loss_v1(instseg, curr_data['instseg'].float()) + (num_iters-iter)/num_iters * (
                    1.0 - calc_ssim(instseg, curr_data['instseg'].float()))
            
        # BG REG
        if config['bg_reg']:
            # hard foce bg
            if iter_time_idx > 0:
                is_bg = self.params['bg'].detach().clone().squeeze() > 0.5
                losses['bg_reg'] = l1_loss_v1(
                    self.params['means3D'][:, :][is_bg],
                    self.variables['means3D'][:, :, iter_time_idx-1][is_bg])

            # bg loss with mask    
            if not config['ssmi_all_mods']:
                losses['bg_loss'] = l1_loss_v1(bg.squeeze(), curr_data['bg'].float().squeeze(), mask=mask.squeeze())
            else:
                losses['bg_loss'] = (iter)/num_iters * l1_loss_v1(bg, curr_data['bg'].float()) + (num_iters-iter)/num_iters * (
                    1.0 - calc_ssim(bg, curr_data['bg'].float()))
        
        l1_mask = self.variables['timestep']<iter_time_idx
        if config['loss_weights']['l1_bg'] and iter_time_idx > 0:
            losses['l1_bg'] = l1_loss_v1(self.params['bg'][l1_mask], self.variables['prev_bg'])
        
        if config['loss_weights']['l1_rgb'] and iter_time_idx > 0:
            losses['l1_rgb'] = l1_loss_v1(self.params['rgb_colors'][l1_mask], self.variables['prev_rgb_colors'])
        
        if config['loss_weights']['l1_embeddings'] and iter_time_idx > 0 and self.dataset.load_embeddings:
            losses['l1_embeddings'] = l1_loss_v1(self.params['embeddings'][l1_mask],self. variables['prev_embeddings'])

        if config['loss_weights']['l1_scale'] != 0 and iter_time_idx > 0:
            losses['l1_scale'] = l1_loss_v1(self.params['log_scales'][l1_mask], self.variables['prev_log_scales'])
            
        # ADD DYNO LOSSES LIKE RIGIDITY
        # DYNO LOSSES
        if config['dyno_losses'] and iter_time_idx > 0:
            # print(variables['timestep'])
            dyno_losses_curr, offset_0 = dyno_losses(
                self.params,
                iter_time_idx,
                transformed_gaussians,
                self.variables,
                self.variables['offset_0'],
                iter,
                use_iso=True,
                update_iso=True, 
                weight=config['dyno_weight'],
                mag_iso=config['mag_iso'],
                weight_rot=config['weight_rot'],
                weight_rigid=config['weight_rigid'],
                weight_iso=config['weight_iso'],
                last_x=config['last_x'])
            self.variables['offset_0'] = offset_0

            losses.update(dyno_losses_curr)
        
        if config['loss_weights']['smoothness'] != 0 and iter_time_idx > 1:
            delta_0 = self.variables['means3D'][:, :, iter_time_idx-1].detach().clone()-self.variables['means3D'][:, :, iter_time_idx-2].detach().clone()
            delta_1 = self.params['means3D'][:, :, iter_time_idx]-self.params['means3D'][:, :, iter_time_idx-1].detach().clone()
            smoothness = torch.abs(delta_0-delta_1)
            smoothness = smoothness[self.variables['timestep']<=iter_time_idx-2]
            losses['smoothness'] = smoothness.mean()
        
        weighted_losses = {k: v * config['loss_weights'][k] for k, v in losses.items()}
        loss = sum(weighted_losses.values())

        seen = radius > 0
        self.variables['max_2D_radius'][time_mask][seen] = torch.max(radius[seen], self.variables['max_2D_radius'][time_mask][seen])
        self.variables['seen'] = torch.zeros_like(self.variables['max_2D_radius'], dtype=bool, device=self.device)
        self.variables['seen'][time_mask] = True
        weighted_losses['loss'] = loss
    
        if (iter == num_iters - 1 or early_stop_eval) and self.config['eval_during']:
            psnr, rmse, depth_l1, ssim, lpips, self.eval_pca = eval_during(
                curr_data,
                self.params,
                iter_time_idx,
                self.eval_dir,
                im=im.detach().clone(),
                rastered_depth=depth.detach().clone(),
                rastered_sil=mask.detach().clone(),
                rastered_bg=bg.detach().clone(),
                rastered_inst=instseg.detach().clone(),
                rendered_embeddings=embeddings.detach().clone() if embeddings is not None else embeddings,
                pca=self.eval_pca,
                viz_config=self.config['viz'],
                num_frames=self.num_frames)
            self.psnr_list.append(psnr.cpu().numpy())
            self.rmse_list.append(rmse.cpu().numpy())
            self.l1_list.append(depth_l1.cpu().numpy())
            self.ssim_list.append(ssim.cpu().numpy())
            self.lpips_list.append(lpips)

        return loss, weighted_losses, visible, weight

    def initialize_new_params(
            self,
            new_pt_cld,
            mean3_sq_dist,
            gaussian_distribution,
            bg,
            time_idx):
        
        num_pts = new_pt_cld.shape[0]

        logit_opacities = torch.ones((num_pts, 1), dtype=torch.float, device=self.device)
        if gaussian_distribution == "isotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
        elif gaussian_distribution == "anisotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
        else:
            raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")

        unnorm_rotations = torch.zeros((num_pts, 4), dtype=torch.float32).to(self.device)
        unnorm_rotations[:, 0] = 1

        params = {
                'means3D': new_pt_cld[:, :3],
                'rgb_colors': new_pt_cld[:, 3:6],
                'unnorm_rotations': unnorm_rotations,
                'logit_opacities': logit_opacities,
                'log_scales': log_scales,
                'instseg': new_pt_cld[:, 6].long().to(self.device)
            }

        if self.dataset.load_embeddings:
            params['embeddings'] = new_pt_cld[:, 7:].to(self.device).float()
        
        if bg is not None:
            params['bg'] = bg
            
        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).to(self.device).float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.to(self.device).float().contiguous().requires_grad_(True))
        
        variables = {
            'means3D': torch.zeros((num_pts, 3, self.num_frames), dtype=torch.float32).to(self.device),
            'unnorm_rotations': params['unnorm_rotations'].detach().clone().unsqueeze(2).repeat(1, 1, self.num_frames)}
        variables['means3D'][:, :, :time_idx+1] = params['means3D'].detach().clone().unsqueeze(2).repeat(1, 1, time_idx+1)

        return params, variables

    def proj_means_to_2D(self, cam, time_idx, shape):
        transformed_gs_traj_3D, _ = transform_to_frame(
                self.params,
                time_idx,
                gaussians_grad=False,
                camera_grad=False,
                delta=0)
        
        # project existing params to instseg
        points_xy = cam.projmatrix.squeeze().T.matmul(torch.cat(
            [transformed_gs_traj_3D['means3D'][:, :], torch.ones(self.params['means3D'].shape[0], 1).to(self.device)], dim=1).T)
        points_xy = points_xy / points_xy[3, :]
        points_xy = points_xy[:2].T
        points_xy[:, 0] = ((points_xy[:, 0]+1)*cam.image_width-1) * 0.5
        points_xy[:, 1] = ((points_xy[:, 1]+1)*cam.image_height-1) * 0.5
        points_xy = torch.round(points_xy).long()
        points_xy[:, 0] = torch.clip(points_xy[:, 0], min=0, max=shape[2]-1)
        points_xy[:, 1] = torch.clip(points_xy[:, 1], min=0, max=shape[1]-1)
        return points_xy.long()

    def deactivate_gaussians_drift(
        self,
        curr_data,
        time_idx,
        optimizer):

        neighbor_dist = torch.cdist(
                self.params['means3D'][:, :, time_idx].detach().clone()[self.variables['self_indices'], :].unsqueeze(1),
                self.params['means3D'][:, :, time_idx].detach().clone()[self.variables['neighbor_indices'], :].unsqueeze(1)
            ).squeeze()
        neighbor_dist = neighbor_dist[self.variables['timestep'][self.variables['self_indices']]<time_idx]            
        if self.variables['offset_0'] is not None:
            neighbor_dist_0 = torch.linalg.norm(self.variables['offset_0'], ord=2, dim=-1)
            torch.use_deterministic_algorithms(False)
            rel_increase_neighbor_dist = scatter_mean(
                torch.abs(neighbor_dist_0-neighbor_dist)/(neighbor_dist_0+1e-10), 
                self.variables['self_indices'][self.variables['timestep'][self.variables['self_indices']]<time_idx])
            rabs_increase_neighbor_dist = scatter_mean(
                torch.abs(neighbor_dist_0-neighbor_dist), 
                self.variables['self_indices'][self.variables['timestep'][self.variables['self_indices']]<time_idx])
            torch.use_deterministic_algorithms(True)

            deactivation_mask  = \
                (rel_increase_neighbor_dist > self.config['deactivate_gaussians']['rem_rel_drift_thresh']) & \
                (rabs_increase_neighbor_dist > self.config['deactivate_gaussians']['rem_abs_drift_thresh']) & \
                (self.variables["to_deactivate"][self.variables['timestep']<time_idx] != 100000)
            self.variables["to_deactivate"][self.variables['timestep']<time_idx][deactivation_mask] = time_idx
            print("Deactivated/Number of Gaussians bceause of drift", deactivation_mask.sum())

    def remove_gaussians_with_depth(
            self,
            curr_data,
            time_idx, 
            optimizer):
        
        # print(torch.histogram(self.params['logit_opacities'].detach().clone().cpu(), bins=100))
        # project current means to 2D
        with torch.no_grad():
            points_xy = self.proj_means_to_2D(curr_data['cam'], time_idx, curr_data['depth'].shape)
        
        # compare projected depth to parameter depth
        existing_params_proj_depth = torch.ones_like(self.params['means3D'][:, 2, time_idx])
        existing_params_proj_depth = curr_data['depth'][:, points_xy[:, 1], points_xy[:, 0]]
        existing_params_depth = self.params['means3D'][:, 2, time_idx]
        depth_error = torch.abs(existing_params_depth - existing_params_proj_depth)

        # if depth error is large, remove gaussian
        factor = self.config['remove_gaussians']['remove_factor']
        to_remove_depth = (existing_params_depth < existing_params_proj_depth) * (depth_error > factor*depth_error.median())

        # remove gaussians with low opacity
        to_remove_opa = torch.sigmoid(self.params['logit_opacities']) < self.config['remove_gaussians']['rem_opa_thresh']
        to_remove = to_remove_depth.squeeze() & to_remove_opa.squeeze()

        # remove gaussians with large_scale
        to_remove_scale = torch.sigmoid(self.params['log_scales']) > self.config['remove_gaussians']['rem_scale_thresh']
        # print(torch.histogram(torch.sigmoid(self.params['log_scales']).detach().clone().cpu(), bins=100))
        to_remove = to_remove.squeeze() & to_remove_scale.squeeze()

        if to_remove.sum():
            self.params, self.variables, self.support_trajs_trans = remove_points(
                to_remove.squeeze(),
                self.params,
                self.variables,
                optimizer=optimizer,
                support_trajs_trans=self.support_trajs_trans)

        if self.config['use_wandb']:
            self.wandb_run.log({
                "Removed/Number of Gaussians with Depth": to_remove_depth.sum(),
                "Removed/Number of Gaussians with Opacity": to_remove_opa.sum(),
                "Removed/Number of Gaussians with Scale": to_remove_scale.sum(),
                "Removed/step": self.wandb_time_step})
        
    def get_depth_for_new_gauss(
            self,
            params,
            time_idx,
            curr_data,
            gauss_time_idx=None,
            rgb=False):
        
        print(params['means3D'])
        # Silhouette Rendering
        transformed_gaussians, _ = transform_to_frame(
            params,
            time_idx,
            gaussians_grad=False,
            camera_grad=False,
            gauss_time_idx=gauss_time_idx)
        
        print(transformed_gaussians['means3D'])

        if rgb:
            rendervar, time_mask = transformed_params2rendervar(
                params,
                transformed_gaussians,
                time_idx,
                first_occurance=self.variables['timestep'],
                active_gaussians_mask=None,
                depth=curr_data['depth'] if self.config['remove_close'] else None)
            im, _, _, _, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
            return im
        
        depth_sil_rendervar, _ = transformed_params2depthplussilhouette(
            params,
            torch.eye(4).to(curr_data['w2c'].device),
            transformed_gaussians,
            time_idx,
            self.variables['timestep'],
            time_window=self.config['time_window'],
            depth=curr_data['depth'] if self.config['remove_close'] else None)

        if self.config['init_scale'] != 1:
            rescaled = self.resize_for_init(
                curr_data['im'],
                curr_data['depth'],
                intrinsics=self.intrinsics)
            intrinsics = rescaled[8]
            cam = setup_camera(
                int(curr_data['im'].shape[2]*self.config['init_scale']), 
                int(curr_data['im'].shape[1]*self.config['init_scale']),
                intrinsics,
                curr_data['w2c'],
                device=self.device
            )
            gt_depth = rescaled[1][0, :, :]
        else:
            cam = curr_data['cam']
            gt_depth = curr_data['depth'][0, :, :]
        print(cam)
        depth_sil, _, _, _, _ = Renderer(raster_settings=cam)(**depth_sil_rendervar)

        return depth_sil, gt_depth

    def store_vis(self, time_idx, to_save, save_dir):
        os.makedirs(os.path.join(self.eval_dir, save_dir), exist_ok=True)
        non_presence_depth_mask_cpu = (to_save.detach().clone()*255).cpu().numpy().astype(np.uint8)
        imageio.imwrite(
            os.path.join(self.eval_dir, save_dir, "gs_{:04d}_{:04d}.png".format(self.batch, time_idx)),
            non_presence_depth_mask_cpu)
        del non_presence_depth_mask_cpu
        if time_idx == self.num_frames - 1:
            make_vid(os.path.join(self.eval_dir, save_dir))

    def add_new_gaussians(
            self,
            curr_data,
            sil_thres, 
            depth_error_factor,
            time_idx,
            mean_sq_dist_method,
            gaussian_distribution,
            params,
            variables):
        
        # get depth for adding gaussians and rescaled cam
        depth_sil, gt_depth = self.get_depth_for_new_gauss(
            params,
            time_idx,
            curr_data)

        silhouette = depth_sil[1, :, :]
        non_presence_sil_mask = (silhouette < sil_thres)
        self.store_vis(time_idx, silhouette, 'presence_mask')
        quit()
        # Check for new foreground objects by u sing GT depth
        render_depth = depth_sil[0, :, :]

        if self.config['add_gaussians']['use_depth_error_for_adding_gaussians']:
            depth_error = (torch.abs(gt_depth - render_depth)/gt_depth) * (gt_depth > 0)
            non_presence_depth_mask =  (depth_error > depth_error_factor*depth_error.median()) # * (render_depth > gt_depth)
            non_presence_mask = non_presence_sil_mask | non_presence_depth_mask

            # depth error
            # depth_error_store = (depth_error-depth_error.min())/(depth_error.max()-depth_error.min())
            # self.store_vis(time_idx, depth_error_store, 'depth_error')
            
            # non presence depth mask
            # self.store_vis(time_idx, non_presence_depth_mask, 'non_presence_depth_mask')

        else:
            non_presence_mask = non_presence_sil_mask

        # Determine non-presence mask
        # Get the new frame Gaussians based on the Silhouette
        if torch.sum(non_presence_mask) > 0:
            # TODO FOR CODE CLEAN UP
            # Get the new pointcloud in the world frame
            curr_cam_rot = torch.nn.functional.normalize(
                self.params['cam_unnorm_rots'][..., time_idx].detach())
            curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
            curr_c2c0 = torch.eye(4).to(self.device).float()
            curr_c2c0[:3, :3] = build_rotation(curr_cam_rot)
            curr_c2c0[:3, 3] = curr_cam_tran
            # TODO CODE CLEAN UP --> check this!! 
            # curr_w2c = curr_data['w2c'] @ curr_c2c0
            curr_w2c = self.variables['gt_w2c_all_frames'][0] @ curr_c2c0

            new_pt_cld, mean3_sq_dist, bg = self.get_pointcloud(
                curr_data['im'],
                curr_data['depth'],
                self.intrinsics, 
                curr_w2c,
                mask=non_presence_mask.unsqueeze(0),
                mean_sq_dist_method=mean_sq_dist_method,
                instseg=curr_data['instseg'],
                embeddings=curr_data['embeddings'],
                time_idx=time_idx,
                bg=curr_data['bg'])
            
            new_params, new_variables = self.initialize_new_params(
                new_pt_cld,
                mean3_sq_dist,
                gaussian_distribution,
                bg,
                time_idx)
            
            # numbers of gaussians
            num_new_gauss = new_params['means3D'].shape[0]
            num_gaussians = num_new_gauss + self.params['means3D'].shape[0]
        
            # cat new and old params
            for k, v in new_params.items():
                self.params[k] = torch.nn.Parameter(torch.cat((self.params[k], v), dim=0).requires_grad_(True))
            
            # update variables
            self.init_reset('max_2D_radius', 0, (num_gaussians))
            self.init_reset('denom', 0, (num_gaussians))
            self.init_reset('means2D_gradient_accum', 0, (num_gaussians))

            self.init_new_var('visibility', 0, (num_new_gauss, self.num_frames))
            self.init_new_var('rgb_colors', 0, (num_new_gauss, 3, self.num_frames))
            self.init_new_var('to_deactivate', 100000, (num_new_gauss))
            self.init_new_var('timestep', time_idx, (num_new_gauss))
            if gaussian_distribution == "anisotropic":
                self.init_new_var('log_scales', 0, (num_new_gauss, 3, self.num_frames))
            else:
                self.init_new_var('log_scales', 0, (num_new_gauss, self.num_frames))

            self.cat_old_new('means3D', new_variables)
            self.cat_old_new('unnorm_rotations', new_variables)

    def init_reset(self, k, val, shape):
        self.variables[k] = val*torch.ones(shape, device=self.device).float().contiguous()

    def init_new_var(self, k, val, shape):
        self.variables[k] = torch.cat((self.variables[k], val*torch.ones(shape, device=self.device, dtype=float)), dim=0).contiguous()

    def cat_old_new(self, k, new_variables):
        self.variables[k] = torch.cat((self.variables[k], new_variables[k]), dim=0).contiguous()

    def make_gaussians_static(
            self,
            curr_time_idx,
            delta_tran):
        
        mask = (curr_time_idx - self.variables['timestep'] >= 1).squeeze()
        with torch.no_grad():
            if delta_tran[mask].shape[0]:
                # Gaussian kNN, point translation
                weight = self.variables['neighbor_weight_sm'].unsqueeze(1)
                torch.use_deterministic_algorithms(False)
                kNN_trans = scatter_add(
                        weight * delta_tran[self.variables['neighbor_indices']],
                        self.variables['self_indices'], dim=0)
                torch.use_deterministic_algorithms(True)
                point_trans = delta_tran

                return kNN_trans, point_trans
            else:
                return delta_tran, delta_tran
    
    def forward_propagate_camera(
            self,
            curr_time_idx,
            forward_prop=True,
            simple_rot=False,
            simple_trans=False):
        with torch.no_grad():
            if curr_time_idx > 1 and forward_prop:
                if simple_rot or simple_trans:
                    print(f"Using simple rot {simple_rot} and simple trans {simple_trans} for cam forward prop!")
                # get time index
                time_1 = curr_time_idx - 1 if curr_time_idx > 0 else curr_time_idx
                time_2 = curr_time_idx

                # forward prop Rotation
                prev_rot2 = normalize_quat(self.params['cam_unnorm_rots'][:, :, time_2].detach())                                                                                                                                  
                prev_rot1 = normalize_quat(self.params['cam_unnorm_rots'][:, :, time_1].detach())
                if not simple_rot:
                    prev_rot1_inv = prev_rot1.clone()
                    prev_rot1_inv[:, 1:] = -1 * prev_rot1_inv[:, 1:]
                    delta_rot = quat_mult(prev_rot2, prev_rot1_inv) # rot from 1 -> 2
                    new_rot = quat_mult(delta_rot, prev_rot2)
                else:
                    new_rot = torch.nn.functional.normalize(prev_rot2 + (prev_rot2 - prev_rot1))
                self.params['cam_unnorm_rots'][..., curr_time_idx + 1] = new_rot.detach()
                
                # forward prop translation
                prev_tran2 = self.params['cam_trans'][..., time_2].detach()
                prev_tran1 = self.params['cam_trans'][..., time_1].detach()
                if not simple_trans:
                    delta_rot_mat = build_rotation(delta_rot).squeeze()
                    new_tran = torch.bmm(delta_rot_mat.unsqueeze(0), prev_tran2.unsqueeze(2)).squeeze() - \
                        torch.bmm(delta_rot_mat.unsqueeze(0), prev_tran1.unsqueeze(2)).squeeze() + prev_tran2.squeeze()
                else:
                    new_tran = prev_tran2 + (prev_tran2 - prev_tran1)

                self.params['cam_trans'][..., curr_time_idx + 1] = new_tran.detach()
            else:
                # Initialize the camera pose for the current frame
                self.params['cam_unnorm_rots'][..., curr_time_idx + 1] = self.params['cam_unnorm_rots'][..., curr_time_idx].detach()
                self.params['cam_trans'][..., curr_time_idx + 1] = self.params['cam_trans'][..., curr_time_idx].detach()
    
    def forward_propagate_gaussians(
            self,
            curr_time_idx,
            forward_prop=True,
            simple_rot=False,
            simple_trans=True):

        # for all other timestamps moving
        with torch.no_grad():
            if forward_prop:
                if simple_rot or not simple_trans:
                    print(f'Using simple rot {simple_rot} and simple trans {simple_trans} for forward prop gauss!')
                # Get time mask 
                mask = (curr_time_idx - self.variables['timestep'] >= 0).squeeze()
                if self.config['make_bg_static']:
                    mask = mask & ~self.params['bg'].detach().clone().squeeze()

                # get time index
                time_1 = curr_time_idx - 1 if curr_time_idx > 0 else curr_time_idx
                time_2 = curr_time_idx
                
                # forward prop rotation
                rot_2 = normalize_quat(self.variables['unnorm_rotations'][:, :, time_2].detach().clone())                                                                                                                                  
                rot_1 = normalize_quat(self.variables['unnorm_rotations'][:, :, time_1].detach().clone())
                if not simple_rot:
                    rot_1_inv = rot_1.clone()
                    rot_1_inv[:, 1:] = -1 * rot_1_inv[:, 1:]
                    delta_rot = quat_mult(rot_2, rot_1_inv)
                    curr_rot = normalize_quat(self.variables['unnorm_rotations'][:, :, curr_time_idx].detach().clone())
                    new_rot = quat_mult(delta_rot, curr_rot)[mask]
                    new_rot = torch.nn.Parameter(new_rot.to(self.device).float().contiguous().requires_grad_(True))
                else:
                    new_rot = torch.nn.functional.normalize(rot_2 + (rot_2 - rot_1))
                self.params['unnorm_rotations'][mask, :] = new_rot

                # forward prop translation
                tran_2 = self.variables['means3D'][:, :, time_2].detach().clone()
                tran_1 = self.variables['means3D'][:, :, time_1].detach().clone()
                if not simple_trans:
                    delta_rot_mat = build_rotation(delta_rot).squeeze()
                    new_tran = torch.bmm(delta_rot_mat, tran_2.unsqueeze(2)).squeeze() - \
                        torch.bmm(delta_rot_mat, point_trans.unsqueeze(2)).squeeze() + tran_2.squeeze()         
                else:
                    delta_tran = tran_2 - tran_1       
                    kNN_trans, point_trans = self.make_gaussians_static(curr_time_idx, delta_tran)
                    curr_tran = self.variables['means3D'][:, :, curr_time_idx].detach().clone()
                    if self.config['mov_init_by'] == 'kNN':
                        new_tran = (curr_tran + kNN_trans)[mask]
                    elif self.config['mov_init_by'] == 'per_point':
                        new_tran = (curr_tran + point_trans)[mask]
                    new_tran = torch.nn.Parameter(new_tran.to(self.device).float().contiguous().requires_grad_(True))
                    self.params['means3D'][mask, :] = new_tran

                # For static objects set new rotation and translation
                if self.config['make_bg_static']:
                    mask = (curr_time_idx - self.variables['timestep'] >= 0).squeeze()
                    mask = mask & self.params['bg'].detach().clone().squeeze()
                    self.params['unnorm_rotations'][mask, :] = curr_rot[mask]
                    self.params['means3D'][mask, :] = curr_tran[mask]
            else:
                print('Not forward propagating gaussians.')
    
    def convert_params_to_store(self, params):
        params_to_store = {}
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                params_to_store[k] = v.detach().clone()
            else:
                params_to_store[k] = v
        return params_to_store
    
    def initialize_timestep(self, scene_radius_depth_ratio, \
            mean_sq_dist_method, gaussian_distribution=None, support_trajs=None, timestep=0):
        # Get RGB-D Data & Camera Parameters
        embeddings = None
        data = self.dataset[timestep]
        color, depth, self.intrinsics, pose, instseg, embeddings, _, bg = data

        # Process Camera Parameters
        self.intrinsics = self.intrinsics[:3, :3]
        w2c = torch.linalg.inv(pose).to(self.device)

        # Setup Camera
        if timestep == 0:
            self.cam = setup_camera(
                color.shape[2],
                color.shape[1],
                self.intrinsics.cpu().numpy(),
                w2c.detach().cpu().numpy(),
                device=self.device)

        if self.config['checkpoint'] and timestep == 0:
            return

        # Get Initial Point Cloud (PyTorch CUDA Tensor)
        if self.config['data']['get_pc_jono']:
            init_pt_cld, mean3_sq_dist, bg = self.get_pointcloud_jono(
                self.intrinsics,
                mean_sq_dist_method=mean_sq_dist_method)
        else:
            init_pt_cld, mean3_sq_dist, bg = self.get_pointcloud(
                color,
                depth,
                self.intrinsics,
                w2c,
                mean_sq_dist_method=mean_sq_dist_method,
                instseg=instseg,
                embeddings=embeddings,
                bg=bg)

        # Initialize Parameters
        params, variables = self.initialize_params(
            init_pt_cld,
            self.num_frames,
            mean3_sq_dist,
            gaussian_distribution,
            bg)

        # Initialize an estimate of scene radius for Gaussian-Splatting Densification
        variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

        return w2c, params, variables

    def get_data(self):
        # Load Dataset
        print("Loading Dataset ...")
        # Poses are relative to the first frame
        self.dataset = get_data(self.config)

        # self.dataset = DataLoader(get_data(self.config), batch_size=1, shuffle=False)
        self.num_frames = self.config["data"]["num_frames"]
        if self.num_frames == -1 or self.num_frames > len(self.dataset):
            self.num_frames = len(self.dataset)

        # maybe load checkpoint
        ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
        temp_params = os.path.isfile(os.path.join(ckpt_output_dir, f"temp_params.npz"))
        final_params = os.path.isfile(os.path.join(ckpt_output_dir, f"params.npz"))

        if self.config['checkpoint'] and temp_params and not final_params:
            self.params, self.variables = load_params_ckpt(ckpt_output_dir, device=self.device)
            first_frame_w2c = self.variables['first_frame_w2c']
            self.initialize_timestep(
                self.config['scene_radius_depth_ratio'],
                self.config['mean_sq_dist_method'],
                gaussian_distribution=self.config['gaussian_distribution'],
                timestep=0)
            time_idx = min(self.num_frames, self.variables['last_time_idx'].item() + 1)
        elif self.config['checkpoint'] and final_params:
            self.params, _, _,first_frame_w2c = load_scene_data(self.config, ckpt_output_dir, device=self.device)
            self.initialize_timestep(
                self.config['scene_radius_depth_ratio'],
                self.config['mean_sq_dist_method'],
                gaussian_distribution=self.config['gaussian_distribution'],
                timestep=0)
            time_idx = self.num_frames
        else:
            self.config['checkpoint'] = False
            first_frame_w2c, self.params, self.variables = self.initialize_timestep(
                self.config['scene_radius_depth_ratio'],
                self.config['mean_sq_dist_method'],
                gaussian_distribution=self.config['gaussian_distribution'],
                timestep=0)
            self.variables['first_frame_w2c'] = first_frame_w2c
            self.variables['offset_0'] = None
            time_idx = 0

        return first_frame_w2c, time_idx
    
    def make_data_dict(self, time_idx, cam_data):
        embeddings, support_trajs = None, None
        data =  self.dataset[time_idx]
        color, depth, self.intrinsics, gt_pose, instseg, embeddings, support_trajs, bg = data
        
        # Process poses
        self.variables['gt_w2c_all_frames'].append(torch.linalg.inv(gt_pose))
        print("append", torch.linalg.inv(gt_pose))

        curr_data = {
            'im': color,
            'depth': depth,
            'id': time_idx,
            'iter_gt_w2c_list': self.variables['gt_w2c_all_frames'],
            'instseg': instseg,
            'embeddings': embeddings,
            'cam': cam_data['cam'],
            'intrinsics': cam_data['intrinsics'],
            'w2c': cam_data['w2c'],
            'support_trajs': support_trajs,
            'bg': bg}
        
        return curr_data

    def render_novel_view(self, novel_view_mode='circle'):
        self.first_frame_w2c, start_time_idx = self.get_data()
        if not 'transformed' in self.eval_dir:
            self.config['data']['do_transform'] = False
        else:
            self.config['data']['do_transform'] = True

        # Evaluate Final Parameters
        with torch.no_grad():
            eval(
                self.dataset,
                self.params,
                self.num_frames,
                self.eval_dir,
                sil_thres=self.config['add_gaussians']['sil_thres_gaussians'],
                wandb_run=self.wandb_run if self.config['use_wandb'] else None,
                variables=self.params,
                viz_config=self.config['viz'],
                get_embeddings=self.config['data']['load_embeddings'],
                remove_close=True,
                novel_view_mode=novel_view_mode,
                config=self.config)
    
    def eval(self):
        self.first_frame_w2c, start_time_idx = self.get_data()
        if not 'transformed' in self.eval_dir:
            self.config['data']['do_transform'] = False
        else:
            self.config['data']['do_transform'] = True

        # Evaluate Final Parameters
        '''with torch.no_grad():
            eval(
                self.dataset,
                self.params,
                self.num_frames,
                self.eval_dir,
                sil_thres=self.config['add_gaussians']['sil_thres_gaussians'],
                wandb_run=self.wandb_run if self.config['use_wandb'] else None,
                variables=self.params,
                viz_config=self.config['viz'],
                get_embeddings=self.config['data']['load_embeddings'],
                remove_close=self.config['remove_close'],
                config=self.config)'''

        if 'iphone' not in self.config['data']['basedir']:
            # eval traj
            with torch.no_grad():
                metrics = eval_traj(
                    self.config,
                    self.params,
                    cam=self.cam,
                    results_dir=self.eval_dir, 
                    do_transform=False if 'transformed' in self.eval_dir else True)

            with open(os.path.join(self.eval_dir, 'traj_metrics.txt'), 'w') as f:
                f.write(f"Trajectory metrics: {metrics}")
            print("Trajectory metrics: ",  metrics)

        if self.config['viz']['vis_grid']:
            vis_grid_trajs(
                self.config,
                self.params, 
                self.cam,
                results_dir=self.eval_dir,
                orig_image_size=True,
                no_bg=True if 'iphone' not in self.config['data']['basedir'] and 'rgb' not in self.config['data']['basedir'] else False)

    def rgbd_slam(self):        
        self.first_frame_w2c, start_time_idx = self.get_data()

        self.motion_mlp = MotionPredictor(device=self.device) if self.config['motion_mlp'] else None
        if self.config['base_transformations']:
            self.base_transformations = BaseTransformations(device=self.device) 
        elif self.config['base_transformations_mlp']:
            self.base_transformations = TransformationMLP(device=self.device, max_time=self.num_frames)
        else:
            self.base_transformations = None
        
        # Initialize list to keep track of Keyframes
        keyframe_list = []
        keyframe_time_indices = []
        
        # Init Variables to keep track of ground truth poses and runtimes
        self.variables['gt_w2c_all_frames'] = []
        self.tracking_obj_iter_time_sum = 0
        self.tracking_obj_iter_time_count = 0
        self.tracking_obj_frame_time_sum = 0
        self.tracking_obj_frame_time_count = 0
        self.tracking_cam_iter_time_sum = 0
        self.tracking_cam_iter_time_count = 0
        self.tracking_cam_frame_time_sum = 0
        self.tracking_cam_frame_time_count = 0
        self.supervision_flow = list()
        
        cam_data = {
            'cam': self.cam,
            'intrinsics': self.intrinsics, 
            'w2c': self.first_frame_w2c
            }

        if start_time_idx != 0:
            time_idx = start_time_idx
        print(f"Starting from time index {start_time_idx}...")
        sec_per_frame = list()
        # Iterate over Scan
        for time_idx in tqdm(range(start_time_idx, self.num_frames)):
            start = time.time()
            curr_data = self.make_data_dict(time_idx, cam_data)
            keyframe_list, keyframe_time_indices = \
                self.optimize_time(
                    time_idx,
                    curr_data,
                    keyframe_list,
                    keyframe_time_indices)
            
            # Checkpoint every iteration
            if time_idx % self.config["checkpoint_interval"] == 0 and self.config['save_checkpoints'] and time_idx != 0:
                ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
                save_params_ckpt(self.params, self.variables, ckpt_output_dir, time_idx)
        
            end = time.time()
            sec_per_frame.append(end-start)

        print(f"Took {sum(sec_per_frame)}, i.e., {sum(sec_per_frame)/len(sec_per_frame)} sec per frame on average")
        duration = sum(sec_per_frame)
        sec_per_frame = sum(sec_per_frame)/len(sec_per_frame)

        if self.config['save_checkpoints']:
            ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
            save_params_ckpt(self.params, self.variables, ckpt_output_dir, time_idx)

        self.log_time_stats()

        # Add Camera Parameters to Save them
        self.update_params_for_saving(keyframe_time_indices, duration, sec_per_frame)

        if self.config['eval_during']:
            self.log_eval_during()
        else:
            # Evaluate Final Parameters
            with torch.no_grad():
                eval(
                    self.dataset,
                    self.params,
                    self.num_frames,
                    self.eval_dir,
                    sil_thres=self.config['add_gaussians']['sil_thres_gaussians'],
                    wandb_run=self.wandb_run if self.config['use_wandb'] else None,
                    variables=self.variables,
                    viz_config=self.config['viz'],
                    get_embeddings=self.config['data']['load_embeddings'],
                    remove_close=self.config['remove_close'],
                    config=self.config)

        # Save Parameters
        save_params(self.params, self.output_dir)

        if 'iphone' not in self.config['data']['basedir']:
            # eval traj
            with torch.no_grad():
                metrics = eval_traj(
                    self.config,
                    self.params,
                    cam=self.cam,
                    results_dir=self.eval_dir, 
                    vis_trajs=self.config['viz']['vis_tracked'],
                    do_transform=False if 'transformed' in self.eval_dir else True)
                with open(os.path.join(self.eval_dir, 'traj_metrics.txt'), 'w') as f:
                    f.write(f"Trajectory metrics: {metrics}")
            print("Trajectory metrics: ",  metrics)

        # Close WandB Run
        if self.config['use_wandb']:
            self.wandb_run.log(metrics)
            wandb.finish()
                
        if self.config['viz']['vis_grid']:
            vis_grid_trajs(
                self.config,
                self.params, 
                self.cam,
                results_dir=self.eval_dir,
                orig_image_size=True,
                no_bg=True if 'iphone' not in self.config['data']['basedir'] and 'rgb' not in self.config['data']['basedir'] else False)
        
        if self.config['save_checkpoints']:
            ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
            if os.path.isfile(os.path.join(ckpt_output_dir, "temp_params.npz")):
                os.remove(os.path.join(ckpt_output_dir, "temp_params.npz"))
                os.remove(os.path.join(ckpt_output_dir, "temp_variables.npz"))

    def log_time_stats(self):
        self.tracking_obj_iter_time_count = max(self.tracking_obj_iter_time_count, 1)
        self.tracking_cam_iter_time_count = max(self.tracking_cam_iter_time_count, 1)
        self.tracking_obj_frame_time_count = max(self.tracking_obj_frame_time_count, 1)
        self.tracking_cam_frame_time_count = max(self.tracking_cam_frame_time_count, 1)
        # Compute Average Runtimes
        print(f"Average Object Tracking/Iter Time: {self.tracking_obj_iter_time_sum/self.tracking_obj_iter_time_count} s")
        print(f"Average Cam Tracking/Iter Time: {self.tracking_cam_iter_time_sum/self.tracking_cam_iter_time_count} s")
        print(f"Average Object Tracking/Frame Time: {self.tracking_obj_frame_time_sum/self.tracking_obj_frame_time_count} s")
        print(f"Average Cam Tracking/Frame Time: {self.tracking_cam_frame_time_sum/self.tracking_cam_frame_time_count} s")
        if self.config['use_wandb']:
            self.wandb_run.log({
                        "Final Stats/Average Object Tracking Iter Time (s)": self.tracking_obj_iter_time_sum/self.tracking_obj_iter_time_count,
                        "Final Stats/Average Cam Tracking Iter Time (s)": self.tracking_cam_iter_time_sum/self.tracking_cam_iter_time_count,
                        "Final Stats/Average Object Tracking Frame Time (s)": self.tracking_obj_frame_time_sum/self.tracking_obj_frame_time_count,
                        "Final Stats/Average Cam Tracking Frame Time (s)": self.tracking_cam_frame_time_sum/self.tracking_cam_frame_time_count,
                        "Final Stats/step": 1})

    def log_eval_during(self):
        # Compute Average Metrics
        psnr_list = np.array(self.psnr_list)
        rmse_list = np.array(self.rmse_list)
        l1_list = np.array(self.l1_list)
        ssim_list = np.array(self.ssim_list)
        lpips_list = np.array(self.lpips_list)

        avg_psnr = psnr_list.mean()
        avg_rmse = rmse_list.mean()
        avg_l1 = l1_list.mean()
        avg_ssim = ssim_list.mean()
        avg_lpips = lpips_list.mean()
        print("Average PSNR: {:.2f}".format(avg_psnr))
        print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
        print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
        print("Average MS-SSIM: {:.3f}".format(avg_ssim))
        print("Average LPIPS: {:.3f}".format(avg_lpips))

        if self.config['use_wandb']:
            self.wandb_run.log({"Final Stats/Average PSNR": avg_psnr, 
                            "Final Stats/Average Depth RMSE": avg_rmse,
                            "Final Stats/Average Depth L1": avg_l1,
                            "Final Stats/Average MS-SSIM": avg_ssim, 
                            "Final Stats/Average LPIPS": avg_lpips,
                            "Final Stats/step": 1})

        # Save metric lists as text files
        np.savetxt(os.path.join(self.eval_dir, "psnr.txt"), psnr_list)
        np.savetxt(os.path.join(self.eval_dir, "rmse.txt"), rmse_list)
        np.savetxt(os.path.join(self.eval_dir, "l1.txt"), l1_list)
        np.savetxt(os.path.join(self.eval_dir, "ssim.txt"), ssim_list)
        np.savetxt(os.path.join(self.eval_dir, "lpips.txt"), lpips_list)

    def update_params_for_saving(self, keyframe_time_indices, duration, sec_per_frame):
        # Add Camera Parameters to Save them
        self.params['timestep'] = self.variables['timestep']
        self.params['intrinsics'] = self.intrinsics.detach().cpu().numpy()
        self.params['w2c'] = self.first_frame_w2c.detach().cpu().numpy()
        self.params['org_width'] = self.config["data"]["desired_image_width"]
        self.params['org_height'] = self.config["data"]["desired_image_height"]
        self.params['gt_w2c_all_frames'] = torch.stack(self.variables['gt_w2c_all_frames']).detach().cpu().numpy()
        self.params['keyframe_time_indices'] = np.array(keyframe_time_indices)

        if 'gauss_ids_to_track' in self.variables.keys():
            if self.variables['gauss_ids_to_track'] == np.array(None):
                self.params['gauss_ids_to_track'] = None
            elif self.variables['gauss_ids_to_track'] is not None:
                self.params['gauss_ids_to_track'] = self.variables['gauss_ids_to_track'].cpu().numpy()
            else:
                self.params['gauss_ids_to_track'] = self.variables['gauss_ids_to_track']

        self.params['visibility'] = self.variables['visibility']
        self.params['rgb_colors'] = self.variables['rgb_colors'].float()
        self.params['log_scales'] = self.variables['log_scales'].float()
        self.params['means3D'] = self.variables['means3D'].float()
        self.params['unnorm_rotations'] = self.variables['unnorm_rotations'].float()
        self.params['logit_opacities'] = self.params['logit_opacities'].detach().clone().float()
        
        self.params['self_indices'] = self.variables['self_indices']
        self.params['neighbor_indices'] = self.variables['neighbor_indices']
        self.params['scene_radius'] = self.variables['scene_radius']
        self.params['duration'] = np.array([sec_per_frame])
        self.params['overall_duration'] = np.array([duration])

        for k, v in self.params.items():
            if isinstance(v, torch.Tensor):
                self.params[k] = v.detach().clone()

    def optimize_time(
            self,
            time_idx,
            curr_data,
            keyframe_list,
            keyframe_time_indices):
        
        # track cam
        if self.config['tracking_cam']['num_iters'] != 0 and time_idx > 0:
            self.track_cam(
                time_idx,
                curr_data)
        else:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_data['iter_gt_w2c_list'][-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                self.params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                self.params['cam_trans'][..., time_idx] = rel_w2c_tran

        # Initialize Gaussian poses for the current frame in params
        # if time_idx > 0:
        #     self.forward_propagate_gaussians(time_idx-1)
        #     self.store_vis_forward_prop(time_idx-1, time_idx, curr_data, 'forward_prop_gauss')


        # Densification
        if not self.config['densify_post'] and (time_idx+1) % self.config['add_every'] == 0 and time_idx > 0:
            self.densify(time_idx, curr_data)

        if self.config['tracking_obj']['num_iters'] != 0:
            optimizer = self.track_objects(
                time_idx,
                curr_data)
        
        # Densification
        if self.config['densify_post'] and time_idx+1 % self.config['add_every'] == 0 and time_idx > 0:
            self.densify(time_idx, curr_data)
        
        if (time_idx > 0) and self.config['densify_post'] and self.config['refine']['num_iters'] != 0:
            _ = self.track_objects(
                time_idx,
                curr_data,
                refine=True)
            
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
                    num_knn=int(self.config['kNN']/self.config['stride']),
                    dist_to_use=self.config['dist_to_use'],
                    primary_device=self.device,
                    exp_weight=self.config['exp_weight'])

        if self.config['deactivate_gaussians']['drift']:
            self.deactivate_gaussians_drift(curr_data, time_idx, optimizer)

        if (time_idx < self.num_frames-1):
            # Initialize Gaussian poses for the next frame in params
            self.forward_propagate_gaussians(time_idx)
            # orward_prop(time_idx, time_idx+1, curr_data, 'forward_prop_gauss')
            
            # initialize cam pos for next frame
            self.forward_propagate_camera(time_idx, forward_prop=self.config['tracking_cam']['forward_prop'])
            # store vis
            # self.store_vis_forward_prop(time_idx+1, time_idx+1, curr_data, 'forward_prop_cam')

        # reset hooks
        for k, p in self.params.items():
            p._backward_hooks: Dict[int, Callable] = OrderedDict()

        return keyframe_list, keyframe_time_indices

    def store_vis_forward_prop(self, time_idx, gauss_dime_idx, curr_data, save_dir):
        # store vis
        forward_rgb = self.get_depth_for_new_gauss(
            self.params,
            time_idx,
            curr_data,
            gauss_time_idx=time_idx+1,
            rgb=True)
        
        forward_rgb = torch.clamp(forward_rgb, 0, 1)
        forward_rgb = forward_rgb.detach().cpu().permute(1, 2, 0).numpy()
        os.makedirs(os.path.join(self.eval_dir, save_dir), exist_ok=True)
        cv2.imwrite(
            os.path.join(self.eval_dir, save_dir, "gs_{:04d}.png".format(time_idx)),
            cv2.cvtColor(forward_rgb*255, cv2.COLOR_RGB2BGR))
        if time_idx == self.num_frames - 2:
            make_vid(os.path.join(self.eval_dir, save_dir))

    def track_objects(
            self,
            time_idx,
            curr_data,
            refine=False):

        if not refine:
            config = self.config['tracking_obj']
            lrs = copy.deepcopy(config['lrs'])
            if self.config['use_wandb']:
                wandb_step = self.wandb_obj_tracking_step
        else:
            config = self.config['refine']
            lrs = copy.deepcopy(config['lrs'])
            if self.config['use_wandb']:
                wandb_step = self.wandb_init_refine
        
        # get instance segementation mask for Gaussians
        tracking_start_time = time.time()
        
        if self.base_transformations is not None:
            self.base_transformations.init_new_time(num_gaussians=self.params['means3D'].shape[0])
        
        # Reset Optimizer & Learning Rates for tracking
        optimizer = self.initialize_optimizer(
            self.params,
            lrs,
            tracking=True)
        optimizer.zero_grad(set_to_none=True)

        if config['take_best_candidate']:
            # Keep Track of Best Candidate Rotation & Translation
            candidate_dyno_rot = self.params['unnorm_rotations'][:, :].detach().clone()
            candidate_dyno_trans = self.params['means3D'][:, :].detach().clone()
            current_min_loss = float(1e20)

        # Tracking Optimization
        iter = 0
        num_iters_tracking = config['num_iters'] if refine or time_idx != 0 else config['num_iters_init']
        if not refine:
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Object Tracking Time Step: {time_idx}")
        else:
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Refining Time Step: {time_idx}")

        self.get_hooks(config, time_idx)

        last_loss = 1000
        early_stop_count = 0
        early_stop_eval = False
        while iter <= num_iters_tracking:
            iter_start_time = time.time()
            # Loss for current frame
            loss, losses, visible, weight = self.get_loss_dyno(
                curr_data,
                time_idx,
                num_iters=num_iters_tracking,
                iter=iter,
                config=config)

            # Backprop
            loss.backward()

            if self.config['use_wandb']:
                # Report Loss
                wandb_step = report_loss(
                    losses,
                    self.wandb_run,
                    wandb_step,
                    obj_tracking=True)

            with torch.no_grad():
                # Prune Gaussians
                if self.config['prune_densify']['prune_gaussians'] and time_idx > 0:
                    self.params, self.variables, self.support_trajs_trans, means2d = prune_gaussians(
                        self.params,
                        self.variables,
                        optimizer, 
                        iter,
                        self.config['prune_densify']['pruning_dict'],
                        time_idx,
                        self.support_trajs_trans,
                        means2d)
                    if self.config['use_wandb']:
                        self.wandb_run.log({"Tracking Object/Number of Gaussians - Pruning": self.params['means3D'].shape[0],
                                        "Mapping/step": self.wandb_mapping_step})
                # Gaussian-Splatting's Gradient-based Densification
                if self.config['prune_densify']['use_gaussian_splatting_densification']:
                    self.params, self.variables, means2d = densify(
                        self.params,
                        self.variables,
                        optimizer,
                        iter,
                        self.config['prune_densify']['densify_dict'],
                        time_idx,
                        means2d)
                    if self.config['use_wandb']:
                        self.wandb_run.log({"Tracking Object/Number of Gaussians - Densification": self.params['means3D'].shape[0],
                                        "Tracking Object/step": self.wandb_mapping_step})

            # Optimizer Update
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if config['take_best_candidate']:
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_dyno_rot = self.params['unnorm_rotations'][:, :, time_idx].detach().clone()
                        candidate_dyno_trans = self.params['means3D'][:, :, time_idx].detach().clone()
            
            # Update the runtime numbers
            iter_end_time = time.time()
            self.tracking_obj_iter_time_sum += iter_end_time - iter_start_time
            self.tracking_obj_iter_time_count += 1
            # Check if we should stop tracking
            iter += 1
            if iter % 50 == 0:
                progress_bar.update(50)

            # early stopping
            early_stop_eval, early_stop_count, last_loss = self.early_check(
                early_stop_count, last_loss, loss, early_stop_eval)
            if early_stop_eval:
                break
        
        progress_bar.close()
        if config['take_best_candidate']:
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                self.params['unnorm_rotations'] = candidate_dyno_rot
                self.params['means3D']= candidate_dyno_trans
        
        # visibility
        _, _, visible, weight = self.get_loss_dyno(
            curr_data,
            time_idx,
            num_iters=num_iters_tracking,
            iter=iter,
            config=config,
            early_stop_eval=early_stop_eval,
            last=True)
        
        optimizer.zero_grad(set_to_none=True)
        self.variables['visibility'][:, time_idx] = \
            compute_visibility(visible, weight, num_gauss=self.params['means3D'].shape[0])
        
        # update params
        for k in ['rgb_colors', 'log_scales', 'means3D', 'unnorm_rotations']:
            self.update_params(k, time_idx)
        
        if self.config['use_wandb']:
            if not refine:
                self.wandb_obj_tracking_step = wandb_step
            else:
                self.wandb_init_refine = wandb_step

        if self.config['motion_mlp']:
            with torch.no_grad():
                self.variables['means3D'][:, :, time_idx] = self.variables['means3D'][:, :, time_idx-1] + \
                    self.motion_mlp(self.variables['means3D'][:, :, time_idx-1], time_idx-1).squeeze()
        if (self.config['base_transformations'] or self.config['base_transformations']) and time_idx > 0:
            with torch.no_grad():
                means3D, unnorm_rotations, _ = self.base_transformations(
                    self.variables['means3D'][:, :, time_idx-1].to(self.device),
                    self.variables['unnorm_rotations'][:, :, time_idx-1].to(self.device),
                    time_idx=time_idx)
                self.params['means3D'][:, :] = means3D.squeeze()
                self.params['unnorm_rotations'][:, :] = unnorm_rotations.squeeze()
              
        # Update the runtime numbers
        tracking_end_time = time.time()
        self.tracking_obj_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_obj_frame_time_count += 1
        
        # update prev values for l1 losses
        self.ema_update_all_prev()

        return optimizer

    def get_hooks(self, config, time_idx):
        if config['disable_rgb_grads_old'] or config['make_grad_bg_smaller']:
            # get list to turn of grads
            to_turn_off = ['logit_opacities']           
            if not config['loss_weights']['l1_bg']:
                to_turn_off.append('bg')
            if not config['loss_weights']['l1_embeddings'] or config['make_grad_bg_smaller']:
                to_turn_off.append('embeddings')
            if not config['loss_weights']['l1_scale']:
                to_turn_off.append('log_scales')
            if not config['loss_weights']['l1_rgb']:
                to_turn_off.append('rgb_colors')
            if time_idx == 0:
                print(f"Turning off {to_turn_off}.")
            
            # remove old hooks
            if len(self.hook_list):
                for h in self.hook_list:
                    h.remove()
                self.hook_list = list()
            
            # add hooks
            for k, p in self.params.items():
                if 'cam' in k:
                    continue
                if k not in to_turn_off:
                    continue
                if config['make_grad_bg_smaller'] != 0 and k in ["embeddings"]:
                    self.hook_list.append(p.register_hook(get_hook(self.variables['timestep'] != time_idx, grad_weight=config['make_grad_bg_smaller'])))
                else:
                    self.hook_list.append(p.register_hook(get_hook(self.variables['timestep'] != time_idx)))

    def early_check(
            self,
            early_stop_count,
            last_loss,
            loss, 
            early_stop_eval,
            early_stop_time_thresh=20,
            early_stop_thresh=0.0001):
         
        if self.config['early_stop']:
            if abs(last_loss - loss.detach().clone().item()) < early_stop_thresh:
                early_stop_count += 1
                if early_stop_count == early_stop_time_thresh:
                    early_stop_eval = True
            else:
                early_stop_count = 0
            last_loss = loss.detach().clone().item()

        return early_stop_eval, early_stop_count, last_loss
    
    def ema_update_all_prev(self):
        for key in ['prev_bg', 'prev_embeddings', 'prev_rgb_colors', 'prev_log_scales']:
            try:
                self.variables[key] = self.ema_update(key, key[5:])
            except:
                if key != 'prev_embeddings' or 'embeddings' in self.params.keys():
                    self.variables[key] = self.params[key[5:]]

    def ema_update(self, key, prev_key):
        return (1-self.config['ema']) * self.params[key].detach().clone() \
            + self.config['ema'] * self.variables[prev_key]

    def update_params(self, k, time_idx):
        if k in self.variables.keys():
            if len(self.variables[k].shape) == 3:
                self.variables[k][:, :, time_idx] = self.params[k].detach().clone().squeeze()
            else:
                self.variables[k][:, time_idx] = self.params[k].detach().clone().squeeze()

    def track_cam(
            self,
            time_idx,
            curr_data):

        # get instance segementation mask for Gaussians
        tracking_start_time = time.time()

        # Reset Optimizer & Learning Rates for tracking
        optimizer = self.initialize_optimizer(
            self.params,
            self.config['tracking_cam']['lrs'],
            tracking=True)

        if self.config['tracking_cam']['take_best_candidate']:
            # Keep Track of Best Candidate Rotation & Translation
            candidate_dyno_rot = self.params['cam_unnorm_rots'][:, :, time_idx].detach().clone()
            candidate_dyno_trans = self.params['cam_trans'][:, :, time_idx].detach().clone()
            current_min_loss = float(1e20)

        # Tracking Optimization
        last_loss = 1000
        early_stop_count = 0
        iter = 0
        num_iters_tracking = self.config['tracking_cam']['num_iters']
        progress_bar = tqdm(range(num_iters_tracking), desc=f"Camera Tracking Time Step: {time_idx}")
        restarted_tracking = False
        early_stop_eval = False

        while iter <= num_iters_tracking:
            iter_start_time = time.time()
            # Loss for current frame
            loss, losses, self.variables = self.get_loss_cam(
                self.params,
                self.variables,
                curr_data,
                time_idx,
                config=self.config['tracking_cam'])

            # Backprop
            loss.backward()

            if self.config['use_wandb']:
                # Report Loss
                self.wandb_obj_tracking_step = report_loss(
                    losses,
                    self.wandb_run,
                    self.wandb_obj_tracking_step,
                    cam_tracking=True)

            # Optimizer Update
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if self.config['tracking_cam']['take_best_candidate'] and iter > 40:
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_dyno_rot = self.params['cam_unnorm_rots'][:, :, time_idx].detach().clone()
                        candidate_dyno_trans = self.params['cam_trans'][:, :, time_idx].detach().clone()

            # Update the runtime numbers
            iter_end_time = time.time()
            self.tracking_cam_iter_time_sum += iter_end_time - iter_start_time
            self.tracking_cam_iter_time_count += 1
            # Check if we should stop tracking
            iter += 1
            if iter % 50 == 0:
                progress_bar.update(50)

            if iter == num_iters_tracking and loss >= 0.5 and \
                    not restarted_tracking and self.config['tracking_cam']['restart_if_fail']:
                iter = 2
                restarted_tracking = True
                with torch.no_grad():
                    self.params['cam_unnorm_rots'][:, :, time_idx] = self.params[
                        'cam_unnorm_rots'][:, :, time_idx-1].detach().clone()
                    self.params['cam_trans'][:, :, time_idx] = self.params[
                        'cam_trans'][:, :, time_idx-1].detach().clone()

                if self.config['tracking_cam']['take_best_candidate']:
                    current_min_loss = float(1e20)
                    candidate_dyno_rot = self.params['cam_unnorm_rots'][:, :, time_idx].detach().clone()
                    candidate_dyno_trans = self.params['cam_trans'][:, :, time_idx].detach().clone()
            
            # early stopping
            early_stop_eval, early_stop_count, last_loss = self.early_check(
                early_stop_count, last_loss, loss, early_stop_eval)
            if early_stop_eval:
                break

        progress_bar.close()
        if self.config['tracking_cam']['take_best_candidate']:
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                self.params['cam_unnorm_rots'][:, :, time_idx] = candidate_dyno_rot
                self.params['cam_trans'][:, :, time_idx] = candidate_dyno_trans

        # Update the runtime numbers
        tracking_end_time = time.time()
        self.tracking_cam_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_cam_frame_time_count += 1
        
        return optimizer

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

    rgbd_slammer = RBDG_SLAMMER(experiment.config)
    if experiment.config['just_eval']:
        rgbd_slammer.eval()
    else:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))
        rgbd_slammer.rgbd_slam()


