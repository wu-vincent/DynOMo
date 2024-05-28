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

from utils.get_data import get_data, just_get_start_pix
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
    matrix_to_quaternion
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

# Make deterministic
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from utils.render_trajectories import just_render
from utils.eval_traj import eval_traj, vis_grid_trajs
from datasets.gradslam_datasets import datautils

from collections import OrderedDict
from typing import Dict, Callable

import gc

from torch.utils.data import DataLoader


class RBDG_SLAMMER():
    def __init__(self, config):
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
            bg = bg.reshape(-1, 1)

        # Compute indices of pixels
        width, height = color.shape[2], color.shape[1]
        x_grid, y_grid = torch.meshgrid(torch.arange(width).to(self.device).float(), 
                                        torch.arange(height).to(self.device).float(),
                                        indexing='xy')

        # get pixel grid into 3D
        mask = mask.reshape(-1)
        xx = (x_grid - intrinsics[0][2])/intrinsics[0][0]
        yy = (y_grid - intrinsics[1][2])/intrinsics[1][1]
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        depth_z = depth[0].reshape(-1)

        # Initialize point cloud
        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        if transform_pts:
            pix_ones = torch.ones(height * width, 1).to(self.device).float()
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
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
        point_cld = torch.cat((pts, cols), -1)
        if instseg is not None:
            instseg = torch.permute(instseg, (1, 2, 0)).reshape(-1, 1) # (C, H, W) -> (H, W, C) -> (H * W, C)
            point_cld = torch.cat((point_cld, instseg), -1)
        if embeddings is not None:
            channels = embeddings.shape[0]
            embeddings = torch.permute(embeddings, (1, 2, 0)).reshape(-1, channels) # (C, H, W) -> (H, W, C) -> (H * W, C)
            point_cld = torch.cat((point_cld, embeddings), -1)
        
        if self.config['remove_outliers_l2'] < 10:
            one_pix_diff = (
                torch.sqrt(torch.pow(self.config['remove_outliers_l2']/intrinsics[0][0], 2)) * depth_z + 
                torch.sqrt(torch.pow(0*self.config['remove_outliers_l2']/intrinsics[1][1], 2)) * depth_z)/2
            dist, _ = torch_3d_knn(point_cld[:, :3].contiguous().float(), num_knn=5)
            dist = dist[:, 1:]
            dist_mask = dist.mean(-1).clip(min=0.0000001) < one_pix_diff
            print(dist_mask.shape, dist_mask.sum())

           # mask everything
            print(point_cld.shape)
            pts = pts[dist_mask]
            point_cld = point_cld[dist_mask]
            mask = mask[dist_mask]
            instseg = instseg[dist_mask]
            mean3_sq_dist = mean3_sq_dist[dist_mask]
            bg = bg[dist_mask]
            print(point_cld.shape)

        if time_idx == 0:
            pcd = o3d.geometry.PointCloud()
            v3d = o3d.utility.Vector3dVector
            cpu_pts = pts.cpu().numpy()
            pcd.points = v3d(cpu_pts)
            o3d.io.write_point_cloud(filename=os.path.join(self.eval_dir, "init_pc.xyz"), pointcloud=pcd)
            print(os.path.join(self.eval_dir, "init_pc.xyz"))
            # quit()
            del cpu_pts

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

        if start_pix is not None:
            means2D = torch.stack((x_grid.reshape(-1), y_grid.reshape(-1)), dim=-1)
            if self.config['remove_outliers_l2'] < 10:
                means2D = means2D[dist_mask]
            if mask is not None:
                means2D = means2D[mask]
            gauss_ids_to_track = find_closest_to_start_pixels(means2D, start_pix)
            gauss_ids_to_track = torch.stack(gauss_ids_to_track)
        else:
            gauss_ids_to_track = None

        return point_cld, mean3_sq_dist, bg, gauss_ids_to_track

    def get_pointcloud_jono(
            self,
            intrinsics,
            mean_sq_dist_method="projective",
            start_pix=None):

        point_cld = torch.from_numpy(np.load(os.path.join(
            self.config['data']['basedir'],
            os.path.dirname(os.path.dirname(self.config['data']['sequence'])),
            'init_pt_cld.npz'))['data']).to(self.device)
        
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
        point_cld = point_cld[point_cld[:, 2] > 1]
        
        dist, _ = torch_3d_knn(point_cld[:, :3].contiguous().float(), num_knn=4)
        dist = dist[:, 1:]
        mean3_sq_dist = dist.mean(-1).clip(min=0.0000001)
        point_cld = point_cld[mean3_sq_dist<0.01]
        mean3_sq_dist = mean3_sq_dist[mean3_sq_dist<0.01]

        # scale_gaussian = transformed_pts[transformed_pts[:, 2] > 1][:, 2] / ((intrinsics[0][0] + intrinsics[1][1])/2)
        # mean3_sq_dist = (2*scale_gaussian)**2
        # print(mean3_sq_dist.max())
        # print(mean3_sq_dist.min())
        
        pcd = o3d.geometry.PointCloud()
        v3d = o3d.utility.Vector3dVector
        cpu_pts = point_cld[:, :3].cpu().numpy()
        pcd.points = v3d(cpu_pts)
        o3d.io.write_point_cloud(filename=os.path.join(self.eval_dir, "init_pc.xyz"), pointcloud=pcd)
        del cpu_pts

        point_cld[:, -1] = 1 - point_cld[:, -1]
        bg = point_cld[:, -1].unsqueeze(1)
        
        if False: # start_pix is not None:
            means2D = None
            gauss_ids_to_track = find_closest_to_start_pixels(means2D, start_pix)
            gauss_ids_to_track = torch.stack(gauss_ids_to_track)
        else:
            gauss_ids_to_track = None

        return point_cld, mean3_sq_dist, bg, gauss_ids_to_track

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
        print(torch.exp(log_scales))
        unnorm_rotations = torch.zeros((num_pts, 4, self.num_frames), dtype=torch.float32).to(self.device)
        unnorm_rotations[:, 0, :] = 1
        means3D = torch.zeros((num_pts, 3, self.num_frames), dtype=torch.float32).to(self.device)
        means3D[:, :, 0] = init_pt_cld[:, :3]

        params = {
                'means3D': means3D,
                'rgb_colors': init_pt_cld[:, 3:6],
                'unnorm_rotations': unnorm_rotations,
                'logit_opacities': logit_opacities,
                'log_scales': log_scales,
            }
    
        params['instseg'] = init_pt_cld[:, 6].to(self.device).long()

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

        variables = {
            'max_2D_radius': torch.zeros(means3D.shape[0]).to(self.device).float(),
            'means2D_gradient_accum': torch.zeros(means3D.shape[0], dtype=torch.float32).to(self.device),
            'denom': torch.zeros(params['means3D'].shape[0]).to(self.device).float(),
            'timestep': torch.zeros(params['means3D'].shape[0]).to(self.device).float(),
            'visibility': torch.zeros(params['means3D'].shape[0], self.num_frames).to(self.device).float()}

        instseg_mask = params["instseg"].long()
        if not self.config['use_seg_for_nn']:
            instseg_mask = torch.ones_like(instseg_mask).long().to(instseg_mask.device)

        if self.config['neighbors_init'] == 'pre' or self.config['neighbors_init'] == 'reset_first_post':
            with torch.no_grad():
                variables, to_remove = calculate_neighbors_seg(
                    params,
                    variables,
                    0,
                    instseg_mask,
                    num_knn=20,
                    dist_to_use=self.config['dist_to_use'],
                    primary_device=self.device)
        else:
            to_remove = None
        
        return params, variables, to_remove
        
    def initialize_optimizer(self, params, lrs_dict, tracking=True, attention_params=None, attention_lr=None, attention_params_bg=None, attention_lr_bg=None):
        lrs = lrs_dict
        param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
        if attention_params is not None:
            param_groups += [{'params': attention_params, 'lr': attention_lr, 'name': 'attention_emb'}]
        if attention_params_bg is not None:
            param_groups += [{'params': attention_params_bg, 'lr': attention_lr_bg, 'name': 'attention_bg'}]
        if self.motion_mlp is not None:
            param_groups += [{'params': self.motion_mlp.parameters(), 'lr': self.config['motion_lr'], 'name': 'motion_mlp'}]

        if tracking:
            return torch.optim.Adam(param_groups)
        else:
            return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    def get_loss_cam(self,
                      params,
                      variables,
                      curr_data,
                      iter_time_idx,
                      config=None):
 
        # Initialize Loss Dictionary
        losses = {}        
        variables, im, _, depth, _, mask, _, _, _, _, _, _, _, _, embeddings, bg = \
            get_renderings(
                params,
                variables,
                iter_time_idx,
                curr_data,
                config,
                track_cam=True,
                get_seg=True,
                mov_thresh=self.config['mov_thresh'],
                get_embeddings=self.config['data']['load_embeddings'])

        bg_mask = bg.detach().clone() > 0.5
        mask = bg_mask & mask.detach()

        # Depth loss
        curr_gt_depth = curr_data['depth']

        if config['use_l1']:
            losses['depth'] = l1_loss_v1(
                curr_gt_depth,
                depth,
                mask,
                reduction='mean')
        
        # RGB Loss
        curr_gt_im = curr_data['im']
        losses['im'] = l1_loss_v1(
                    curr_data['im'].permute(1, 2, 0),
                    im.permute(1, 2, 0),
                    mask.squeeze(),
                    reduction='mean')

        # EMBEDDING LOSS
        if self.dataset.load_embeddings:
            losses['embeddings'] = l1_loss_v1(
                torch.nn.functional.normalize(curr_data['embeddings'], p=2, dim=0).permute(1, 2, 0),
                torch.nn.functional.normalize(embeddings, p=2, dim=0).permute(1, 2, 0),
                mask.squeeze(),
                reduction='mean')

        weighted_losses = {k: v * config['loss_weights'][k] for k, v in losses.items()}
        loss = sum(weighted_losses.values())
        
        weighted_losses['loss'] = loss

        return loss, weighted_losses, variables


    def get_loss_dyno(self,
                      params,
                      variables,
                      curr_data,
                      iter_time_idx,
                      num_iters,
                      iter=0,
                      config=None,
                      next_params=None,
                      next_variables=None,
                      init_next=False,
                      rendered_im=None,
                      rendered_next_im=None,
                      rendered_depth=None,
                      rendered_next_depth=None,
                      neighbor_dict=None,
                      compute_next=False,
                      max_contrib_flow=10):
 

        # Initialize Loss Dictionary
        losses = {}
        depth, im, next_depth, next_im = None, None, None, None
        prev_means2d, weight, visible = self.variables['prev_means2d'], self.variables['prev_weight'], self.variables['prev_visible']
        if not init_next:
            # get renderings for current time
            variables, im, radius, depth, instseg, mask, transformed_gaussians, prev_means2d, visible, weight, motion2d, _, _, _, embeddings, bg = \
                get_renderings(
                    params,
                    variables,
                    iter_time_idx,
                    curr_data,
                    config,
                    get_seg=True,
                    disable_grads=init_next,
                    mov_thresh=self.config['mov_thresh'],
                    get_embeddings=self.config['data']['load_embeddings'],
                    motion_mlp=self.motion_mlp if iter_time_idx != 0 else None)
            
        # intiialize next point cloud for delta tracking
        elif init_next:
            _, next_im, _, next_depth, next_instseg, next_mask, next_transformed_gaussians, next_means2d, _, _, motion2d, _, _, _, next_embeddings, next_bg = \
                get_renderings(
                    params,
                    variables,
                    iter_time_idx,
                    next_data,
                    config,
                    get_seg=True,
                    mov_thresh=self.config['mov_thresh'],
                    prev_means2d=prev_means2d,
                    get_motion=config['use_flow']=='rendered',
                    get_embeddings=self.config['data']['load_embeddings'])

        if not init_next:
            mask = mask.detach()
        else:
            mask = next_mask.detach()

        # Depth loss
        if config['use_l1']:
            if not init_next:
                depth_pred, depth_gt = depth, curr_data['depth']
            else:
                next_mask = next_mask.detach()

            if True: # not config['ssmi_all_mods']:
                losses['depth'] = l1_loss_v1(depth_gt, depth_pred, mask, reduction='mean')
            else:
                losses['depth'] = (iter)/num_iters * l1_loss_v1(depth_pred, depth_gt) + (num_iters-iter)/num_iters * (
                    1.0 - calc_ssim(depth_pred, depth_gt))
        
        # RGB Loss
        if not init_next:
            rgb_pred, rgb_gt = im, curr_data['im']
        else:
            rgb_pred, rgb_gt = next_im, next_data['im']

        if init_next or config['use_sil_for_loss'] or config['ignore_outlier_depth_loss']:
            color_mask = torch.tile(mask, (3, 1, 1))
            color_mask = color_mask.detach()
            losses['im'] = torch.abs(rgb_gt - rgb_pred)[color_mask].mean()

        elif not config['calc_ssmi']:
            losses['im'] = torch.abs(rgb_gt - rgb_pred).mean()
        else:
            losses['im'] = 0.8 * l1_loss_v1(rgb_pred, rgb_gt) + 0.2 * (
                1.0 - calc_ssim(rgb_pred, rgb_gt))

        # EMBEDDING LOSS
        if self.dataset.load_embeddings and config['loss_weights']['embeddings'] != 0:
            if not init_next:
                embeddings_pred, embeddings_gt = embeddings, curr_data['embeddings']
            else:
                embeddings_pred, embeddings_gt = next_embeddings, next_data['embeddings']
            
            if not config['ssmi_all_mods']:
                embeddings_gt = torch.nn.functional.normalize(embeddings_gt, p=2, dim=0).permute(1, 2, 0)
                embeddings_pred = torch.nn.functional.normalize(embeddings_pred, p=2, dim=0).permute(1, 2, 0)
                losses['embeddings'] = l2_loss_v2(
                    embeddings_gt,
                    embeddings_pred,
                    mask.squeeze(),
                    reduction='mean')
            else:
                losses['embeddings'] = (iter)/num_iters * l1_loss_v1(embeddings_pred, embeddings_gt) + (num_iters-iter)/num_iters * (
                    1.0 - calc_ssim(embeddings_pred, embeddings_gt))

        # BG REG
        if config['bg_reg']:
            if not init_next:
                bg_pred, bg_gt = bg, curr_data['bg'].float()
            else: 
                bg_pred, bg_gt = next_bg, next_data['bg'].float()
            
            # hard foce bg
            if iter_time_idx > 0:
                is_bg = params['bg'].detach().clone().squeeze() > 0.5
                losses['bg_reg'] = l1_loss_v1(
                    params['means3D'][:, :, iter_time_idx][is_bg],
                    params['means3D'][:, :, iter_time_idx-1].detach().clone()[is_bg])

            # bg loss with mask    
            if not config['ssmi_all_mods']:
                losses['bg_loss'] = l1_loss_v1(bg_pred, bg_gt)
            else:
                losses['bg_loss'] = (iter)/num_iters * l1_loss_v1(bg_pred, bg_gt) + (num_iters-iter)/num_iters * (
                    1.0 - calc_ssim(bg_pred, bg_gt))
        
        if config['loss_weights']['l1_bg'] and iter_time_idx > 0:
            losses['l1_bg'] = l1_loss_v1(params['bg'][self.variables['timestep']<iter_time_idx], self.variables['prev_bg'])
        
        if config['loss_weights']['l1_rgb'] and iter_time_idx > 0:
            losses['l1_rgb'] = l1_loss_v1(params['rgb_colors'][self.variables['timestep']<iter_time_idx], self.variables['prev_rgb'])
        
        if config['loss_weights']['l1_embeddings'] and iter_time_idx > 0 and self.dataset.load_embeddings:
            losses['l1_embeddings'] = l1_loss_v1(params['embeddings'][self.variables['timestep']<iter_time_idx], self.variables['prev_embeddings'])

        if config['loss_weights']['l1_scale'] and iter_time_idx > 0:
            losses['l1_scale'] = l1_loss_v1(params['log_scales'][self.variables['timestep']<iter_time_idx], self.variables['prev_scales'])

        # ADD DYNO LOSSES LIKE RIGIDITY
        # DYNO LOSSES
        if config['dyno_losses'] and iter_time_idx > 0 and not init_next:
            # print(variables['timestep'])
            dyno_losses_curr, self.variables['offset_0'] = dyno_losses(
                params,
                iter_time_idx,
                transformed_gaussians,
                variables,
                self.variables['offset_0'],
                iter,
                use_iso=True,
                update_iso=True, 
                weight=config['dyno_weight'],
                post_init=self.config['neighbors_init']=='post',
                mag_iso=config['mag_iso'],
                weight_rot=config['weight_rot'],
                weight_rigid=config['weight_rigid'],
                weight_iso=config['weight_iso'])

            losses.update(dyno_losses_curr)
        elif config['dyno_losses'] and init_next:
            dyno_losses_curr, offset_0 = dyno_losses(
                params,
                iter_time_idx,
                next_transformed_gaussians,
                variables,
                self.variables['offset_0'],
                iter,
                use_iso=True,
                update_iso=False if iter_time_idx-1 != 0 else True, 
                weight=config['dyno_weight'],
                post_init=self.config['neighbors_init']=='post',
                mag_iso=config['mag_iso'],
                weight_rot=config['weight_rot'],
                weight_rigid=config['weight_rigid'],
                weight_iso=config['weight_iso'])

            if iter_time_idx-1 == 0 and iter == 0:
                self.variables['offset_0'] = offset_0
            losses.update(dyno_losses_curr)
        
        weighted_losses = {k: v * config['loss_weights'][k] for k, v in losses.items()}
        loss = sum(weighted_losses.values())

        if not init_next:
            seen = radius > 0
            variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
            variables['seen'] = seen
        weighted_losses['loss'] = loss
    
        if iter == num_iters - 1 and self.config['eval_during']:
            psnr, rmse, depth_l1, ssim, lpips, self.eval_pca = eval_during(
                curr_data,
                self.params,
                iter_time_idx,
                self.eval_dir,
                im=im.detach().clone(),
                rastered_depth=depth.detach().clone(),
                rastered_sil=mask.detach().clone(),
                rastered_bg=bg.detach().clone(),
                rendered_embeddings=embeddings.detach().clone() if embeddings is not None else embeddings,
                pca=self.eval_pca,
                sil_thres=self.config['add_gaussians']['sil_thres_gaussians'],
                wandb_run=self.wandb_run if self.config['use_wandb'] else None,
                wandb_save_qual=self.config['wandb']['eval_save_qual'],
                variables=self.variables,
                mov_thresh=self.config['mov_thresh'],
                save_pc=self.config['viz']['save_pc'],
                save_videos=self.config['viz']['save_videos'],
                get_embeddings=self.config['data']['load_embeddings'],
                vis_gt=self.config['viz']['vis_gt'],
                save_depth=self.config['viz']['vis_all'],
                save_rendered_embeddings=self.config['viz']['vis_all'],
                rendered_bg=self.config['viz']['vis_all'])
            self.psnr_list.append(psnr.cpu().numpy())
            self.rmse_list.append(rmse.cpu().numpy())
            self.l1_list.append(depth_l1.cpu().numpy())
            self.ssim_list.append(ssim.cpu().numpy())
            self.lpips_list.append(lpips)

        return loss, weighted_losses, variables, im, next_im, depth, next_depth, prev_means2d, visible, weight

    def visibility(self, visible, weight, visibility_modus='thresh', get_norm_pix_pos=False, thresh=0.5):
        w, h, contrib = visible.shape[2], visible.shape[1], visible.shape[0]

        # get max visible gauss per pix
        max_gauss_weight, max_gauss_idx = weight.detach().clone().max(dim=0)
        x_grid, y_grid = torch.meshgrid(torch.arange(w).to(self.device).long(), 
                                        torch.arange(h).to(self.device).long(),
                                        indexing='xy')
        max_gauss_id = visible.detach().clone()[
            max_gauss_idx.flatten(),
            y_grid.flatten(),
            x_grid.flatten()].int().reshape(max_gauss_idx.shape)
        
        if visibility_modus == 'max':
            visibility = torch.zeros(self.params['means3D'].shape[0], dtype=bool)
            visibility[max_gauss_id.flatten().long()] = True
            if not get_norm_pix_pos:
                return visibility

        # flattened visibility, weight, and pix id 
        # (1000 because of #contributers per tile not pix in cuda code)
        vis_pix_flat = visible.detach().clone().reshape(contrib, -1).permute(1, 0).flatten().long()
        weight_pix_flat = weight.detach().clone().reshape(contrib, -1).permute(1, 0).flatten()
        pix_id = torch.arange(w * h).unsqueeze(1).repeat(1, contrib).flatten()

        # filter above tensors where id tensor (visibility) is 0
        # by this we loos pixel Gaussian with ID 0, but should be fine
        weight_pix_flat = weight_pix_flat[vis_pix_flat!=0]
        pix_id = pix_id[vis_pix_flat!=0]
        vis_pix_flat = vis_pix_flat[vis_pix_flat!=0]

        # get overall sum of weights of one Gaussian for all pixels
        weight_sum_per_gauss = torch.zeros(self.params['means3D'].shape[0]).to(weight_pix_flat.device)
        weight_sum_per_gauss[torch.unique(vis_pix_flat)] = \
            scatter_add(weight_pix_flat, vis_pix_flat)[torch.unique(vis_pix_flat)]

        if visibility_modus == 'thresh':
            visibility = weight_sum_per_gauss > thresh
            if not get_norm_pix_pos:
                return visibility

        weighted_norm_x, weighted_norm_y = self.get_weighted_pix_pos(
            w,
            h,
            pix_id,
            vis_pix_flat,
            weight_pix_flat,
            weight_sum_per_gauss)
        
        return visibility, weighted_norm_x, weighted_norm_y
    
    def get_weighted_pix_pos(self, w, h, pix_id, vis_pix_flat, weight_pix_flat, weight_sum_per_gauss):
        # normalize weights per Gaussian for normalized pix position
        weight_pix_flat_norm = weight_pix_flat/(weight_sum_per_gauss[vis_pix_flat])
        
        x_grid, y_grid = torch.meshgrid(torch.arange(w).to(self.device).float(), 
                                        torch.arange(h).to(self.device).float(),
                                        indexing='xy')
        x_grid = (x_grid.flatten()+1)/w
        y_grid = (y_grid.flatten()+1)/h

        # initializing necessary since last gaussian might be invisible
        weighted_norm_x = torch.zeros(self.params['means3D'].shape[0]).to(weight_pix_flat.device)
        weighted_norm_y = torch.zeros(self.params['means3D'].shape[0]).to(weight_pix_flat.device)
        weighted_norm_x[torch.unique(vis_pix_flat)] = \
            scatter_add(x_grid[pix_id] * weight_pix_flat_norm, vis_pix_flat)[torch.unique(vis_pix_flat)]
        weighted_norm_y[torch.unique(vis_pix_flat)] = \
            scatter_add(y_grid[pix_id] * weight_pix_flat_norm, vis_pix_flat)[torch.unique(vis_pix_flat)]
        
        weighted_norm_x[0] = 1/w
        weighted_norm_y[0] = 1/h

        return weighted_norm_x, weighted_norm_y

    def initialize_new_params(
            self,
            new_pt_cld,
            mean3_sq_dist,
            gaussian_distribution,
            time_idx,
            cam,
            instseg,
            bg,
            width=1):
        
        variables = dict()
        num_pts = new_pt_cld.shape[0]

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
        means3D[:, :, :time_idx+1] = new_pt_cld[:, :3].unsqueeze(2).repeat(1, 1, time_idx+1)

        params = {
                'means3D': means3D,
                'rgb_colors': new_pt_cld[:, 3:6],
                'unnorm_rotations': unnorm_rotations,
                'logit_opacities': logit_opacities,
                'log_scales': log_scales
            }

        params['instseg'] = new_pt_cld[:, 6].long().to(self.device)

        if self.dataset.load_embeddings:
            params['embeddings'] = new_pt_cld[:, 7:].to(self.device).float()
        
        if bg is not None:
            params['bg'] = bg
        
        if self.config['use_seg_for_nn']:
            instseg_mask = params["instseg"].long()
            existing_instseg_mask = self.params['instseg'].long()
        else: 
            instseg_mask = torch.ones_like(instseg_mask).long().to(instseg_mask.device)
            existing_instseg_mask = torch.ones_like(self.params['instseg']).long().to(instseg_mask.device)

        if self.config['neighbors_init'] != 'post':
            with torch.no_grad():
                variables, to_remove = calculate_neighbors_seg(
                    params,
                    variables,
                    time_idx,
                    instseg_mask,
                    num_knn=20,
                    existing_params=self.params,
                    existing_instseg_mask=existing_instseg_mask,
                    dist_to_use=self.config['dist_to_use'],
                    primary_device=self.device)
        else:
            to_remove = None
            
        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).to(self.device).float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.to(self.device).float().contiguous().requires_grad_(True))
        
        return params, variables, to_remove

    def proj_means_to_2D(self, cam, time_idx, shape):
        transformed_gs_traj_3D = transform_to_frame(
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
            curr_data):
        # Silhouette Rendering
        transformed_gaussians = transform_to_frame(
            params,
            time_idx,
            gaussians_grad=False,
            camera_grad=False)

        depth_sil_rendervar, _ = transformed_params2depthplussilhouette(
            params,
            curr_data['w2c'],
            transformed_gaussians,
            time_idx,
            self.variables['timestep'],
            time_window=self.config['time_window'])

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

        depth_sil, _, _, _, _ = Renderer(raster_settings=cam)(**depth_sil_rendervar)

        return depth_sil, gt_depth

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

        os.makedirs(os.path.join(self.eval_dir, 'presence_mask'), exist_ok=True)
        silhouette_cpu = (silhouette.detach().clone()*255).cpu().numpy().astype(np.uint8)
        imageio.imwrite(
            os.path.join(self.eval_dir, 'presence_mask', 'gs_{:04d}.png'.format(time_idx)),
            silhouette_cpu)
        del silhouette_cpu

        # Check for new foreground objects by using GT depth
        render_depth = depth_sil[0, :, :]

        if self.config['add_gaussians']['use_depth_error_for_adding_gaussians']:
            depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
            non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > depth_error_factor*depth_error.median())
            non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
        else:
            non_presence_mask = non_presence_sil_mask

        # Determine non-presence mask
        # Get the new frame Gaussians based on the Silhouette
        if torch.sum(non_presence_mask) > 0:
            # Get the new pointcloud in the world frame
            curr_cam_rot = torch.nn.functional.normalize(
                self.params['cam_unnorm_rots'][..., time_idx].detach())
            curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
            curr_w2c = torch.eye(4).to(self.device).float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran

            # os.makedirs(os.path.join(self.eval_dir, 'non_presence_mask'), exist_ok=True)
            # imageio.imwrite(
            #     os.path.join(self.eval_dir, 'non_presence_mask', 'gs_{:04d}.png'.format(time_idx)),
            #     non_presence_mask.cpu().numpy().astype(np.uint8)*255)

            new_pt_cld, mean3_sq_dist, bg, _ = self.get_pointcloud(
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
            
            new_params, new_variables, to_remove = self.initialize_new_params(
                new_pt_cld,
                mean3_sq_dist,
                gaussian_distribution,
                time_idx,
                curr_data['cam'],
                curr_data['instseg'],
                bg)

            # new_params, new_variables, _ = remove_points(
            #     to_remove.squeeze(),
            #     new_params,
            #     new_variables)
            
            num_gaussians = new_params['means3D'].shape[0] + self.params['means3D'].shape[0]
        
            # make new static Gaussians parameters and update static variables
            for k, v in new_params.items():
                self.params[k] = torch.nn.Parameter(torch.cat((self.params[k], v), dim=0).requires_grad_(True))

            self.variables['means2D_gradient_accum'] = torch.zeros((num_gaussians), device=self.device).float()
            self.variables['denom'] = torch.zeros((num_gaussians), device=self.device).float()
            self.variables['max_2D_radius'] = torch.zeros((num_gaussians), device=self.device).float()

            new_timestep = time_idx*torch.ones(new_params['means3D'].shape[0], device=self.device).float()
            self.variables['timestep'] = torch.cat((self.variables['timestep'], new_timestep), dim=0)
            if 'visibility' in self.variables.keys():
                new_visibility = torch.zeros(new_params['means3D'].shape[0], self.num_frames).to(self.device).float()
                self.variables['visibility'] = torch.cat((self.variables['visibility'], new_visibility), dim=0)

            if self.config["compute_normals"]:
                self.variables['normals'] = torch.cat((self.variables['normals'], new_variables['normals']), dim=0)
            
            if self.config['neighbors_init'] != 'post':
                self.variables['self_indices'] = torch.cat((self.variables['self_indices'], new_variables['self_indices']), dim=0)
                self.variables['neighbor_indices'] = torch.cat((self.variables['neighbor_indices'], new_variables['neighbor_indices']), dim=0)
                self.variables['neighbor_weight'] = torch.cat((self.variables['neighbor_weight'], new_variables['neighbor_weight']), dim=0)
                self.variables['neighbor_weight_sm'] = torch.cat((self.variables['neighbor_weight_sm'], new_variables['neighbor_weight_sm']), dim=0)
                self.variables['neighbor_dist'] = torch.cat((self.variables['neighbor_dist'], new_variables['neighbor_dist']), dim=0)

    def make_gaussians_static(
            self,
            curr_time_idx,
            delta_rot,
            delta_tran,
            support_trajs_trans,
            determine_mov):
        
        mask = (curr_time_idx - self.variables['timestep'] >= 1).squeeze()
        with torch.no_grad():
            if delta_tran[mask].shape[0]:
                # mean translation per segment
                mean_trans = scatter_mean(delta_tran, self.params['instseg'].long(), dim=0)
                # Gaussian seg, kNN, point translation
                seg_trans = mean_trans[self.params['instseg'].long()]
                weight = self.variables['neighbor_weight_sm'].unsqueeze(1)
                kNN_trans = scatter_add(
                        weight * delta_tran[self.variables['neighbor_indices']],
                        self.variables['self_indices'], dim=0)
                point_trans = delta_tran

                # seg mean translation
                mean_trans = torch.linalg.norm(mean_trans, dim=1)

                return seg_trans, kNN_trans, point_trans
            else:
                return delta_tran, delta_tran, delta_tran
    
    def forward_propagate_camera(
            self,
            params,
            curr_time_idx,
            forward_prop=True):
        with torch.no_grad():
            if curr_time_idx > 1 and forward_prop:
                time_1 = curr_time_idx - 1 if curr_time_idx > 0 else curr_time_idx
                time_2 = curr_time_idx
                # Initialize the camera pose for the current frame based on a constant velocity model
                # Rotation
                prev_rot2 = normalize_quat(params['cam_unnorm_rots'][:, :, time_2].detach())                                                                                                                                  
                prev_rot1 = normalize_quat(params['cam_unnorm_rots'][:, :, time_1].detach())
                prev_rot1_inv = prev_rot1.clone()
                prev_rot1_inv[:, 1:] = -1 * prev_rot1_inv[:, 1:]
                delta_rot = quat_mult(prev_rot2, prev_rot1_inv) # rot from 1 -> 2
                new_rot = quat_mult(delta_rot, prev_rot2)
                params['cam_unnorm_rots'][..., curr_time_idx + 1] = new_rot.detach()
                # delta_rot_from_mat = matrix_to_quaternion(build_rotation(prev_rot2).squeeze() @ build_rotation(prev_rot1).squeeze().T)
                # new_rot_from_mat = matrix_to_quaternion(build_rotation(delta_rot_from_mat.unsqueeze(0)) @ build_rotation(prev_rot2).squeeze())

                # Translation
                prev_tran1 = params['cam_trans'][..., time_2].detach()
                prev_tran2 = params['cam_trans'][..., time_1].detach()
                new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
                params['cam_trans'][..., curr_time_idx + 1] = new_tran.detach()
            else:
                # Initialize the camera pose for the current frame
                params['cam_unnorm_rots'][..., curr_time_idx + 1] = params['cam_unnorm_rots'][..., curr_time_idx].detach()
                params['cam_trans'][..., curr_time_idx + 1] = params['cam_trans'][..., curr_time_idx].detach()
        
        return params
    
    def forward_propagate_gaussians(
            self,
            curr_time_idx,
            params,
            variables,
            mov_init_by,
            mov_static_init,
            determine_mov,
            forward_prop=True,
            support_trajs_trans=None,
            make_bg_static=False):

        # for all other timestamps moving
        with torch.no_grad():
            if forward_prop:
                # Get detla rotation and translation
                time_1 = curr_time_idx - 1 if curr_time_idx > 0 else curr_time_idx
                time_2 = curr_time_idx

                rot_2 = normalize_quat(params['unnorm_rotations'][:, :, time_2].detach())                                                                                                                                  
                rot_1 = normalize_quat(params['unnorm_rotations'][:, :, time_1].detach())
                rot_1_inv = rot_1.clone()
                rot_1_inv[:, 1:] = -1 * rot_1_inv[:, 1:]
                delta_rot = quat_mult(rot_2, rot_1_inv)

                tran_2 = params['means3D'][:, :, time_2]
                tran_1 = params['means3D'][:, :, time_1]
                delta_tran = tran_2 - tran_1                

                # make Gaussians static
                seg_trans, kNN_trans, point_trans = self.make_gaussians_static(
                    curr_time_idx, delta_rot, delta_tran, support_trajs_trans, determine_mov)
                
                # Get time mask 
                mask = (curr_time_idx - variables['timestep'] >= 0).squeeze()

                if make_bg_static:
                    mask = mask & ~params['bg'].detach().clone().squeeze()
                
                # For moving objects set new rotation and translation
                if mov_init_by == 'sparse_flow' or mov_init_by == 'sparse_flow_simple' or mov_init_by == 'im_loss' or mov_init_by == 'rendered_flow':
                    new_rot = params['unnorm_rotations'][:, :, curr_time_idx+1].detach()
                else:
                    curr_rot = normalize_quat(params['unnorm_rotations'][:, :, curr_time_idx].detach())
                    new_rot = quat_mult(delta_rot, curr_rot)[mask]
                    new_rot = torch.nn.Parameter(new_rot.to(self.device).float().contiguous().requires_grad_(True))
                params['unnorm_rotations'][mask, :, curr_time_idx + 1] = new_rot

                # use either segment mean, kNN, per point or support trajectory translation
                curr_tran = params['means3D'][:, :, curr_time_idx].detach()
                # if mov_init_by != 'sparse_flow' and curr_time_idx == 0:
                #     new_tran = (curr_tran + support_trajs_trans)[mask]
                if mov_init_by == 'seg':
                    new_tran = (curr_tran + seg_trans)[mask]
                elif mov_init_by == 'kNN':
                    new_tran = (curr_tran + kNN_trans)[mask]
                elif mov_init_by == 'per_point':
                    new_tran = (curr_tran + point_trans)[mask]
                elif mov_init_by == 'support_trajs':
                    new_tran = (curr_tran + support_trajs_trans)[mask]
                    # new_tran[:, 2] = (curr_tran + kNN_trans)[mask, 2]
                elif mov_init_by == 'sparse_flow' or mov_init_by == 'sparse_flow_simple' or mov_init_by == 'im_loss' or mov_init_by == 'rendered_flow':
                    new_tran = params['means3D'][:, :, curr_time_idx+1].detach()[mask]

                new_tran = torch.nn.Parameter(new_tran.to(self.device).float().contiguous().requires_grad_(True))
                params['means3D'][mask, :, curr_time_idx + 1] = new_tran

                # For static objects set new rotation and translation
                if make_bg_static:
                    mask = (curr_time_idx - variables['timestep'] >= 0).squeeze()
                    mask = mask & params['bg'].detach().clone().squeeze()
                    params['unnorm_rotations'][mask, :, curr_time_idx+1] = curr_rot[mask]
                    params['means3D'][mask, :, curr_time_idx+1] = curr_tran[mask]
    
        return params, variables

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
        color, depth, self.intrinsics, pose, instseg, embeddings, support_trajs, bg = data

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
            start_pix = just_get_start_pix(
                self.config,
                in_torch=True,
                normalized=False,
                h=self.cam.image_height,
                w=self.cam.image_width,
                rounded=True)
        else:
            start_pix = None
    
        if self.config['checkpoint'] and timestep == 0:
            return

        # Get Initial Point Cloud (PyTorch CUDA Tensor)
        if self.config['data']['get_pc_jono']:
            init_pt_cld, mean3_sq_dist, bg, gauss_ids_to_track = self.get_pointcloud_jono(
                self.intrinsics,
                mean_sq_dist_method=mean_sq_dist_method,
                start_pix=start_pix)
        else:
            init_pt_cld, mean3_sq_dist, bg, gauss_ids_to_track = self.get_pointcloud(
                color,
                depth,
                self.intrinsics,
                w2c,
                mean_sq_dist_method=mean_sq_dist_method,
                instseg=instseg,
                embeddings=embeddings,
                support_trajs=support_trajs,
                bg=bg,
                start_pix=start_pix)

        # Initialize Parameters
        params, variables, to_remove = self.initialize_params(
            init_pt_cld,
            self.num_frames,
            mean3_sq_dist,
            gaussian_distribution,
            bg)
        
        # new_params, new_variables, _ = remove_points(
        #     to_remove.squeeze(),
        #     new_params,
        #     new_variables)
        
        if timestep == 0:
            # store gauss ids to track
            variables['gauss_ids_to_track'] = gauss_ids_to_track

        # Initialize an estimate of scene radius for Gaussian-Splatting Densification
        variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

        return w2c, params, variables

    def re_initialize_scale(self, time_idx):
        scale_gaussian = self.params['means3D'][:, :, time_idx+1] / ((self.intrinsics[0][0] + self.intrinsics[1][1])/2)
        mean3_sq_dist = scale_gaussian**2
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
        self.params["log_scales"] = torch.nn.Parameter(log_scales.to(self.device).float().contiguous().requires_grad_(True))

    def get_data(self):
        # Load Dataset
        print("Loading Dataset ...")
        # Poses are relative to the first frame
        self.dataset = get_data(self.config)
        # self.dataset = DataLoader(get_data(self.config), batch_size=1, shuffle=False)
        self.num_frames = self.config["data"]["num_frames"]
        if self.num_frames == -1:
            self.num_frames = len(self.dataset)

        # maybe load checkpoint
        ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
        if not os.path.isfile(os.path.join(ckpt_output_dir, f"temp_params.npz")):
            self.config['checkpoint'] = False

        if self.config['checkpoint']:
            self.params, self.variables = load_params_ckpt(ckpt_output_dir, device=self.device)
            first_frame_w2c = self.variables['first_frame_w2c']
            self.initialize_timestep(
                self.config['scene_radius_depth_ratio'],
                self.config['mean_sq_dist_method'],
                gaussian_distribution=self.config['gaussian_distribution'],
                timestep=0)
            time_idx = min(self.num_frames, self.variables['last_time_idx'].item() + 1)
        else:
            first_frame_w2c, self.params, self.variables = self.initialize_timestep(
                self.config['scene_radius_depth_ratio'],
                self.config['mean_sq_dist_method'],
                gaussian_distribution=self.config['gaussian_distribution'],
                timestep=0)
            self.variables['first_frame_w2c'] = first_frame_w2c
            self.variables['offset_0'] = None
            time_idx = 0

        return first_frame_w2c, time_idx
    
    def make_data_dict(self, time_idx, gt_w2c_all_frames, cam_data):
        embeddings, support_trajs = None, None
        data =  self.dataset[time_idx]
        color, depth, self.intrinsics, gt_pose, instseg, embeddings, support_trajs, bg = data
        
        # Process poses
        gt_w2c_all_frames.append(torch.linalg.inv(gt_pose))

        curr_data = {
            'im': color,
            'depth': depth,
            'id': time_idx,
            'iter_gt_w2c_list': gt_w2c_all_frames,
            'instseg': instseg,
            'embeddings': embeddings,
            'cam': cam_data['cam'],
            'intrinsics': cam_data['intrinsics'],
            'w2c': cam_data['w2c'],
            'support_trajs': support_trajs,
            'bg': bg}
        
        return curr_data, gt_w2c_all_frames
    
    def eval(self):
        self.first_frame_w2c, start_time_idx = self.get_data()
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
                save_rendered_embeddings=self.config['viz']['vis_all'],
                rendered_bg=self.config['viz']['vis_all'])
        

    def rgbd_slam(self):        
        self.first_frame_w2c, start_time_idx = self.get_data()

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
        
        cam_data = {
            'cam': self.cam,
            'intrinsics': self.intrinsics, 
            'w2c': self.first_frame_w2c
            }

        self.variables['prev_means2d'], self.variables['prev_weight'], self.variables['prev_visible'] = None, None, None
        if start_time_idx != 0:
            time_idx = start_time_idx
        print(f"Starting from time index {start_time_idx}...")
        
        # Iterate over Scan
        for time_idx in tqdm(range(start_time_idx, self.num_frames)):
            curr_data, gt_w2c_all_frames = self.make_data_dict(time_idx, gt_w2c_all_frames, cam_data)
            # if (time_idx < self.num_frames-1):
            #     next_data, _ = self.make_data_dict(time_idx+1, gt_w2c_all_frames, cam_data)
            start = time.time()
            keyframe_list, keyframe_time_indices = \
                self.optimize_time(
                    time_idx,
                    curr_data,
                    keyframe_list,
                    keyframe_time_indices)

            if time_idx == 0:
                pcd = o3d.geometry.PointCloud()
                v3d = o3d.utility.Vector3dVector
                pts_cpu = self.params['means3D'][:, :, 0].detach().clone().cpu().numpy()
                pcd.points = v3d(pts_cpu)
                o3d.io.write_point_cloud(filename=os.path.join(self.eval_dir, "init_pc_after_update.xyz"), pointcloud=pcd)
                del pts_cpu

            # Checkpoint every iteration
            if time_idx % self.config["checkpoint_interval"] == 0 and self.config['save_checkpoints'] and time_idx != 0:
                ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
                save_params_ckpt(self.params, self.variables, ckpt_output_dir, time_idx)
        
        if self.config['save_checkpoints']:
            ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
            save_params_ckpt(self.params, self.variables, ckpt_output_dir, time_idx)

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

        if self.config['eval_during']:
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
                    wandb_save_qual=self.config['wandb']['eval_save_qual'],
                    eval_every=self.config['eval_every'],
                    variables=self.variables,
                    mov_thresh=self.config['mov_thresh'],
                    save_pc=self.config['viz']['save_pc'],
                    save_videos=self.config['viz']['save_videos'],
                    get_embeddings=self.config['data']['load_embeddings'],
                    vis_gt=self.config['viz']['vis_gt'])

        pts = self.params['means3D'][:, :, -1].detach()
        pcd = o3d.geometry.PointCloud()
        v3d = o3d.utility.Vector3dVector
        pts_cpu = pts.cpu().numpy()
        pcd.points = v3d(pts_cpu)
        path = os.path.join(self.eval_dir, "final_pc.xyz")
        o3d.io.write_point_cloud(filename=path, pointcloud=pcd)
        del pts_cpu
        print(f"Stored final pc to {path}...")

        # Add Camera Parameters to Save them
        self.params['timestep'] = self.variables['timestep']
        self.params['intrinsics'] = self.intrinsics.detach().cpu().numpy()
        self.params['w2c'] = self.first_frame_w2c.detach().cpu().numpy()
        self.params['org_width'] = self.config["data"]["desired_image_width"]
        self.params['org_height'] = self.config["data"]["desired_image_height"]
        self.params['gt_w2c_all_frames'] = []
        # for gt_w2c_tensor in gt_w2c_all_frames:
        #     self.params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
        # self.params['gt_w2c_all_frames'] = np.stack(self.params['gt_w2c_all_frames'], axis=0)
        self.params['keyframe_time_indices'] = np.array(keyframe_time_indices)

        if self.variables['gauss_ids_to_track'] == np.array(None):
            self.params['gauss_ids_to_track'] = None
        elif self.variables['gauss_ids_to_track'] is not None:
            self.params['gauss_ids_to_track'] = self.variables['gauss_ids_to_track'].cpu().numpy()
        else:
            self.params['gauss_ids_to_track'] = self.variables['gauss_ids_to_track']
        if 'visibility' in self.variables.keys():
            self.params['visibility'] = self.variables['visibility']

        # Save Parameters
        save_params(self.params, self.output_dir)

        # eval traj
        with torch.no_grad():
            metrics = eval_traj(
                self.config,
                self.params,
                cam=self.cam,
                results_dir=self.eval_dir, 
                vis_trajs=self.config['viz']['vis_tracked'],
                gauss_ids_to_track=self.variables['gauss_ids_to_track'])
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
                orig_image_size=True)
    
    def get_basis_mlps(self):
        pass


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
                    num_knn=20,
                    dist_to_use=self.config['dist_to_use'],
                    primary_device=self.device)

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
            
        if self.config['re_init_scale']:
            self.re_initialize_scale(time_idx)

        for k, p in self.params.items():
            p._backward_hooks: Dict[int, Callable] = OrderedDict()

        return keyframe_list, keyframe_time_indices

    def track_objects(
            self,
            time_idx,
            curr_data,
            init_next=False):
        if not init_next:
            config = self.config['tracking_obj']
            lrs = copy.deepcopy(config['lrs'])
            if time_idx == 0:
                lrs['means3D'] = lrs['means3D']/10
                lrs['unnorm_rotations'] = lrs['unnorm_rotations']/10
            if self.config['use_wandb']:
                wandb_step = self.wandb_obj_tracking_step
        else:
            config = self.config['init_next']
            lrs = copy.deepcopy(config['lrs'])
            if self.config['use_wandb']:
                wandb_step = self.wandb_init_next_step
        # intialize next point cloud
        # first_frame_w2c, next_params, next_variables = self.initialize_timestep(
        #     self.config['scene_radius_depth_ratio'],
        #     self.config['mean_sq_dist_method'],
        #     gaussian_distribution=self.config['gaussian_distribution'],
        #     timestep=time_idx+1)
        first_frame_w2c, next_params, next_variables = None, None, None
        
        # get instance segementation mask for Gaussians
        tracking_start_time = time.time()
        kwargs = dict()
        if config['rgb_attention_bg']:
            bg_attention_layer = DotAttentionLayer(q_dim=3, k_dim=3, v_dim=1, attention=config['attention_bg'], num_layers=1).to(self.device)
            kwargs.update({
                'attention_params': list(bg_attention_layer.parameters()) if config['attention_bg'] == 'multihead' else None,
                'attention_lr_bg': config['attention_lrs']})

        if config['rgb_attention_embeddings'] and self.config['data']['load_embeddings']:
            embedding_attention_layer = DotAttentionLayer(q_dim=3, k_dim=3, v_dim=self.params['embeddings'].shape[1], attention=config['attention'], num_layers=config['attention_layers']).to(self.device)
            kwargs.update({
                'attention_params': list(embedding_attention_layer.parameters()) if config['attention'] == 'multihead' else None,
                'attention_lr': config['attention_lrs']})
        
        # Reset Optimizer & Learning Rates for tracking
        optimizer = self.initialize_optimizer(
            self.params,
            lrs,
            tracking=True,
            **kwargs)

        if config['take_best_candidate']:
            # Keep Track of Best Candidate Rotation & Translation
            candidate_dyno_rot = self.params['unnorm_rotations'][:, :, time_idx].detach().clone()
            candidate_dyno_trans = self.params['means3D'][:, :, time_idx].detach().clone()
            current_min_loss = float(1e20)
            best_time_idx = 0

        # Tracking Optimization
        iter = 0
        best_iter = 0
        num_iters_tracking = config['num_iters'] if init_next or time_idx != 0 else config['num_iters_init']
        if not init_next:
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Object Tracking Time Step: {time_idx}")
        else:
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Initializing Time Step: {time_idx}")

        if config['disable_rgb_grads_old'] or config['make_grad_bg_smaller']:
            to_turn_off = ['logit_opacities']           
            if not config['loss_weights']['l1_bg']:
                to_turn_off.append('bg')
            if not config['loss_weights']['l1_embeddings']:
                to_turn_off.append('embeddings')
            if not config['loss_weights']['l1_scale']:
                to_turn_off.append('log_scales')
            if not config['loss_weights']['l1_rgb']:
                to_turn_off.append('rgb_colors')
            if len(self.hook_list):
                for h in self.hook_list:
                    h.remove()
                self.hook_list = list()
            for k, p in self.params.items():
                if 'cam' in k:
                    continue
                if k not in to_turn_off:
                    continue
                if config['make_grad_bg_smaller']:
                    self.hook_list.append(p.register_hook(get_hook_bg(self.variables['timestep'] != time_idx, grad_weight=0)))
                else:
                    self.hook_list.append(p.register_hook(get_hook(self.variables['timestep'] != time_idx)))

        while iter <= num_iters_tracking:
            if not ((self.config['neighbors_init'] == 'post' or self.config['neighbors_init'] == 'first_post') and time_idx == 0):
                if config['rgb_attention_embeddings'] and self.config['data']['load_embeddings']:
                    self.params['embeddings'] = embedding_attention_layer(
                        self.params['rgb_colors'],
                        self.params['rgb_colors'],
                        self.params['embeddings'],
                        self.variables['self_indices'],
                        self.variables['neighbor_indices'])
                
                if config['rgb_attention_bg']:
                    bg = bg_attention_layer(
                        self.params['rgb_colors'],
                        self.params['rgb_colors'],
                        self.params['bg'],
                        self.variables['self_indices'],
                        self.variables['neighbor_indices'])
                    self.params['bg'] = bg

            iter_start_time = time.time()
            # Loss for current frame
            loss, losses, self.variables, im, next_im, depth, next_depth, means2d, visible, weight = self.get_loss_dyno(
                self.params,
                self.variables,
                curr_data,
                time_idx,
                num_iters=num_iters_tracking,
                iter=iter,
                config=config,
                next_params=next_params,
                next_variables=next_variables,
                init_next=init_next)

            # Backprop
            loss.backward()

            if self.config['use_wandb']:
                # Report Loss
                wandb_step = report_loss(
                    losses,
                    self.wandb_run,
                    wandb_step,
                    obj_tracking=True if not init_next else False,
                    init_next=init_next)

            with torch.no_grad():
                # Prune Gaussians
                if self.config['prune_densify']['prune_gaussians'] and time_idx > 0:
                    self.params, self.variables, self.offset_0, self.support_trajs_trans = prune_gaussians(
                        self.params,
                        self.variables,
                        optimizer, 
                        iter,
                        self.config['prune_densify']['pruning_dict'],
                        time_idx,
                        self.offset_0,
                        self.support_trajs_trans)
                    if self.config['use_wandb']:
                        self.wandb_run.log({"Tracking Object/Number of Gaussians - Pruning": self.params['means3D'].shape[0],
                                        "Mapping/step": self.wandb_mapping_step})
                # Gaussian-Splatting's Gradient-based Densification
                if self.config['prune_densify']['use_gaussian_splatting_densification']:
                    self.params, self.variables = densify(
                        self.params,
                        self.variables,
                        optimizer,
                        iter,
                        self.config['prune_densify']['densify_dict'],
                        time_idx)
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
                        best_iter = iter
                        candidate_dyno_rot = self.params['unnorm_rotations'][:, :, time_idx].detach().clone()
                        candidate_dyno_trans = self.params['means3D'][:, :, time_idx].detach().clone()
                        if not init_next:
                            candidate_means2d, candidate_weight, candidate_visible = means2d.detach().clone(), weight.detach().clone(), visible.detach().clone()
            

            # Update the runtime numbers
            iter_end_time = time.time()
            self.tracking_obj_iter_time_sum += iter_end_time - iter_start_time
            self.tracking_obj_iter_time_count += 1
            # Check if we should stop tracking
            iter += 1
            progress_bar.update(1)

        visibility = self.visibility(visible, weight)
        if 'visibility' in self.variables.keys():
            self.variables['visibility'][:, time_idx] = visibility

        progress_bar.close()
        if config['take_best_candidate']:
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                self.params['unnorm_rotations'][:, :, time_idx] = candidate_dyno_rot
                self.params['means3D'][:, :, time_idx] = candidate_dyno_trans
                if not init_next:
                    self.variables['prev_means2d'], self.variables['prev_weight'], self.variables['prev_visible'] = candidate_means2d, candidate_weight, candidate_visible
        elif not init_next:
            self.variables['prev_means2d'], self.variables['prev_weight'], self.variables['prev_visible'] = means2d.detach().clone(), weight.detach().clone(), visible.detach().clone()
        
        if self.config['use_wandb']:
            if not init_next:
                self.wandb_obj_tracking_step = wandb_step
            else:
                self.wandb_init_next_step = wandb_step

        if self.config['motion_mlp']:
            with torch.no_grad():
                self.params['means3D'][:, :, time_idx] = self.params['means3D'][:, :, time_idx-1] + self.motion_mlp(self.params['means3D'][:, :, time_idx-1], time_idx-1).squeeze()

        # Update the runtime numbers
        tracking_end_time = time.time()
        self.tracking_obj_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_obj_frame_time_count += 1

        self.variables['prev_bg'] = self.params['bg'].detach().clone()
        if self.dataset.load_embeddings:
            self.variables['prev_embeddings'] = self.params['embeddings'].detach().clone()
        self.variables['prev_rgb'] = self.params['rgb_colors'].detach().clone()
        self.variables['prev_scales'] = self.params['log_scales'].detach().clone()
        
        return optimizer
    
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
        iter = 0
        best_iter = 0
        num_iters_tracking = self.config['tracking_cam']['num_iters']
        progress_bar = tqdm(range(num_iters_tracking), desc=f"Camera Tracking Time Step: {time_idx}")

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

            if self.config['tracking_cam']['take_best_candidate']:
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        best_iter = iter
                        candidate_dyno_rot = self.params['cam_unnorm_rots'][:, :, time_idx].detach().clone()
                        candidate_dyno_trans = self.params['cam_trans'][:, :, time_idx].detach().clone()

            # Update the runtime numbers
            iter_end_time = time.time()
            self.tracking_cam_iter_time_sum += iter_end_time - iter_start_time
            self.tracking_cam_iter_time_count += 1
            # Check if we should stop tracking
            iter += 1
            progress_bar.update(1)

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
        seq_experiment.config['checkpoint'] = True

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


