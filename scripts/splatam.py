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

from datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset,
    ReplicaDataset,
    ReplicaV2Dataset,
    AzureKinectDataset,
    ScannetDataset,
    Ai2thorDataset,
    Record3DDataset,
    RealsenseDataset,
    TUMDataset,
    ScannetPPDataset,
    NeRFCaptureDataset,
    DynoSplatamDataset,
    SyntheticDynoSplatamDataset,
    PointOdysseeDynoSplatamDataset,
    DavisDynoSplatamDataset,
    JonoDynoSplatamDataset
)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion, quat_mult, l2_loss_v2,
    transformed_params2instsegmov, weighted_l2_loss_v2, get_smallest_axis, mask_timestamp, get_hook, dyno_losses
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify, normalize_quat
from utils.neighbor_search import calculate_neighbors_seg

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from torch_scatter import scatter_mean, scatter_add
from utils.average_quat import averageQuaternions
import open3d as o3d
from utils.dyno_helpers import intersect_and_union2D, get_assignments2D, get_assignments3D, dbscan_filter
import imageio
import torchvision
from torchvision.transforms.functional import InterpolationMode

# from utils.trajectory_evaluation import eval_traj

# Make deterministic
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from utils.render_trajectories import just_render



def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
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


class RBDG_SLAMMER():
    def __init__(self, config):
        self.config = config
        # Print Config
        print("Loaded Config:")
        if "use_depth_loss_thres" not in config['tracking']:
            config['tracking']['use_depth_loss_thres'] = False
            config['tracking']['depth_loss_thres'] = 100000
        if "visualize_tracking_loss" not in config['tracking']:
            config['tracking']['visualize_tracking_loss'] = False
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
            self.wandb_tracking_step = 0
            self.wandb_obj_tracking_step = 0
            self.wandb_mapping_step = 0
            self.wandb_run = wandb.init(project=config['wandb']['project'],
                                group=config['wandb']['group'],
                                name=config['wandb']['name'],
                                config=config)

        # Get Device
        self.device = torch.device(config["primary_device"])
        self.offset_0 = None
    
    def resize_for_init(self, color, depth, instseg, embeddings=None, mask=None, x_grid=None, y_grid=None, reverse=False):
        if self.config['init_scale'] != 1:
            if reverse:
                scale = 1/self.config['init_scale']
            else:
                scale = self.config['init_scale']
            
            h, w = int(color.shape[1]*scale), int(color.shape[2]*scale)
            trans_nearest = torchvision.transforms.Resize(
                    (h, w), InterpolationMode.NEAREST)
            trans_bilinear = torchvision.transforms.Resize(
                    (h, w), InterpolationMode.BILINEAR)
            
            return_vals = [trans_bilinear(color), trans_nearest(depth), trans_nearest(instseg)]
            
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
            return return_vals
        
        else:
            return color, depth, instseg, embeddings, mask, x_grid, y_grid
    
    def pre_process_depth(self, depth, color):
        if self.config['zeodepth']:
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=(int(depth.shape[1]/4), int(depth.shape[2]/4)), mode='nearest')
            depth = torch.nn.functional.interpolate(
                depth, size=(color.shape[1], color.shape[2]), mode='nearest').squeeze(1)
        return depth
        
    def get_pointcloud(self, color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, mean_sq_dist_method="projective",
                   instseg=None, embeddings=None, time_idx=0, prev_instseg=None, support_trajs=None):
        
        self.pre_process_depth(depth, color)

        # Compute indices of pixels
        width, height = color.shape[2], color.shape[1]
        x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                        torch.arange(height).cuda().float(),
                                        indexing='xy')
        
        # downscale 
        color, depth, instseg, embeddings, mask, x_grid, y_grid = \
            self.resize_for_init(color, depth, instseg, embeddings, mask, x_grid, y_grid)
        width, height = color.shape[2], color.shape[1]
        mask = mask.reshape(-1)
        
        xx = (x_grid - intrinsics[0][2])/intrinsics[0][0]
        yy = (y_grid - intrinsics[1][2])/intrinsics[1][1]
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        depth_z = depth[0].reshape(-1)

        # Initialize point cloud
        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        if transform_pts:
            pix_ones = torch.ones(height * width, 1).cuda().float()
            pts4 = torch.cat((pts_cam, pix_ones), dim=1)
            c2w = torch.inverse(w2c)
            pts = (c2w @ pts4.T).T[:, :3]
        else:
            pts = pts_cam
        
        # filter background
        if self.config['remove_background']:
            background_mask = torch.permute(instseg == 255, (1, 2, 0)).reshape(-1, 1).squeeze()
        else:
            background_mask = torch.permute(torch.zeros_like(instseg, dtype=bool), (1, 2, 0)).reshape(-1, 1).squeeze()

        # Compute mean squared distance for initializing the scale of the Gaussians
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((intrinsics[0][0] + intrinsics[1][1])/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
        
        # Colorize point cloud
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
        if instseg is not None:
            instseg = torch.permute(instseg, (1, 2, 0)).reshape(-1, 1) # (C, H, W) -> (H, W, C) -> (H * W, C)
            if embeddings is not None:
                channels = embeddings.shape[0]
                embeddings = torch.permute(embeddings, (1, 2, 0)).reshape(-1, channels) # (C, H, W) -> (H, W, C) -> (H * W, C)
                point_cld = torch.cat((pts, cols, instseg, embeddings), -1)
            else:
                point_cld = torch.cat((pts, cols, instseg), -1)
        else:
            point_cld = torch.cat((pts, cols), -1)
        
        if self.config["compute_normals"]:
            pcd = o3d.geometry.PointCloud()
            v3d = o3d.utility.Vector3dVector
            pcd.points = v3d(pts.cpu().numpy())
            pcd.estimate_normals()
            normals = np.asarray(pcd.normals)
            point_cld = torch.cat((point_cld, torch.from_numpy(normals).to(point_cld.device)), -1)

        # Select points based on mask
        if mask is not None:
            mask = mask & (~background_mask)
        else:
            mask = ~background_mask

        # filter small segments
        if instseg is not None:
            uniques, counts = point_cld[:, 6][~background_mask].unique(return_counts=True)
            big_segs = torch.isin(point_cld[:, 6], uniques[counts>self.config['filter_small_segments']])

        if instseg is not None:
            mask = mask & big_segs

        # mask background points and incoming points
        point_cld = point_cld[mask]
        mean3_sq_dist = mean3_sq_dist[mask]

        return point_cld, mean3_sq_dist

    def initialize_params(self, init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution, support_trajs=None, width=1):
        num_pts = init_pt_cld.shape[0]

        logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
        if gaussian_distribution == "isotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
        elif gaussian_distribution == "anisotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
        else:
            raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")

        delta_unnorm_rotations = torch.zeros((num_pts, 4, width), dtype=torch.float32).cuda()
        delta_unnorm_rotations[:, 0, :] = 1
        delta_means3D = torch.zeros((num_pts, 3, width), dtype=torch.float32).cuda()
        
        unnorm_rotations = torch.zeros((num_pts, 4, self.num_frames), dtype=torch.float32).cuda()
        unnorm_rotations[:, 0, :] = 1
        means3D = torch.zeros((num_pts, 3, self.num_frames), dtype=torch.float32).cuda()
        means3D[:, :, 0] = init_pt_cld[:, :3]
        moving = torch.zeros(num_pts, dtype=torch.float32).cuda()

        params = {
                'means3D': means3D,
                'delta_means3D': delta_means3D,
                'rgb_colors': init_pt_cld[:, 3:6],
                'unnorm_rotations': unnorm_rotations,
                'delta_unnorm_rotations': delta_unnorm_rotations,
                'logit_opacities': logit_opacities,
                'log_scales': log_scales,
                'moving': moving
            }
    
        params['instseg'] = init_pt_cld[:, 6].cuda().long()
        if self.dataset.load_embeddings:
            params['embeddings'] = init_pt_cld[:, 7:].cuda().float()

        # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
        cam_rots = np.tile([1, 0, 0, 0], (1, 1))
        cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
        params['cam_unnorm_rots'] = cam_rots
        params['cam_trans'] = np.zeros((1, 3, num_frames))

        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

        variables = {
            'max_2D_radius': torch.zeros(means3D.shape[0]).cuda().float(),
            'means2D_gradient_accum': torch.zeros(means3D.shape[0], dtype=torch.float32).cuda(),
            'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
            'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float(),
            'moving': torch.ones(params['means3D'].shape[0]).cuda().float()}
        
        if self.config["compute_normals"] and self.dataset.load_embeddings:
            variables['normals'] = init_pt_cld[:, 8:]
        elif self.config["compute_normals"]:
            variables['normals'] = init_pt_cld[:, 7:]

        instseg_mask = params["instseg"].long()
        if self.config['use_seg_for_nn']:
            instseg_mask = instseg_mask
        else: 
            instseg_mask = torch.ones_like(instseg_mask).long().to(instseg_mask.device)
        variables = calculate_neighbors_seg(
                params, variables, 0, instseg_mask, num_knn=20,
                dist_to_use=self.config['dist_to_use'])
        
        return params, variables
        
    def initialize_optimizer(self, params, lrs_dict, tracking):
        lrs = lrs_dict
        param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
        if tracking:
            return torch.optim.Adam(param_groups)
        else:
            return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
    
    def get_renderings(self, params, variables, iter_time_idx, data, config, delta_tracking=0):
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                            gaussians_grad=True if delta_tracking==0 else False,
                                            camera_grad=False,
                                            delta=delta_tracking)
        # RGB Rendering
        rendervar, time_mask = transformed_params2rendervar(
            params,
            transformed_gaussians,
            iter_time_idx,
            first_occurance=variables['timestep'])
        rendervar['means2D'].retain_grad()
        im, radius, _, = Renderer(raster_settings=data['cam'])(**rendervar)
        variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

        # Depth & Silhouette Rendering
        depth_sil_rendervar, _ = transformed_params2depthplussilhouette(
            params,
            data['w2c'],
            transformed_gaussians,
            iter_time_idx,
            first_occurance=variables['timestep'])
        depth_sil, _, _, = Renderer(raster_settings=data['cam'])(**depth_sil_rendervar)

        # Instseg rendering
        seg_rendervar, _ = transformed_params2instsegmov(
            params,
            data['w2c'],
            transformed_gaussians,
            iter_time_idx,
            variables,
            variables['moving'] > self.moving_forward_thresh)
        instseg, _, _, = Renderer(raster_settings=data['cam'])(**seg_rendervar)

        # silouette
        presence_sil_mask = (depth_sil[1, :, :] > config['sil_thres'])
        # depth 
        depth = depth_sil[0, :, :].unsqueeze(0)
        uncertainty = (depth_sil[2, :, :].unsqueeze(0) - depth**2).detach()
        # instseg 
        instseg = instseg[2, :, :]

        # Mask with valid depth values (accounts for outlier depth values)
        nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
        mask = (data['depth'] > 0) & nan_mask

        # Mask with presence silhouette mask (accounts for empty space)
        if config['use_sil_for_loss']:
            mask = mask & presence_sil_mask

        return variables, im, radius, depth, instseg, mask, transformed_gaussians


    def get_loss_dyno(self, params, variables, curr_data, iter_time_idx, cam_tracking=False, obj_tracking=False,
                mapping=False, iter=0, config=None, next_data=None, next_params=None, next_variables=None, delta_tracking=0,
                rendered_im=None, rendered_next_im=None, rendered_depth=None, rendered_next_depth=None):
        # Initialize Loss Dictionary
        losses = {}
        
        variables, im, radius, depth, instseg, mask, transformed_gaussians = \
            self.get_renderings(params, variables, iter_time_idx, curr_data, config, delta_tracking)   
        next_variables, next_im, _, next_depth, next_instseg, next_mask, next_transformed_gaussians = \
            self.get_renderings(next_params, next_variables, iter_time_idx+1, next_data, config, delta_tracking)
        
        # Depth loss
        if delta_tracking == 0:
            curr_gt_depth, next_gt_depth = curr_data['depth'], next_data['depth']
        else:
            curr_gt_depth, next_gt_depth = next_data['depth'], curr_data['depth']
            # curr_gt_depth, next_gt_depth = rendered_next_depth, rendered_depth
            
        if config['use_l1']:
            mask = mask.detach()
            losses['depth'] = l2_loss_v2(curr_gt_depth, depth, mask, reduction='mean')
            losses['next_depth'] = l2_loss_v2(next_gt_depth, next_depth, next_mask, reduction='mean')
        
        # RGB Loss
        if delta_tracking == 0:
            curr_gt_im, next_gt_im = curr_data['im'], next_data['im']
        else:
            curr_gt_im, next_gt_im = next_data['im'], curr_data['im']
            # curr_gt_im, next_gt_im = rendered_next_im, rendered_im

        if (config['use_sil_for_loss'] or config['ignore_outlier_depth_loss']) and not config['calc_ssmi']:
            color_mask = torch.tile(mask, (3, 1, 1))
            color_mask = color_mask.detach()
            losses['im'] = torch.abs(curr_gt_im - im)[color_mask].mean()
            next_color_mask = torch.tile(mask, (3, 1, 1))
            next_color_mask = next_color_mask.detach()
            losses['next_im'] = torch.abs(next_gt_im - next_im)[next_color_mask].mean()
        elif not config['calc_ssmi']:
            losses['im'] = torch.abs(curr_gt_im - im).mean()
            losses['next_im'] = torch.abs(next_gt_im - next_im).mean()
        else:
            losses['im'] = 0.8 * l1_loss_v1(im, curr_gt_im) + 0.2 * (
                1.0 - calc_ssim(im, curr_data['im']))
            losses['next_im'] = 0.8 * l1_loss_v1(next_im, next_gt_im) + 0.2 * (
                1.0 - calc_ssim(next_im, next_gt_im))

        # SEG LOSS
        if config['use_seg_loss']:
            # intersect, union, _, _  = intersect_and_union2D(curr_data['instseg'], instseg)
            losses['instseg'] = l2_loss_v2(curr_data['instseg'], instseg, mask, reduction='mean')
            losses['next_instseg'] = l2_loss_v2(next_data['instseg'], next_instseg, mask, reduction='mean')
        
        # EMBEDDING LOSS
        if self.dataset.load_embeddings:
            pass

        # ADD DYNO LOSSES LIKE RIGIDITY
        # DYNO LOSSES
        index_time_mask = variables['timestep'][variables["self_indices"]] <= iter_time_idx - 1

        if config['dyno_losses']:
            dyno_losses_curr, self.offset_0 = dyno_losses(
                params,
                iter_time_idx,
                transformed_gaussians,
                index_time_mask,
                variables,
                self.offset_0,
                iter,
                update_iso=True)
            losses.update(dyno_losses_curr)
            
            '''
            dyno_losses_next, _ = dyno_losses(
                params,
                iter_time_idx,
                transformed_gaussians,
                index_time_mask,
                variables,
                self.offset_0,
                iter,
                update_iso=True)
            dyno_losses_next = {f'next_{k}': v for k, v in dyno_losses_next.items()}
            losses.update(dyno_losses_next)
            '''
        
        weighted_losses = {k: v * config['loss_weights'][k] for k, v in losses.items()}

        loss = sum(weighted_losses.values())
        if not mapping:
            seen = radius > 0
            variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
            variables['seen'] = seen
        weighted_losses['loss'] = loss

        return loss, weighted_losses, variables, im, next_im, depth, next_depth

    def initialize_new_params(self, new_pt_cld, mean3_sq_dist, gaussian_distribution, time_idx, cam, instseg, width=1):
        variables = dict()
        num_pts = new_pt_cld.shape[0]

        logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
        if gaussian_distribution == "isotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
        elif gaussian_distribution == "anisotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
        else:
            raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
        
        delta_unnorm_rotations = torch.zeros((num_pts, 4, width), dtype=torch.float32).cuda()
        delta_unnorm_rotations[:, 0, :] = 1
        delta_means3D = torch.zeros((num_pts, 3, width), dtype=torch.float32).cuda()

        unnorm_rotations = torch.zeros((num_pts, 4, self.num_frames), dtype=torch.float32).cuda()
        unnorm_rotations[:, 0, :] = 1
        means3D = torch.zeros((num_pts, 3, self.num_frames), dtype=torch.float32).cuda()
        means3D[:, :, :time_idx+1] = new_pt_cld[:, :3].unsqueeze(2).repeat(1, 1, time_idx+1)
        moving = torch.zeros(num_pts, dtype=torch.float32).cuda()

        params = {
                'means3D': means3D,
                'delta_means3D': delta_means3D,
                'rgb_colors': new_pt_cld[:, 3:6],
                'unnorm_rotations': unnorm_rotations,
                'delta_unnorm_rotations': delta_unnorm_rotations,
                'logit_opacities': logit_opacities,
                'log_scales': log_scales,
                'moving': moving
            }

        params['instseg'] = new_pt_cld[:, 6].long().cuda()

        if self.dataset.load_embeddings:
            params['embeddings'] = new_pt_cld[:, 7:].cuda().float()
        
        if self.config["compute_normals"] and self.dataset.load_embeddings:
            variables['normals'] = new_pt_cld[:, 8:]
        elif self.config["compute_normals"]:
            variables['normals'] = new_pt_cld[:, 7:]
        
        instseg_mask = params["instseg"].long()
        if self.config['use_seg_for_nn']:
            instseg_mask = instseg_mask
            with torch.no_grad():
                points_xy = self.proj_means_to_2D(cam, time_idx, instseg.shape)
            existing_instseg_mask = (torch.ones_like(self.params['instseg']) * 255)
            existing_instseg_mask = instseg[:, points_xy[:, 1], points_xy[:, 0]]
            existing_instseg_mask = existing_instseg_mask.long().squeeze()
            self.params['instseg'] = existing_instseg_mask
        else: 
            instseg_mask = torch.ones_like(instseg_mask).long().to(instseg_mask.device)
            existing_instseg_mask = torch.ones_like(self.params['instseg']).long().to(instseg_mask.device)
        
        variables = calculate_neighbors_seg(
                params, variables, 0, instseg_mask, num_knn=20,
                existing_params=self.params, existing_instseg_mask=existing_instseg_mask,
                dist_to_use=self.config['dist_to_use'])

        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

        return params, variables

    def proj_means_to_2D(self, cam, time_idx, shape):
        # project existing params to instseg
        points_xy = cam.projmatrix.squeeze().T.matmul(torch.cat(
            [self.params['means3D'][:, :, time_idx-1], torch.ones(self.params['means3D'].shape[0], 1).cuda()], dim=1).T)
        points_xy = points_xy / points_xy[3, :]
        points_xy = points_xy[:2].T
        points_xy[:, 0] = ((points_xy[:, 0]+1)*cam.image_width-1) * 0.5
        points_xy[:, 1] = ((points_xy[:, 1]+1)*cam.image_height-1) * 0.5
        points_xy = torch.round(points_xy).long()
        points_xy[:, 0] = torch.clip(points_xy[:, 0], min=0, max=shape[2]-1)
        points_xy[:, 1] = torch.clip(points_xy[:, 1], min=0, max=shape[1]-1)
        return points_xy.long()

    def add_new_gaussians(
            self,
            curr_data,
            sil_thres, 
            time_idx,
            mean_sq_dist_method,
            gaussian_distribution,
            prev_instseg,
            prev_color,
            params,
            variables):
        # Silhouette Rendering
        transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
        depth_sil_rendervar, _ = transformed_params2depthplussilhouette(
            params, curr_data['w2c'], transformed_gaussians, time_idx, self.variables['timestep'])
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)

        silhouette = depth_sil[1, :, :]
        non_presence_sil_mask = (silhouette < sil_thres)

        # Check for new foreground objects by using GT depth
        gt_depth = curr_data['depth'][0, :, :]
        render_depth = depth_sil[0, :, :]
        if self.config['mapping']['use_depth_error_for_adding_gaussians']:
            depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
            non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
            non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
        else:
            non_presence_mask = non_presence_sil_mask
        # Determine non-presence mask
        # Get the new frame Gaussians based on the Silhouette
        if torch.sum(non_presence_mask) > 0:
            # Get the new pointcloud in the world frame
            curr_cam_rot = torch.nn.functional.normalize(self.params['cam_unnorm_rots'][..., time_idx].detach())
            curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
            curr_w2c = torch.eye(4).cuda().float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran
            valid_depth_mask = (curr_data['depth'][0, :, :] > 0)

            # Timestep 1 there is no motion yet
            if time_idx == 1:
                non_presence_mask = torch.zeros_like(non_presence_mask, dtype=bool).to(non_presence_mask.device)

            os.makedirs(os.path.join(self.eval_dir, 'presence_mask'), exist_ok=True)
            imageio.imwrite(
                os.path.join(self.eval_dir, 'presence_mask', f'{time_idx}.png'),
                (~non_presence_mask).cpu().numpy().astype(np.uint8)*255)

            non_presence_mask = non_presence_mask & valid_depth_mask
            os.makedirs(os.path.join(self.eval_dir, 'non_presence_mask'), exist_ok=True)
            imageio.imwrite(
                os.path.join(self.eval_dir, 'non_presence_mask', f'{time_idx}.png'),
                non_presence_mask.cpu().numpy().astype(np.uint8)*255)

            new_pt_cld, mean3_sq_dist = self.get_pointcloud(
                curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                curr_w2c, mask=non_presence_mask.unsqueeze(0),
                mean_sq_dist_method=mean_sq_dist_method, instseg=curr_data['instseg'], embeddings=curr_data['embeddings'],
                time_idx=time_idx, prev_instseg=prev_instseg)
            
            print(f"Added {new_pt_cld.shape[0]} Gaussians...")
            new_params, new_variables = self.initialize_new_params(
                new_pt_cld, mean3_sq_dist, gaussian_distribution, time_idx, curr_data['cam'], curr_data['instseg'])
            num_gaussians = new_params['means3D'].shape[0] + self.params['means3D'].shape[0]
        
            # make new static Gaussians parameters and update static variables
            for k, v in new_params.items():
                self.params[k] = torch.nn.Parameter(torch.cat((self.params[k], v), dim=0).requires_grad_(True))

            self.variables['means2D_gradient_accum'] = torch.zeros((num_gaussians), device="cuda").float()
            self.variables['denom'] = torch.zeros((num_gaussians), device="cuda").float()
            self.variables['max_2D_radius'] = torch.zeros((num_gaussians), device="cuda").float()

            new_timestep = time_idx*torch.ones(new_params['means3D'].shape[0], device="cuda").float()
            self.variables['timestep'] = torch.cat((self.variables['timestep'], new_timestep), dim=0)
            new_moving = torch.ones(new_params['means3D'].shape[0]).float().cuda()
            self.variables['moving'] = torch.cat((self.variables['moving'], new_moving), dim=0)
            if self.config["compute_normals"]:
                self.variables['normals'] = torch.cat((self.variables['normals'], new_variables['normals']), dim=0)
            self.variables['self_indices'] = torch.cat((self.variables['self_indices'], new_variables['self_indices']), dim=0)
            self.variables['neighbor_indices'] = torch.cat((self.variables['neighbor_indices'], new_variables['neighbor_indices']), dim=0)
            self.variables['neighbor_weight'] = torch.cat((self.variables['neighbor_weight'], new_variables['neighbor_weight']), dim=0)
            self.variables['neighbor_dist'] = torch.cat((self.variables['neighbor_dist'], new_variables['neighbor_dist']), dim=0)

        return curr_data

    def make_gaussians_static(self, curr_time_idx, delta_rot, delta_tran, determine_mov):
        mask = (curr_time_idx - self.variables['timestep'] >= 2).squeeze()
        with torch.no_grad():
            if delta_tran[mask].shape[0]:
                # mean translation per segment
                mean_trans = scatter_mean(delta_tran, self.params['instseg'].long(), dim=0)
                
                # Gaussian seg, kNN, point translation
                seg_trans = mean_trans[self.params['instseg'].long()]
                kNN_trans = scatter_add(
                        delta_tran[self.variables['neighbor_indices']]*self.variables['neighbor_weight'],
                        self.variables['self_indices'], dim=0)
                point_trans = delta_tran

                # seg mean translation
                mean_trans = torch.linalg.norm(mean_trans, dim=1)

                # determine velocity for moving
                if determine_mov == 'seg':
                    self.variables['moving'][mask] = torch.linalg.norm(seg_trans[mask], dim=1)
                elif determine_mov == 'kNN':
                    self.variables['moving'][mask] = torch.linalg.norm(kNN_trans[mask], dim=1)
                else:
                    self.variables['moving'] = torch.linalg.norm(point_trans, dim=1)

                return seg_trans, kNN_trans, point_trans
            else:
                return delta_tran, delta_tran, delta_tran
    
    def support_trajs_to_xyz(self, support_trajs):
        # Initialize point cloud
        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        if transform_pts:
            pix_ones = torch.ones(height * width, 1).cuda().float()
            pts4 = torch.cat((pts_cam, pix_ones), dim=1)
            c2w = torch.inverse(w2c)
            pts = (c2w @ pts4.T).T[:, :3]
        else:
            pts = pts_cam
    
    def forward_propagate_gaussians(self, curr_time_idx, params, variables, mov_init_by, mov_static_init, determine_mov, forward_prop=True, support_trajs=None):
        # for timestamp 1
        mask = (curr_time_idx - variables['timestep'] == 1).squeeze()
        with torch.no_grad():
            if forward_prop:
                # Initialize the camera pose for the current frame based on a constant velocity model
                # Rotation
                new_rot = params['unnorm_rotations'][mask, :, 0]
                new_rot = torch.nn.Parameter(new_rot.cuda().float().contiguous().requires_grad_(True))
                params['unnorm_rotations'][mask, :, curr_time_idx] = new_rot

                # Translation
                new_tran = params['means3D'][mask, :, 0]
                new_tran = torch.nn.Parameter(new_tran.cuda().float().contiguous().requires_grad_(True))
                params['means3D'][mask, :, curr_time_idx] = new_tran

        # for all other timestamps moving
        with torch.no_grad():
            if forward_prop:
                # Get detla rotation
                prev_rot1 = normalize_quat(params['unnorm_rotations'][:, :, curr_time_idx-1].detach())
                prev_rot2 = normalize_quat(params['unnorm_rotations'][:, :, curr_time_idx-2].detach())
                prev_rot2_inv = prev_rot2
                prev_rot2_inv[:, 1:] = -1 * prev_rot2_inv[:, 1:]
                delta_rot = quat_mult(prev_rot1, prev_rot2_inv)

                # Get delta translation
                prev_tran1 = params['means3D'][:, :, curr_time_idx-1]
                prev_tran2 = params['means3D'][:, :, curr_time_idx-2]
                delta_tran = prev_tran1 - prev_tran2

                # make Gaussians static
                seg_trans, kNN_trans, point_trans = self.make_gaussians_static(
                    curr_time_idx, delta_rot, delta_tran, determine_mov)
                
                # Get time mask 
                mask = (curr_time_idx - variables['timestep'] > 1).squeeze()

                # add moving mask if to be used
                if mov_static_init and curr_time_idx > 1:
                    if self.config['use_rendered_moving']:
                        mask = mask & (params['moving'] > 0.5)
                    else:
                        mask = mask & (variables['moving'] > self.moving_forward_thresh)

                # For moving objects set new rotation and translation
                new_rot = quat_mult(delta_rot, prev_rot1)[mask]
                new_rot = torch.nn.Parameter(new_rot.cuda().float().contiguous().requires_grad_(True))
                params['unnorm_rotations'][mask, :, curr_time_idx] = new_rot
                # params['unnorm_rotations'][mask, :, curr_time_idx] = prev_rot1[mask]

                if mov_init_by == 'seg':
                    new_tran = (prev_tran1 + seg_trans)[mask]
                elif mov_init_by == 'kNN':
                    new_tran = (prev_tran1 + kNN_trans)[mask]
                elif mov_init_by == 'per_point':
                    new_tran = (prev_tran1 + point_trans)[mask]
                
                new_tran = torch.nn.Parameter(new_tran.cuda().float().contiguous().requires_grad_(True))
                params['means3D'][mask, :, curr_time_idx] = new_tran
                # params['means3D'][mask, :, curr_time_idx] = prev_tran1[mask]

                # For static objects set new rotation and translation
                if mov_static_init and curr_time_idx > 1:
                    mask = (curr_time_idx - variables['timestep'] > 1).squeeze()
                    if self.config['use_rendered_moving']:
                        mask = mask & ~(params['moving'] > 0.5)
                    else:
                        mask = mask & ~(variables['moving'] > self.moving_forward_thresh)
                    params['unnorm_rotations'][mask, :, curr_time_idx] = prev_rot1[mask]
                    params['means3D'][mask, :, curr_time_idx] = prev_tran1[mask]
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
        color, depth, self.intrinsics, pose, instseg = data[0], data[1], data[2], data[3], data[4]
        if self.dataset.load_embeddings:
            embeddings = data[5]
        if self.dataset.load_support_trajs:
            support_trajs = data[6]
        
        # Process Camera Parameters
        self.intrinsics = self.intrinsics[:3, :3]
        w2c = torch.linalg.inv(pose)

        # Setup Camera
        if timestep == 0:
            self.cam = setup_camera(
                color.shape[2],
                color.shape[1],
                self.intrinsics.cpu().numpy(),
                w2c.detach().cpu().numpy())

        # Get Initial Point Cloud (PyTorch CUDA Tensor)
        mask = (depth > 0) # Mask out invalid depth values
        init_pt_cld, mean3_sq_dist = self.get_pointcloud(
            color, depth, self.intrinsics, w2c, mask=mask, mean_sq_dist_method=mean_sq_dist_method,
            instseg=instseg, embeddings=embeddings, support_trajs=support_trajs)

        # Initialize Parameters
        params, variables = self.initialize_params(
            init_pt_cld, self.num_frames, mean3_sq_dist, gaussian_distribution, support_trajs)
        
        # Initialize an estimate of scene radius for Gaussian-Splatting Densification
        variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

        return w2c, params, variables

    def get_data(self):
        # Load Dataset
        print("Loading Dataset ...")
        self.dataset_config = self.config["data"]
        if "gradslam_data_cfg" not in self.dataset_config:
            self.gradslam_data_cfg = {}
            self.gradslam_data_cfg["dataset_name"] = self.dataset_config["dataset_name"]
        else:
            self.gradslam_data_cfg = load_dataset_config(self.dataset_config["gradslam_data_cfg"])
        if "ignore_bad" not in self.dataset_config:
            self.dataset_config["ignore_bad"] = False
        if "use_train_split" not in self.dataset_config:
            self.dataset_config["use_train_split"] = True

        # Poses are relative to the first frame
        self.dataset = get_dataset(
            config_dict=self.gradslam_data_cfg,
            basedir=self.dataset_config["basedir"],
            sequence=self.dataset_config["sequence"],
            start=self.dataset_config["start"],
            end=self.dataset_config["end"],
            stride=self.dataset_config["stride"],
            desired_height=self.dataset_config["desired_image_height"],
            desired_width=self.dataset_config["desired_image_width"],
            device=self.device,
            relative_pose=True,
            ignore_bad=self.dataset_config["ignore_bad"],
            use_train_split=self.dataset_config["use_train_split"],
            load_embeddings=self.dataset_config["load_embeddings"]
        )
        self.num_frames = self.dataset_config["num_frames"]
        if self.num_frames == -1:
            self.num_frames = len(self.dataset)

        first_frame_w2c, self.params, self.variables = self.initialize_timestep(
            self.config['scene_radius_depth_ratio'],
            self.config['mean_sq_dist_method'],
            gaussian_distribution=self.config['gaussian_distribution'],
            timestep=0)
                
        return first_frame_w2c
    
    def make_data_dict(self, time_idx, gt_w2c_all_frames, cam_data):
        embeddings, support_trajs = None, None
        data =  self.dataset[time_idx]
        color, depth, self.intrinsics, gt_pose = data[0], data[1], data[2], data[3]
        instseg = data[4]
        if self.dataset.load_embeddings:
            embeddings = data[5]
        if self.dataset.load_support_trajs:
            support_trajs = data[6]
        
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
            'support_trajs': support_trajs}
        
        return curr_data, gt_w2c_all_frames

    def rgbd_slam(self):        
        self.first_frame_w2c = self.get_data()
        # Initialize list to keep track of Keyframes
        keyframe_list = []
        keyframe_time_indices = []
        
        # Init Variables to keep track of ground truth poses and runtimes
        gt_w2c_all_frames = []
        self.tracking_iter_time_sum = 0
        self.tracking_iter_time_count = 0
        self.mapping_iter_time_sum = 0
        self.mapping_iter_time_count = 0
        self.tracking_obj_iter_time_sum = 0
        self.tracking_obj_iter_time_count = 0
        self.tracking_frame_time_sum = 0
        self.tracking_frame_time_count = 0
        self.mapping_frame_time_sum = 0
        self.mapping_frame_time_count = 0
        self.tracking_obj_frame_time_sum = 0
        self.tracking_obj_frame_time_count = 0
        
        cam_data = {'cam': self.cam, 'intrinsics': self.intrinsics, 
            'w2c': self.first_frame_w2c}

        # Iterate over Scan
        for time_idx in tqdm(range(self.num_frames-1)):
            curr_data, gt_w2c_all_frames = self.make_data_dict(time_idx, gt_w2c_all_frames, cam_data)
            next_data, _ = self.make_data_dict(time_idx+1, gt_w2c_all_frames, cam_data)

            print(f"Optimizing time step {time_idx}...")
            keyframe_list, keyframe_time_indices, = \
                self.optimize_time(time_idx, curr_data, keyframe_list, keyframe_time_indices, next_data)

        # Compute Average Runtimes
        if self.tracking_frame_time_sum == 0:
            self.tracking_frame_time_sum = 1
            self.tracking_frame_time_sum = 1
        if self.mapping_iter_time_count == 0:
            self.mapping_iter_time_count = 1
            self.mapping_frame_time_count = 1
        tracking_iter_time_avg = self.tracking_iter_time_sum / self.tracking_frame_time_sum
        tracking_frame_time_avg = self.tracking_frame_time_sum / self.tracking_frame_time_sum
        mapping_iter_time_avg = self.mapping_iter_time_sum / self.mapping_iter_time_count
        mapping_frame_time_avg = self.mapping_frame_time_sum / self.mapping_frame_time_count
        print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
        print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
        print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
        print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
        if self.config['use_wandb']:
            self.wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                        "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                        "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                        "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                        "Final Stats/step": 1})

        # Evaluate Final Parameters
        with torch.no_grad():
            self.variables['moving'] = self.variables['moving'] > self.moving_forward_thresh
            if self.config['use_wandb']:
                eval(self.dataset, self.params, self.num_frames, self.eval_dir, sil_thres=self.config['mapping']['sil_thres'],
                    wandb_run=self.wandb_run, wandb_save_qual=self.config['wandb']['eval_save_qual'],
                    mapping_iters=self.config['mapping']['num_iters'], add_new_gaussians=self.config['mapping']['add_new_gaussians'],
                    eval_every=self.config['eval_every'], variables=self.variables, mov_thresh=self.moving_forward_thresh,
                    use_rendered_moving=self.config['use_rendered_moving'])
            else:
                eval(self.dataset, self.params, self.num_frames, self.eval_dir, sil_thres=self.config['mapping']['sil_thres'],
                    mapping_iters=self.config['mapping']['num_iters'], add_new_gaussians=self.config['mapping']['add_new_gaussians'],
                    eval_every=self.config['eval_every'], variables=self.variables, mov_thresh=self.moving_forward_thresh,
                    use_rendered_moving=self.config['use_rendered_moving'])

        # Add Camera Parameters to Save them
        self.params['timestep'] = self.variables['timestep']
        self.params['moving'] = self.variables['moving']
        self.params['intrinsics'] = self.intrinsics.detach().cpu().numpy()
        self.params['w2c'] = self.first_frame_w2c.detach().cpu().numpy()
        self.params['org_width'] = self.config["data"]["desired_image_width"]
        self.params['org_height'] = self.config["data"]["desired_image_height"]
        self.params['gt_w2c_all_frames'] = []
        for gt_w2c_tensor in gt_w2c_all_frames:
            self.params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
        self.params['gt_w2c_all_frames'] = np.stack(self.params['gt_w2c_all_frames'], axis=0)
        self.params['keyframe_time_indices'] = np.array(keyframe_time_indices)
        
        # Save Parameters
        save_params(self.params, self.output_dir)

        # eval_traj(
        #     self.params,
        #     self.config["data"]["basedir"],
        #     os.path.basename(self.config["data"]["sequence"]))

        # Close WandB Run
        if self.config['use_wandb']:
            wandb.finish()


    def optimize_time(self, time_idx, curr_data, keyframe_list, keyframe_time_indices, next_data):
        # Initialize the camera pose for the current frame in params
        self.moving_forward_thresh = self.config['mov_thresh']

        if time_idx > 0:
            # Initialize Gaussian poses for the current frame in params
            params, variables = self.forward_propagate_gaussians(
                time_idx,
                self.params,
                self.variables,
                self.config['mov_init_by'],
                self.config['mov_static_init'],
                self.config['determine_mov'],
                support_trajs=curr_data['support_trajs'])
        else:
            params, variables = self.params, self.variables

        # Densification
        if (time_idx+1) % self.config['map_every'] == 0:
            print('Densifying!')
            curr_data = self.densify(time_idx, curr_data, params, variables, keyframe_list)

        if time_idx > 0:
            # Initialize Gaussian poses for the current frame in params
            params, variables = self.forward_propagate_gaussians(
                time_idx,
                self.params,
                self.variables,
                self.config['mov_init_by'],
                self.config['mov_static_init'],
                self.config['determine_mov'],
                support_trajs=curr_data['support_trajs'])

        # Select key frames 
        selected_keyframes = self.select_keyframes(time_idx, keyframe_list, curr_data['depth'])
    
        if self.config['tracking']['num_iters'] != 0:
            delta_optim_in = self.track_objects(time_idx, curr_data, time_idx, next_data)
        
        if self.config['delta_optim']['num_iters'] != 0:
            self.delta_optim(delta_optim_in)

        # KeyFrame-based Mapping
        num_mapping_iters = self.config['mapping']['num_iters'] # if time_idx >= 2 else self.config['tracking']['num_iters']
        if (num_mapping_iters != 0 and (time_idx+1) % self.config['map_every'] == 0):
            # Reset Optimizer & Learning Rates for Full Map Optimization
            self.map(num_mapping_iters, time_idx, selected_keyframes, keyframe_list, curr_data)
        
        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % self.config['keyframe_every'] == 0) or \
                    (time_idx == self.num_frames-2)) and (not torch.isinf(curr_data['iter_gt_w2c_list'][-1]).any()) and (not torch.isnan(curr_data['iter_gt_w2c_list'][-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {
                    'id': time_idx,
                    'est_w2c': curr_w2c,
                    'depth': curr_data['depth'],
                    'instseg': curr_data['instseg'],
                    'embeddings': curr_data['embeddings']}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % self.config["checkpoint_interval"] == 0 and self.config['save_checkpoints']:
            ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
            save_params_ckpt(self.params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if self.config['use_wandb']:
            self.wandb_time_step += 1

        torch.cuda.empty_cache()

        return keyframe_list, keyframe_time_indices

    def track_objects(self, time_idx, curr_data, iter_time_idx, next_data):
        # intialize next point cloud
        first_frame_w2c, next_params, next_variables = self.initialize_timestep(
            self.config['scene_radius_depth_ratio'],
            self.config['mean_sq_dist_method'],
            gaussian_distribution=self.config['gaussian_distribution'],
            timestep=time_idx+1)
        
        # get instance segementation mask for Gaussians
        tracking_start_time = time.time()

        # Reset Optimizer & Learning Rates for tracking
        optimizer = self.initialize_optimizer(self.params, self.config['tracking_obj']['lrs'], tracking=True)

        if self.config['tracking_obj']['take_best_candidate']:
            # Keep Track of Best Candidate Rotation & Translation
            candidate_dyno_rot = self.params['unnorm_rotations'][:, :, time_idx].detach().clone()
            candidate_dyno_trans = self.params['means3D'][:, :, time_idx].detach().clone()
            current_min_loss = float(1e20)
            best_time_idx = 0

        # Tracking Optimization
        iter = 0
        best_iter = 0
        num_iters_tracking = self.config['tracking']['num_iters']
        progress_bar = tqdm(range(num_iters_tracking), desc=f"Object Tracking Time Step: {time_idx}")
        
        while iter <= num_iters_tracking:
            iter_start_time = time.time()
            # Loss for current frame
            loss, losses, self.variables, im, next_im, depth, next_depth = self.get_loss_dyno(self.params, self.variables, curr_data, iter_time_idx, \
                cam_tracking=False, obj_tracking=True, mapping=False, iter=iter, config=self.config['tracking_obj'],
                next_data=next_data, next_params=next_params, next_variables=next_variables)

            if self.config['tracking_obj']['disable_rgb_grads_old']:
                for k, p in self.params.items():
                    if 'cam' in k:
                        continue
                    if k not in ['rgb_colors', 'logit_opacities', 'log_scales'] :
                        continue
                    p.register_hook(get_hook(self.variables['timestep'] != iter_time_idx))
            
            # Backprop
            loss.backward()

            if self.config['use_wandb']:
                # Report Loss
                self.wandb_obj_tracking_step = report_loss(losses, self.wandb_run, self.wandb_obj_tracking_step, obj_tracking=True, params=self.params, grads=None)

            with torch.no_grad():
                # Prune Gaussians
                if self.config['tracking_obj']['prune_gaussians']:
                    self.params, self.variables = prune_gaussians(self.params, self.variables, optimizer, iter, self.config['tracking_obj']['pruning_dict'], iter_time_idx)
                    if self.config['use_wandb']:
                        self.wandb_run.log({"Tracking Object/Number of Gaussians - Pruning": self.params['means3D'].shape[0],
                                        "Mapping/step": self.wandb_mapping_step})
                # Gaussian-Splatting's Gradient-based Densification
                if self.config['tracking_obj']['use_gaussian_splatting_densification']:
                    self.params, self.variables = densify(self.params, self.variables, optimizer, iter, self.config['tracking_obj']['densify_dict'], iter_time_idx)
                    if self.config['use_wandb']:
                        self.wandb_run.log({"Tracking Object/Number of Gaussians - Densification": self.params['means3D'].shape[0],
                                        "Tracking Object/step": self.wandb_mapping_step})

            # Optimizer Update
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if self.config['tracking_obj']['take_best_candidate']:
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        best_iter = iter
                        candidate_dyno_rot = self.params['unnorm_rotations'][:, :, time_idx].detach().clone()
                        candidate_dyno_trans = self.params['means3D'][:, :, time_idx].detach().clone()

            # Update the runtime numbers
            iter_end_time = time.time()
            self.tracking_obj_iter_time_sum += iter_end_time - iter_start_time
            self.tracking_obj_frame_time_sum += 1
            # Check if we should stop tracking
            iter += 1
            progress_bar.update(1)

        progress_bar.close()
        if self.config['tracking_obj']['take_best_candidate']:
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                self.params['unnorm_rotations'][:, :, time_idx] = candidate_dyno_rot
                self.params['means3D'][:, :, time_idx] = candidate_dyno_trans
                print(f'Best candidate at iteration {best_iter}!')

        # Update the runtime numbers
        tracking_end_time = time.time()
        self.tracking_obj_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_obj_frame_time_sum += 1

        delta_optim_in = (time_idx, curr_data, iter_time_idx, next_data, im.clone().detach(), next_im.clone().detach(), depth.clone().detach(), next_depth.clone().detach(), next_params, next_variables)
        return delta_optim_in
    
    def delta_optim(self, delta_optim_in):
        time_idx, curr_data, iter_time_idx, next_data, im, next_im, depth, next_depth, next_params, next_variables = delta_optim_in
        # get instance segementation mask for Gaussians
        tracking_start_time = time.time()

        # Reset Optimizer & Learning Rates for tracking
        optimizer = self.initialize_optimizer(self.params, self.config['tracking_obj']['lrs'], tracking=True)

        if self.config['tracking_obj']['take_best_candidate']:
            # Keep Track of Best Candidate Rotation & Translation
            candidate_delta_rot = self.params['delta_unnorm_rotations'][:, :, time_idx].detach().clone()
            candidate_delta_trans = self.params['delta_means3D'][:, :, time_idx].detach().clone()
            current_min_loss = float(1e20)
            best_time_idx = 0

        # Tracking Optimization
        iter = 0
        best_iter = 0
        num_iters_tracking = self.config['tracking']['num_iters']
        progress_bar = tqdm(range(num_iters_tracking), desc=f"Object Delta Tracking Time Step: {time_idx}")
        
        while iter <= num_iters_tracking:
            iter_start_time = time.time()
            # Loss for current frame
            loss, losses, self.variables, im, next_im, depth, next_depth = self.get_loss_dyno(self.params, self.variables, curr_data, iter_time_idx, \
                cam_tracking=False, obj_tracking=True, mapping=False, delta_tracking=1, iter=iter, config=self.config['delta_optim'],
                next_data=next_data, next_params=next_params, next_variables=next_variables, rendered_im=im, rendered_next_im=next_im, rendered_depth=depth, rendered_next_depth=next_depth)

            if self.config['tracking_obj']['disable_rgb_grads_old']:
                for k, p in self.params.items():
                    if 'cam' in k:
                        continue
                    if k not in ['rgb_colors', 'logit_opacities', 'log_scales'] :
                        continue
                    p.register_hook(get_hook(self.variables['timestep'] != iter_time_idx))
            
            # Backprop
            loss.backward()

            if self.config['use_wandb']:
                # Report Loss
                self.wandb_obj_tracking_step = report_loss(losses, self.wandb_run, self.wandb_obj_tracking_step, delta_optim=True, params=self.params, grads=None)

            with torch.no_grad():
                # Prune Gaussians
                if self.config['tracking_obj']['prune_gaussians']:
                    self.params, self.variables = prune_gaussians(self.params, self.variables, optimizer, iter, self.config['tracking_obj']['pruning_dict'], iter_time_idx)
                    if self.config['use_wandb']:
                        self.wandb_run.log({"Tracking Object/Number of Gaussians - Pruning": self.params['means3D'].shape[0],
                                        "Mapping/step": self.wandb_mapping_step})
                # Gaussian-Splatting's Gradient-based Densification
                if self.config['tracking_obj']['use_gaussian_splatting_densification']:
                    self.params, self.variables = densify(self.params, self.variables, optimizer, iter, self.config['tracking_obj']['densify_dict'], iter_time_idx)
                    if self.config['use_wandb']:
                        self.wandb_run.log({"Tracking Object/Number of Gaussians - Densification": self.params['means3D'].shape[0],
                                        "Tracking Object/step": self.wandb_mapping_step})

            # Optimizer Update
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if self.config['tracking_obj']['take_best_candidate']:
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        best_iter = iter
                        candidate_dyno_rot = self.params['unnorm_rotations'][:, :, time_idx].detach().clone()
                        candidate_dyno_trans = self.params['means3D'][:, :, time_idx].detach().clone()

            # Update the runtime numbers
            iter_end_time = time.time()
            self.tracking_obj_iter_time_sum += iter_end_time - iter_start_time
            self.tracking_obj_frame_time_sum += 1
            # Check if we should stop tracking
            iter += 1
            progress_bar.update(1)

        progress_bar.close()
        if self.config['tracking_obj']['take_best_candidate']:
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                self.params['unnorm_rotations'][:, :, time_idx] = candidate_dyno_rot
                self.params['means3D'][:, :, time_idx] = candidate_dyno_trans
                print(f'Best candidate at iteration {best_iter}!')

        # Update the runtime numbers
        tracking_end_time = time.time()
        self.tracking_obj_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_obj_frame_time_sum += 1

    def densify(self, time_idx, curr_data, params, variables, keyframe_list):
        if self.config['mapping']['add_new_gaussians'] and time_idx >= 1:
            if len(keyframe_list) == 0:
                densify_prev_data = curr_data
            else:
                densify_prev_data = keyframe_list[time_idx-1]

            # Add new Gaussians to the scene based on the Silhouette
            curr_data = self.add_new_gaussians(curr_data, 
                                    self.config['mapping']['sil_thres_gaussians'],
                                    time_idx,
                                    self.config['mean_sq_dist_method'],
                                    self.config['gaussian_distribution'],
                                    densify_prev_data['instseg'],
                                    densify_prev_data['color'],
                                    params, 
                                    variables)

            post_num_pts = self.params['means3D'].shape[0]
            if self.config['use_wandb']:
                self.wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                "Mapping/step": self.wandb_time_step})
        return curr_data
        
    def select_keyframes(self, time_idx, keyframe_list, depth):
        with torch.no_grad():
            # Get the current estimated rotation & translation
            curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., time_idx].detach())
            curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
            curr_w2c = torch.eye(4).cuda().float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran
            # Select Keyframes for Mapping
            num_keyframes = self.config['mapping_window_size']-2
            selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, self.intrinsics, keyframe_list[:-1], num_keyframes)
            selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
            # if len(keyframe_list) > 0:
            #     # Add last keyframe to the selected keyframes
            #     selected_time_idx.append(keyframe_list[-1]['id'])
            #     selected_keyframes.append(len(keyframe_list)-1)
            # Add current frame to the selected keyframes
            selected_time_idx.append(time_idx)
            selected_keyframes.append(-1)
            # Print the selected keyframes
            print(f"Selected Keyframes at Frame {time_idx}: {selected_time_idx}")
        
        return selected_keyframes

    def map(self, num_iters_mapping, time_idx, selected_keyframes, keyframe_list, curr_data):
       
        optimizer = self.initialize_optimizer(self.params, self.config['mapping']['lrs'], tracking=False) 
        mapping_start_time = time.time()

        # Mapping
        if num_iters_mapping > 0:
            progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")

        for iter in range(num_iters_mapping):
            iter_start_time = time.time()
            # Randomly select a frame until current time step amongst keyframes
            rand_idx = np.random.randint(0, len(selected_keyframes))
            selected_rand_keyframe_idx = selected_keyframes[rand_idx]
            if selected_rand_keyframe_idx == -1:
                # Use Current Frame Data
                iter_time_idx = time_idx
                iter_color = curr_data['im']
                iter_depth = curr_data['depth']
                iter_instseg = curr_data['instseg']
                iter_embeddings = curr_data['embeddings']
            else:
                # Use Keyframe Data
                iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                iter_instseg = keyframe_list[selected_rand_keyframe_idx]['instseg']
                iter_embeddings = keyframe_list[selected_rand_keyframe_idx]['embeddings']

            iter_gt_w2c = curr_data['iter_gt_w2c_list'][:iter_time_idx+1]

            iter_data = {
                'cam': curr_data['cam'],
                'im': iter_color,
                'depth': iter_depth,
                'id': iter_time_idx, 
                'intrinsics': self.intrinsics,
                'w2c': self.first_frame_w2c,
                'iter_gt_w2c_list': iter_gt_w2c,
                'instseg': iter_instseg,
                'embeddings': iter_embeddings}

            # Loss for current frame
            loss, losses, self.variables = self.get_loss_dyno(self.params, self.variables, iter_data, iter_time_idx, \
                    cam_tracking=False, obj_tracking=False, mapping=True, iter=iter, config=self.config['mapping'])

            if self.config['use_wandb']:
                # Report Loss
                self.wandb_mapping_step = report_loss(losses, self.wandb_run, self.wandb_mapping_step, mapping=True)
            
            if (iter+1)%self.config['mapping']['batch_size'] == 0:

                if self.config['mapping']['disable_rgb_grads_old']:
                    for k, p in self.params.items():
                        if k not in ['rgb_colors', 'logit_opacities', 'log_scales'] :
                            continue
                        p.register_hook(
                            get_hook(self.variables['timestep'] != iter_time_idx))
                
                # Backprop
                loss.backward()
                
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    # Report Progress
                    if self.config['report_iter_progress']:
                        if self.config['use_wandb']:
                            report_progress(self.params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=self.config['mapping']['sil_thres'], 
                                            wandb_run=self.wandb_run, wandb_step=self.wandb_mapping_step, wandb_save_qual=self.config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(self.params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=self.config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                self.mapping_iter_time_sum += iter_end_time - iter_start_time
                self.mapping_iter_time_count += 1

        # LOGGING
        if num_iters_mapping > 0:
            progress_bar.close()

        # Update the runtime numbers
        mapping_end_time = time.time()
        self.mapping_frame_time_sum += mapping_end_time - mapping_start_time
        self.mapping_frame_time_count += 1


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

    rgbd_slammer = RBDG_SLAMMER(experiment.config)
    rgbd_slammer.rgbd_slam()

    just_render(experiment.config, results_dir)

