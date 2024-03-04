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
    SyntheticDynoSplatamDataset
)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion, quat_mult, l2_loss_v2,
    transformed_params2instsegmov, weighted_l2_loss_v2
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify, normalize_quat
from utils.neighbor_search import calculate_neighbors_seg

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from torch_scatter import scatter_mean
from utils.average_quat import averageQuaternions

# Make deterministic
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)


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
            self.wandb_mapping_step = 0
            self.wandb_run = wandb.init(project=config['wandb']['project'],
                                group=config['wandb']['group'],
                                name=config['wandb']['name'],
                                config=config)

        # Get Device
        self.device = torch.device(config["primary_device"])

        # is dynosplatam
        self.dataset_config = self.config["data"]
        if "gradslam_data_cfg" not in self.config["data"]:
            dataset_name = self.config["data"]["dataset_name"]
        else:
            dataset_name = load_dataset_config(self.dataset_config["gradslam_data_cfg"])["dataset_name"]

        if dataset_name.lower() in ["dynosplatam", "synthetic"]:
            self.dynosplatam = True
        else:
            self.dynosplatam = False
        
        self.per_point_params = ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales', 'instseg']

    def get_pointcloud(self, color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective",
                   instseg=None, embeddings=None):
        width, height = color.shape[2], color.shape[1]
        CX = intrinsics[0][2]
        CY = intrinsics[1][2]
        FX = intrinsics[0][0]
        FY = intrinsics[1][1]

        # Compute indices of pixels
        x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                        torch.arange(height).cuda().float(),
                                        indexing='xy')
        xx = (x_grid - CX)/FX
        yy = (y_grid - CY)/FY
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

        # Compute mean squared distance for initializing the scale of the Gaussians
        if compute_mean_sq_dist:
            if mean_sq_dist_method == "projective":
                # Projective Geometry (this is fast, farther -> larger radius)
                scale_gaussian = depth_z / ((FX + FY)/2)
                mean3_sq_dist = scale_gaussian**2
            else:
                raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
        
        # Colorize point cloud
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
        if instseg is not None:
            instseg = torch.permute(instseg, (1, 2, 0)).reshape(-1, 1) # (C, H, W) -> (H, W, C) -> (H * W, C)
            if embeddings is not None:
                embeddings = torch.permute(embeddings, (1, 2, 0)).reshape(-1, 1) # (C, H, W) -> (H, W, C) -> (H * W, C)
                point_cld = torch.cat((pts, cols, instseg, embeddings), -1)
            else:
                point_cld = torch.cat((pts, cols, instseg), -1)
        else:
            point_cld = torch.cat((pts, cols), -1)


        # Select points based on mask
        if mask is not None:
            point_cld = point_cld[mask]
            if compute_mean_sq_dist:
                mean3_sq_dist = mean3_sq_dist[mask]

        if compute_mean_sq_dist:
            return point_cld, mean3_sq_dist
        else:
            return point_cld

    def initialize_params(self, init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
        num_pts = init_pt_cld.shape[0]
        means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
        unnorm_rots = torch.from_numpy(np.tile([1, 0, 0, 0], (num_pts, 1))) # [num_gaussians, 4]
        logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
        if gaussian_distribution == "isotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
        elif gaussian_distribution == "anisotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
        else:
            raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")

        if self.dynosplatam and self.config['mode'] != 'splatam':
            dyno_mask = (init_pt_cld[:, 6] != 0).squeeze()
        else:
            dyno_mask = torch.zeros(means3D.shape[0], dtype=bool)

        _unnorm_rotations = torch.zeros((means3D.shape[0], 4, self.num_frames), dtype=torch.float32).cuda()
        _unnorm_rotations[:, :, 0] = unnorm_rots
        _means3D = torch.zeros((means3D.shape[0], 3, self.num_frames), dtype=torch.float32).cuda()
        _means3D[:, :, 0] = means3D

        self.params = {
                'means3D': _means3D,
                'rgb_colors': init_pt_cld[:, 3:6],
                'unnorm_rotations': _unnorm_rotations,
                'logit_opacities': logit_opacities,
                'log_scales': log_scales
            }
    
        if self.dynosplatam:
            self.params['instseg'] = init_pt_cld[:, 6].cuda().long()
            if self.dataset.load_embeddings:
                self.params['embeddings'] = init_pt_cld[:, 7:]

        # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
        cam_rots = np.tile([1, 0, 0, 0], (1, 1))
        cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
        self.params['cam_unnorm_rots'] = cam_rots
        self.params['cam_trans'] = np.zeros((1, 3, num_frames))

        for k, v in self.params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                self.params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
            else:
                self.params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

        self.variables = {'max_2D_radius': torch.zeros(self.params['means3D'].shape[0]).cuda().float(),
            'means2D_gradient_accum': torch.zeros(self.params['means3D'].shape[0]).cuda().float(),
            'denom': torch.zeros(self.params['means3D'].shape[0]).cuda().float(),
            'timestep': torch.zeros(self.params['means3D'].shape[0]).cuda().float(),
            'dyno_mask': dyno_mask,
            'moving': torch.ones(self.params['means3D'].shape[0], dtype=bool).cuda().float()}

        instseg_mask = self.params["instseg"].long()
        self.variables = calculate_neighbors_seg(
                self.params, self.variables, 0, instseg_mask, num_knn=20)
        
    def initialize_optimizer(self, params, lrs_dict, tracking):
        lrs = lrs_dict
        param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
        if tracking:
            return torch.optim.Adam(param_groups)
        else:
            return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


    def get_loss(self, params, variables, curr_data, iter_time_idx, loss_weights, use_sil_for_loss,
                sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
                mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, \
                    tracking_iteration=None):
        
        # Initialize Loss Dictionary
        losses = {}

        # Transform Gaussians to camera frame
        if tracking:
            # Get current frame Gaussians, where only the camera pose gets gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                                gaussians_grad=False,
                                                camera_grad=True)
        elif mapping:
            if do_ba:
                # Get current frame Gaussians, where both camera pose and Gaussians get gradient
                transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                    gaussians_grad=True,
                                                    camera_grad=True)
            else:
                # Get current frame Gaussians, where only the Gaussians get gradient
                transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                    gaussians_grad=True,
                                                    camera_grad=False)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                gaussians_grad=True,
                                                camera_grad=False)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                    transformed_gaussians)

        # RGB Rendering
        rendervar['means2D'].retain_grad()
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

        # Depth & Silhouette Rendering
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
        depth = depth_sil[0, :, :].unsqueeze(0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)
        depth_sq = depth_sil[2, :, :].unsqueeze(0)
        uncertainty = depth_sq - depth**2
        uncertainty = uncertainty.detach()

        # Mask with valid depth values (accounts for outlier depth values)
        nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
        if ignore_outlier_depth_loss:
            depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
            mask = (depth_error < 10*depth_error.median())
            mask = mask & (curr_data['depth'] > 0)
        else:
            mask = (curr_data['depth'] > 0)

        mask = mask & nan_mask
        # Mask with presence silhouette mask (accounts for empty space)
        if tracking and use_sil_for_loss:
            mask = mask & presence_sil_mask

        # Depth loss
        if use_l1:
            mask = mask.detach()
            if tracking:
                losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
            else:
                losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
        
        # RGB Loss
        if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
            color_mask = torch.tile(mask, (3, 1, 1))
            color_mask = color_mask.detach()
            losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
        elif tracking:
            losses['im'] = torch.abs(curr_data['im'] - im).sum()
        else:
            losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
        
        # EMBEDDING LOSS
        if self.dataset.load_embeddings:
            pass
            # losses['emb'] = curr_data['embeddings']

        # ADD DYNO LOSSES LIKE RIGIDITY
        if self.dynosplatam:
            pass

        # Visualize the Diff Images
        if tracking and visualize_tracking_loss:
            fig, ax = plt.subplots(2, 4, figsize=(12, 6))
            weighted_render_im = im * color_mask
            weighted_im = curr_data['im'] * color_mask
            weighted_render_depth = depth * mask
            weighted_depth = curr_data['depth'] * mask
            diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
            diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
            viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
            ax[0, 0].imshow(viz_img)
            ax[0, 0].set_title("Weighted GT RGB")
            viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
            ax[1, 0].imshow(viz_render_img)
            ax[1, 0].set_title("Weighted Rendered RGB")
            ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
            ax[0, 1].set_title("Weighted GT Depth")
            ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
            ax[1, 1].set_title("Weighted Rendered Depth")
            ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
            ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
            ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
            ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
            ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
            ax[0, 3].set_title("Silhouette Mask")
            ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
            ax[1, 3].set_title("Loss Mask")
            # Turn off axis
            for i in range(2):
                for j in range(4):
                    ax[i, j].axis('off')
            # Set Title
            fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
            # Figure Tight Layout
            fig.tight_layout()
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
            plt.close()
            plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
            cv2.imshow('Diff Images', plot_img)
            cv2.waitKey(1)
            ## Save Tracking Loss Viz
            # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
            # os.makedirs(save_plot_dir, exist_ok=True)
            # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
            # plt.close()

        weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
        loss = sum(weighted_losses.values())

        seen = radius > 0
        variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
        variables['seen'] = seen
        weighted_losses['loss'] = loss

        return loss, weighted_losses, variables

    def get_loss_dyno(self, params, variables, curr_data, iter_time_idx, loss_weights, use_sil_for_loss,
                sil_thres, use_l1, ignore_outlier_depth_loss, dyno=False, cam_tracking=False, obj_tracking=False,
                mapping=False, moving_mask=False):
        # Initialize Loss Dictionary
        losses = {}
        
        if cam_tracking:
            transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                                gaussians_grad=False,
                                                camera_grad=True)
        else:
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                        gaussians_grad=True,
                                                        camera_grad=False)
        
        if moving_mask and obj_tracking:
            for k, v in transformed_gaussians.items():
                if 'cam' not in k:
                    transformed_gaussians[k] = v[variables['moving']]
        elif moving_mask and mapping:
            for k, v in transformed_gaussians.items():
                if 'cam' not in k:
                    transformed_gaussians[k] = v[~variables['moving']]

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_gaussians, iter_time_idx)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                transformed_gaussians, iter_time_idx)
        # seg_rendervar = transformed_params2instsegmov(params, curr_data['w2c'],
        #                                                 transformed_gaussians, iter_time_idx)
        
        # print(rendervar['means3D'], rendervar['means3D'].shape)
        # RGB Rendering
        rendervar['means2D'].retain_grad()
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
        
        # Depth & Silhouette Rendering
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)

        # Instseg
        # instseg, _, _, = Renderer(raster_settings=curr_data['cam'])(**seg_rendervar)
        # instseg = instseg[2, :, :]
        # moving = instseg[1, :, :]
        # moving = (moving > mov_thresh)
        
        # silouette
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        # depth 
        depth = depth_sil[0, :, :].unsqueeze(0)
        depth_sq = depth_sil[2, :, :].unsqueeze(0)
        uncertainty = depth_sq - depth**2
        uncertainty = uncertainty.detach()

        # Mask with valid depth values (accounts for outlier depth values)
        nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
        if ignore_outlier_depth_loss:
            depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
            mask = (depth_error < 10*depth_error.median())
            mask = mask & (curr_data['depth'] > 0)
        else:
            mask = (curr_data['depth'] > 0)

        mask = mask & nan_mask
        # Mask with presence silhouette mask (accounts for empty space)
        if use_sil_for_loss:
            mask = mask & presence_sil_mask

        # Depth loss
        if use_l1:
            mask = mask.detach()
            if cam_tracking:
                losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
            else:
                losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
        
        # RGB Loss
        # if (cam_tracking) and (use_sil_for_loss or ignore_outlier_depth_loss):
        if (cam_tracking and obj_tracking) and (use_sil_for_loss or ignore_outlier_depth_loss):
            color_mask = torch.tile(mask, (3, 1, 1))
            color_mask = color_mask.detach()
            losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
        elif cam_tracking:
            losses['im'] = torch.abs(curr_data['im'] - im).sum()
        else:
            losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (
                1.0 - calc_ssim(im, curr_data['im']))

        # SEG LOSS
        # if use_sil_for_loss or ignore_outlier_depth_loss:
        #     losses['instseg'] = torch.abs(curr_data['instseg'] - instseg)[mask].sum()
        # else:
        #     losses['instseg'] = 0.8 * l1_loss_v1(instseg, curr_data['instseg']) + 0.2 * (
        #         1.0 - calc_ssim(instseg, curr_data['instseg']))
        
        # EMBEDDING LOSS
        if self.dataset.load_embeddings:
            pass
            # losses['emb'] = curr_data['embeddings']

        # ADD DYNO LOSSES LIKE RIGIDITY
        # DYNO LOSSES
        if dyno:
            moving = variables['moving']
            # get relative rotation
            prev_rot = params["unnorm_rotations"][:, :, iter_time_idx-1].clone().detach()
            prev_rot[:, 1:] = -1 * prev_rot[:, 1:]
            prev_means = params["means3D"][:, :, iter_time_idx-1].clone().detach()
            curr_rot = transformed_gaussians["unnorm_rotations"] # params["unnorm_rotations"][:, :, iter_time_idx]
            rel_rot = quat_mult(curr_rot, prev_rot)
            rel_rot_mat = build_rotation(rel_rot)

            # Force same segment to have similar rotation and translation
            # mean_rel_rot_seg = scatter_mean(rel_rot, instseg_mask.squeeze(), dim=0)
            losses['rot'] = weighted_l2_loss_v2(
                rel_rot[variables["neighbor_indices"]],
                rel_rot[variables["self_indices"]],
                variables["neighbor_weight"])

            # translation 
            # dx = params["means3D"] - prev_means
            # mean_dx_seg = scatter_mean(dx, instseg_mask.squeeze(), dim=0)
            # losses['dx'] = l2_loss_v2(mean_dx_seg[instseg_mask], dx)

            # rigid body
            curr_means = transformed_gaussians["means3D"] # params["means3D"][:, :, iter_time_idx]
            offset = curr_means[variables["self_indices"]] - curr_means[variables["neighbor_indices"]]
            offset_prev_coord = (rel_rot_mat[variables["self_indices"]].transpose(2, 1) @ offset.unsqueeze(-1)).squeeze(-1)
            prev_offset = prev_means[variables["self_indices"]] - prev_means[variables["neighbor_indices"]]
            losses['rigid'] = l2_loss_v2(offset_prev_coord, prev_offset)

            # isometry
            zero_means = params["means3D"][:, :, 0].clone().detach()
            offset_0 =  zero_means[variables["self_indices"]] - zero_means[variables["neighbor_indices"]]
            losses['iso'] = l2_loss_v2(offset, offset_0)
            

        weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
        loss = sum(weighted_losses.values())

        seen = radius > 0
        variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
        variables['seen'] = seen
        weighted_losses['loss'] = loss

        return loss, weighted_losses, variables

    def initialize_new_params(self, new_pt_cld, mean3_sq_dist, gaussian_distribution):
        num_pts = new_pt_cld.shape[0]
        means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
        unnorm_rots = torch.from_numpy(np.tile([1, 0, 0, 0], (num_pts, 1))) # [num_gaussians, 4]
        logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
        if gaussian_distribution == "isotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
        elif gaussian_distribution == "anisotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
        else:
            raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
        
        if self.dynosplatam and self.config['mode'] != 'splatam':
            dyno_mask = (new_pt_cld[:, 6] != 0).squeeze()
        else:
            dyno_mask = torch.zeros(means3D.shape[0], dtype=bool)

        _unnorm_rotations = torch.zeros((means3D.shape[0], 4, self.num_frames), dtype=torch.float32).cuda()
        _unnorm_rotations[:, :, 0] = unnorm_rots
        _means3D = torch.zeros((means3D.shape[0], 3, self.num_frames), dtype=torch.float32).cuda()
        _means3D[:, :, 0] = means3D
        
        params = {
                    'means3D': _means3D,
                    'rgb_colors': new_pt_cld[:, 3:6],
                    'unnorm_rotations': _unnorm_rotations,
                    'logit_opacities': logit_opacities,
                    'log_scales': log_scales
            }
        if self.dynosplatam:
            self.params['instseg'] = new_pt_cld[:, 6].long()

        if self.dataset.load_embeddings:
            self.params['embeddings'] = new_pt_cld[:, 7:]

        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

        return params, dyno_mask

    def add_new_gaussians(self, curr_data, sil_thres, 
                        time_idx, mean_sq_dist_method, gaussian_distribution):
        # Silhouette Rendering
        transformed_gaussians = transform_to_frame(self.params, time_idx, gaussians_grad=False, camera_grad=False)
        depth_sil_rendervar = transformed_params2depthplussilhouette(self.params, curr_data['w2c'],
                                                                    transformed_gaussians)
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
        silhouette = depth_sil[1, :, :]
        non_presence_sil_mask = (silhouette < sil_thres)
        # Check for new foreground objects by using GT depth
        gt_depth = curr_data['depth'][0, :, :]
        render_depth = depth_sil[0, :, :]
        depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
        non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
        # Determine non-presence mask
        non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
        # Flatten mask
        non_presence_mask = non_presence_mask.reshape(-1)

        # Get the new frame Gaussians based on the Silhouette
        if torch.sum(non_presence_mask) > 0:
            # Get the new pointcloud in the world frame
            curr_cam_rot = torch.nn.functional.normalize(self.params['cam_unnorm_rots'][..., time_idx].detach())
            curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
            curr_w2c = torch.eye(4).cuda().float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran
            valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
            non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
            new_pt_cld, mean3_sq_dist = self.get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                        curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                        mean_sq_dist_method=mean_sq_dist_method, instseg=curr_data['instseg'], embeddings=curr_data['embeddings'])
            new_params, dyno_mask = self.initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
            num_gaussians = new_params['means3D'].shape[0] + self.params['means3D'].shape[0]

            # make new static Gaussians parameters and update static variables
            for k, v in new_params.items():
                self.params[k] = torch.nn.Parameter(torch.cat((self.params[k], v), dim=0).requires_grad_(True))
            self.variables['means2D_gradient_accum'] = torch.zeros(num_gaussians, device="cuda").float()
            self.variables['denom'] = torch.zeros(num_gaussians, device="cuda").float()
            self.variables['max_2D_radius'] = torch.zeros(num_gaussians, device="cuda").float()
            new_timestep = time_idx*torch.ones(new_params['means3D'].shape[0], device="cuda").float()
            self.variables['timestep'] = torch.cat((self.variables['timestep'], new_timestep), dim=0)
            self.variables['dyno_mask'] = dyno_mask
            new_moving = torch.ones(self.params['means3D'].shape[0], dtype=bool).cuda().float()
            self.variables['moving'] = torch.cat((self.variables['moving'], new_moving), dim=0)
            
    def initialize_camera_pose(self, curr_time_idx, forward_prop):
        with torch.no_grad():
            if curr_time_idx > 1 and forward_prop:
                # Initialize the camera pose for the current frame based on a constant velocity model
                # Rotation
                prev_rot1 = F.normalize(self.params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
                prev_rot2 = F.normalize(self.params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
                new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
                self.params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
                # Translation
                prev_tran1 = self.params['cam_trans'][..., curr_time_idx-1].detach()
                prev_tran2 = self.params['cam_trans'][..., curr_time_idx-2].detach()
                new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
                self.params['cam_trans'][..., curr_time_idx] = new_tran.detach()
            else:
                # Initialize the camera pose for the current frame
                self.params['cam_unnorm_rots'][..., curr_time_idx] = self.params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
                self.params['cam_trans'][..., curr_time_idx] = self.params['cam_trans'][..., curr_time_idx-1].detach()
    
    def initialize_dyno_poses(self, curr_time_idx, forward_prop=True, moving_forward_thresh=0.0):
        mask = (curr_time_idx - self.variables['timestep'] == 1).squeeze()
        with torch.no_grad():
            if forward_prop:
                # Initialize the camera pose for the current frame based on a constant velocity model
                # Rotation
                new_rot = self.params['unnorm_rotations'][mask, :, 0]
                new_rot = torch.nn.Parameter(new_rot.cuda().float().contiguous().requires_grad_(True))
                self.params['unnorm_rotations'][mask, :, curr_time_idx] = new_rot

                # Translation
                new_tran = self.params['means3D'][mask, :, 0]
                new_tran = torch.nn.Parameter(new_tran.cuda().float().contiguous().requires_grad_(True))
                self.params['means3D'][mask, :, curr_time_idx] = new_tran
        
        mask = (curr_time_idx - self.variables['timestep'] == 2).squeeze()
        with torch.no_grad():
            if moving_forward_thresh:
                prev_tran1 = self.params['means3D'][mask, :, curr_time_idx-1]
                prev_tran2 = self.params['means3D'][mask, :, curr_time_idx-2]
                new_tran = prev_tran1 - prev_tran2
                if new_tran.shape[0]:
                    mean_trans = scatter_mean(new_tran, self.params['instseg'].long(), dim=0)
                    mean_trans = torch.linalg.norm(mean_trans, dim=1)
                    self.variables['moving'] = mean_trans[self.params['instseg'].long()] > moving_forward_thresh

        mask = (curr_time_idx - self.variables['timestep'] > 1).squeeze()
        with torch.no_grad():
            if forward_prop:
                # Initialize the camera pose for the current frame based on a constant velocity model
                # Rotation
                prev_rot1 = normalize_quat(self.params['unnorm_rotations'][mask, :, curr_time_idx-1].detach())
                prev_rot2 = normalize_quat(self.params['unnorm_rotations'][mask, :, curr_time_idx-2].detach())
                prev_rot2_inv = prev_rot2
                prev_rot2_inv[:, 1:] = -1 * prev_rot2_inv[:, 1:]
                new_rot = quat_mult(quat_mult(prev_rot1, prev_rot2_inv), prev_rot1)
                new_rot = torch.nn.Parameter(new_rot.cuda().float().contiguous().requires_grad_(True))
                self.params['unnorm_rotations'][mask, :, curr_time_idx] = new_rot

                # Translation
                prev_tran1 = self.params['means3D'][mask, :, curr_time_idx-1]
                prev_tran2 = self.params['means3D'][mask, :, curr_time_idx-2]
                new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
                new_tran = torch.nn.Parameter(new_tran.cuda().float().contiguous().requires_grad_(True))
                self.params['means3D'][mask, :, curr_time_idx] = new_tran

        return mask

    def convert_params_to_store(self, params):
        params_to_store = {}
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                params_to_store[k] = v.detach().clone()
            else:
                params_to_store[k] = v
        return params_to_store
    
    def initialize_first_timestep(self, scene_radius_depth_ratio, \
            mean_sq_dist_method, gaussian_distribution=None):
        # Get RGB-D Data & Camera Parameters
        embeddings = None
        if not self.dynosplatam:
            color, depth, self.intrinsics, pose = self.dataset[0]
            instseg = None
        else:
            if self.dataset.load_embeddings:
                color, depth, self.intrinsics, pose, instseg, embeddings = self.dataset[0]
            else:
                color, depth, self.intrinsics, pose, instseg = self.dataset[0]
            instseg = instseg.permute(2, 0, 1)

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        
        # Process Camera Parameters
        self.intrinsics = self.intrinsics[:3, :3]
        w2c = torch.linalg.inv(pose)

        # Setup Camera
        self.cam = setup_camera(color.shape[2], color.shape[1], self.intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

        if self.densify_dataset is not None:
            # Get Densification RGB-D Data & Camera Parameters
            if not self.dynosplatam:
                densify_color, densify_depth, densify_intrinsics, _ = self.densify_dataset[0]
            else:
                densify_color, densify_depth, densify_intrinsics, _, _ = self.densify_dataset[0]
            densify_color = densify_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
            densify_depth = densify_depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
            self.densify_intrinsics = densify_intrinsics[:3, :3]
            self.densify_cam = setup_camera(densify_color.shape[2], densify_color.shape[1], self.densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
        else:
            self.densify_intrinsics = self.intrinsics

        # Get Initial Point Cloud (PyTorch CUDA Tensor)
        mask = (depth > 0) # Mask out invalid depth values
        mask = mask.reshape(-1)
        init_pt_cld, mean3_sq_dist = self.get_pointcloud(color, depth, self.densify_intrinsics, w2c, 
                                                    mask=mask, compute_mean_sq_dist=True, 
                                                    mean_sq_dist_method=mean_sq_dist_method,
                                                    instseg=instseg, embeddings=embeddings)

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        v3d = o3d.utility.Vector3dVector
        pcd.points = v3d(init_pt_cld[:, :3].cpu().numpy())
        print(pcd)
        o3d.io.write_point_cloud(filename='first_time_pcd.pcd', pointcloud=pcd)
        # quit()
        # Initialize Parameters
        self.initialize_params(
            init_pt_cld, self.num_frames, mean3_sq_dist, gaussian_distribution)
        
        # Initialize an estimate of scene radius for Gaussian-Splatting Densification
        self.variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

        return w2c

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
        if "densification_image_height" not in self.dataset_config:
            self.dataset_config["densification_image_height"] = self.dataset_config["desired_image_height"]
            self.dataset_config["densification_image_width"] = self.dataset_config["desired_image_width"]
            self.seperate_densification_res = False
        else:
            if self.dataset_config["densification_image_height"] != self.dataset_config["desired_image_height"] or \
                self.dataset_config["densification_image_width"] != self.dataset_config["desired_image_width"]:
                self.seperate_densification_res = True
            else:
                self.seperate_densification_res = False
        if "tracking_image_height" not in self.dataset_config:
            self.dataset_config["tracking_image_height"] = self.dataset_config["desired_image_height"]
            self.dataset_config["tracking_image_width"] = self.dataset_config["desired_image_width"]
            self.seperate_tracking_res = False
        else:
            if self.dataset_config["tracking_image_height"] != self.dataset_config["desired_image_height"] or \
                self.dataset_config["tracking_image_width"] != self.dataset_config["desired_image_width"]:
                self.seperate_tracking_res = True
            else:
                self.seperate_tracking_res = False

        # Poses are relative to the first frame
        self.dataset = get_dataset(
            config_dict=self.gradslam_data_cfg,
            basedir=self.dataset_config["basedir"],
            sequence=os.path.basename(self.dataset_config["sequence"]),
            start=self.dataset_config["start"],
            end=self.dataset_config["end"],
            stride=self.dataset_config["stride"],
            desired_height=self.dataset_config["desired_image_height"],
            desired_width=self.dataset_config["desired_image_width"],
            device=self.device,
            relative_pose=True,
            ignore_bad=self.dataset_config["ignore_bad"],
            use_train_split=self.dataset_config["use_train_split"],
        )
        self.num_frames = self.dataset_config["num_frames"]
        if self.num_frames == -1:
            self.num_frames = len(self.dataset)

        # Init seperate dataloader for densification if required
        if self.seperate_densification_res:
            self.densify_dataset = get_dataset(
                config_dict=self.gradslam_data_cfg,
                basedir=self.dataset_config["basedir"],
                sequence=os.path.basename(self.dataset_config["sequence"]),
                start=self.dataset_config["start"],
                end=self.dataset_config["end"],
                stride=self.dataset_config["stride"],
                desired_height=self.dataset_config["densification_image_height"],
                desired_width=self.dataset_config["densification_image_width"],
                device=self.device,
                relative_pose=True,
                ignore_bad=self.dataset_config["ignore_bad"],
                use_train_split=self.dataset_config["use_train_split"],
            )
            # Initialize Parameters, Canonical & Densification Camera parameters
            first_frame_w2c = self.initialize_first_timestep(
                                            self.config['scene_radius_depth_ratio'],
                                            self.config['mean_sq_dist_method'],
                                            densify_dataset=self.densify_dataset,
                                            gaussian_distribution=self.config['gaussian_distribution'])                                                                                                                  
        else:
            # Initialize Parameters & Canoncial Camera parameters
            self.densify_cam, self.densify_intrinsics, self.densify_dataset = None, None, None
            first_frame_w2c = self.initialize_first_timestep(
                                            self.config['scene_radius_depth_ratio'],
                                            self.config['mean_sq_dist_method'],
                                            gaussian_distribution=self.config['gaussian_distribution'])
        
        # Init seperate dataloader for tracking if required
        if self.seperate_tracking_res:
            self.tracking_dataset = get_dataset(
                config_dict=self.gradslam_data_cfg,
                basedir=self.dataset_config["basedir"],
                sequence=os.path.basename(self.dataset_config["sequence"]),
                start=self.dataset_config["start"],
                end=self.dataset_config["end"],
                stride=self.dataset_config["stride"],
                desired_height=self.dataset_config["tracking_image_height"],
                desired_width=self.dataset_config["tracking_image_width"],
                device=self.device,
                relative_pose=True,
                ignore_bad=self.dataset_config["ignore_bad"],
                use_train_split=self.dataset_config["use_train_split"],
            )
            tracking_color, _, self.tracking_intrinsics, _ = self.tracking_dataset[0]
            tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
            self.tracking_intrinsics = self.tracking_intrinsics[:3, :3]
            self.tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                        self.tracking_intrinsics.cpu().numpy(), self.first_frame_w2c.detach().cpu().numpy())
        else:
            self.tracking_cam, self.tracking_intrinsics, self.tracking_dataset = None, None, None
        
        return first_frame_w2c

    def load_checkpoint(self, gt_w2c_all_frames, keyframe_list):
        if self.config['load_checkpoint']:
            checkpoint_time_idx = self.config['checkpoint_time_idx']
            print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
            ckpt_path = os.path.join(self.config['workdir'], self.config['run_name'], f"params{checkpoint_time_idx}.npz")
            params = dict(np.load(ckpt_path, allow_pickle=True))
            params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
            self.variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
            self.variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
            self.variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
            self.variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
            # Load the keyframe time idx list
            keyframe_time_indices = np.load(os.path.join(self.config['workdir'], self.config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
            keyframe_time_indices = keyframe_time_indices.tolist()
            # Update the ground truth poses list
            for time_idx in range(checkpoint_time_idx):
                # Load RGBD frames incrementally instead of all frames
                embeddings = None
                if not self.dynosplatam:
                    color, depth, _, gt_pose = self.dataset[time_idx]
                    instseg = None
                else:
                    if self.dataset.load_embeddings:
                        color, depth, _, gt_pose, instseg, embeddings = self.dataset[time_idx]
                    else:
                        color, depth, _, gt_pose, instseg = self.dataset[time_idx]
                # Process poses
                gt_w2c = torch.linalg.inv(gt_pose)
                gt_w2c_all_frames.append(gt_w2c)
                # Initialize Keyframe List
                if time_idx in keyframe_time_indices:
                    # Get the estimated rotation & translation
                    curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Initialize Keyframe Info
                    color = color.permute(2, 0, 1) / 255
                    depth = depth.permute(2, 0, 1)
                    if self.dynosplatam:
                        instseg = instseg.permute(2, 0, 1)
                        if self.dataset.load_embeddings:
                            embeddings = embeddings.permute(2, 0, 1)
                    curr_keyframe = {
                        'id': time_idx,
                        'est_w2c': curr_w2c,
                        'color': color,
                        'depth': depth,
                        'instseg': instseg,
                        'embeddings': embeddings}
                    # Add to keyframe list
                    keyframe_list.append(curr_keyframe)
        else:
            checkpoint_time_idx = 0
        
        return gt_w2c_all_frames, keyframe_list, checkpoint_time_idx

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

        # Load Checkpoint
        gt_w2c_all_frames, keyframe_list, checkpoint_time_idx = \
            self.load_checkpoint(gt_w2c_all_frames, keyframe_list)
        
        curr_data = {'cam': self.cam, 'intrinsics': self.intrinsics, 
            'w2c': self.first_frame_w2c}

        tracking_curr_data = {'cam': self.tracking_cam, 'intrinsics': self.tracking_intrinsics, 
            'w2c': self.first_frame_w2c}

        densify_curr_data = {'cam': self.densify_cam, 'intrinsics': self.densify_intrinsics, 
            'w2c': self.first_frame_w2c}

        # Iterate over Scan
        for time_idx in tqdm(range(checkpoint_time_idx, self.num_frames)):
            # Load RGBD frames incrementally instead of all frames
            embeddings = None
            if not self.dynosplatam:
                color, depth, _, gt_pose = self.dataset[time_idx]
                instseg = None
            else:
                if self.dataset.load_embeddings:
                    color, depth, _, gt_pose, instseg, embeddings = self.dataset[time_idx]
                else:
                    color, depth, _, gt_pose, instseg = self.dataset[time_idx]

            gt_w2c_all_frames, keyframe_list, keyframe_time_indices = \
                self.optimize_time(color, depth, gt_pose, time_idx, gt_w2c_all_frames, curr_data, \
                    tracking_curr_data, densify_curr_data, keyframe_list, keyframe_time_indices, instseg, embeddings)

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
            if self.config['use_wandb']:
                eval(self.dataset, self.params, self.num_frames, self.eval_dir, sil_thres=self.config['mapping']['sil_thres'],
                    wandb_run=self.wandb_run, wandb_save_qual=self.config['wandb']['eval_save_qual'],
                    mapping_iters=self.config['mapping']['num_iters'], add_new_gaussians=self.config['mapping']['add_new_gaussians'],
                    eval_every=self.config['eval_every'], dynosplatam=self.dynosplatam, variables=self.variables)
            else:
                eval(self.dataset, self.params, self.num_frames, self.eval_dir, sil_thres=self.config['mapping']['sil_thres'],
                    mapping_iters=self.config['mapping']['num_iters'], add_new_gaussians=self.config['mapping']['add_new_gaussians'],
                    eval_every=self.config['eval_every'], dynosplatam=self.dynosplatam, variables=self.variables)

        # Add Camera Parameters to Save them
        self.params['timestep'] = self.variables['timestep']
        self.params['intrinsics'] = self.intrinsics.detach().cpu().numpy()
        self.params['w2c'] = self.first_frame_w2c.detach().cpu().numpy()
        self.params['org_width'] = self.dataset_config["desired_image_width"]
        self.params['org_height'] = self.dataset_config["desired_image_height"]
        self.params['gt_w2c_all_frames'] = []
        for gt_w2c_tensor in gt_w2c_all_frames:
            self.params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
        self.params['gt_w2c_all_frames'] = np.stack(self.params['gt_w2c_all_frames'], axis=0)
        self.params['keyframe_time_indices'] = np.array(keyframe_time_indices)
        
        # Save Parameters
        save_params(self.params, self.output_dir)

        # Close WandB Run
        if self.config['use_wandb']:
            wandb.finish()


    def optimize_time(self, color, depth, gt_pose, time_idx, gt_w2c_all_frames, _curr_data, \
        tracking_curr_data, densify_curr_data, keyframe_list, keyframe_time_indices, instseg, embeddings):

        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        if instseg is not None:
            instseg = instseg.permute(2, 0, 1)
        if embeddings is not None:
            embeddings = embeddings.permute(2, 0, 1)
            
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {
            'im': color,
            'depth': depth,
            'id': iter_time_idx,
            'iter_gt_w2c_list': curr_gt_w2c,
            'instseg': instseg,
            'embeddings': embeddings,
            'cam': _curr_data['cam'],
            'intrinsics': _curr_data['intrinsics'],
            'w2c': _curr_data['w2c']}
        
        # Initialize Data for Tracking
        if self.seperate_tracking_res:
            if not self.dynosplatam:
                tracking_color, tracking_depth, _, _ = self.tracking_dataset[time_idx]
                tracking_instseg = None
            else:
                tracking_color, tracking_depth, _, gt_pose, tracking_instseg = self.dataset[time_idx]
                tracking_instseg = tracking_instseg.permute(2, 0, 1)
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data.update(
                {'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx, 'iter_gt_w2c_list': curr_gt_w2c, 'instseg': tracking_instseg})
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = self.config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame in params
        if time_idx > 0:
            self.initialize_camera_pose(
                time_idx, forward_prop=self.config['tracking']['forward_prop'])

        
        # self.track_camera(time_idx, tracking_curr_data, iter_time_idx, curr_gt_w2c)
        if self.dynosplatam and self.config['mode'] == 'static_dyno':
            self.track_objects(time_idx, curr_data, iter_time_idx, curr_gt_w2c)

        self.densify_and_map(time_idx, curr_data, curr_gt_w2c, color, depth, instseg, embeddings,\
            keyframe_list, num_iters_mapping, gt_w2c_all_frames, densify_curr_data)

        if time_idx == 0:
            self.config['tracking_obj']['loss_weights']['rgb_colors'] = 0
            self.config['tracking_obj']['loss_weights']['logit_opacities'] = 0
            self.config['tracking_obj']['loss_weights']['log_scales'] = 0
            self.config['mapping']['loss_weights']['rgb_colors'] = 0
            self.config['mapping']['loss_weights']['logit_opacities'] = 0
            self.config['mapping']['loss_weights']['log_scales'] = 0
        
        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % self.config['keyframe_every'] == 0) or \
                    (time_idx == self.num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth, 'instseg': instseg, 'embeddings': embeddings}
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

        return gt_w2c_all_frames, keyframe_list, keyframe_time_indices

    def track_objects(self, time_idx, curr_data, iter_time_idx, curr_gt_w2c):
        # get instance segementation mask for Gaussians
        time_mask = self.initialize_dyno_poses(
            time_idx,
            moving_forward_thresh=self.config['tracking_obj']['moving_forward_thresh'])
        tracking_start_time = time.time()

        if time_idx > 0 and not self.config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = self.initialize_optimizer(self.params, self.config['tracking_obj']['lrs'], tracking=True)

            # Keep Track of Best Candidate Rotation & Translation
            candidate_dyno_rot = self.params['unnorm_rotations'].detach().clone()
            candidate_dyno_trans = self.params['means3D'].detach().clone()
            current_min_loss = float(1e20)

            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = self.config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Object Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, losses, self.variables = self.get_loss_dyno(self.params, self.variables, curr_data, iter_time_idx, \
                    self.config['tracking_obj']['loss_weights'], self.config['tracking_obj']['use_sil_for_loss'], self.config['tracking_obj']['sil_thres'],
                    self.config['tracking_obj']['use_l1'], self.config['tracking_obj']['ignore_outlier_depth_loss'], cam_tracking=False,
                    dyno=self.config['tracking_obj']['dyno_losses'])
                
                if self.config['use_wandb']:
                    # Report Loss
                    self.wandb_obj_tracking_step = report_loss(losses, self.wandb_run, self.wandb_obj_tracking_step, tracking=True)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_dyno_rot = self.params['unnorm_rotations'][:, :, time_idx].detach().clone()
                        candidate_dyno_trans = self.params['means3D'][:, :, time_idx].detach().clone()

                # Update the runtime numbers
                iter_end_time = time.time()
                self.tracking_obj_iter_time_sum += iter_end_time - iter_start_time
                self.tracking_obj_frame_time_sum += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < self.config['tracking']['depth_loss_thres'] and self.config['tracking']['use_depth_loss_thres']:
                        break
                    elif self.config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if self.config['use_wandb']:
                            self.wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": self.wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                self.params['unnorm_rotations'][:, :, time_idx] = candidate_dyno_rot
                self.params['means3D'][:, :, time_idx] = candidate_dyno_trans

        # Update the runtime numbers
        tracking_end_time = time.time()
        self.tracking_obj_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_obj_frame_time_sum += 1

    def track_camera(self, time_idx, tracking_curr_data, iter_time_idx, curr_gt_w2c):       
        tracking_start_time = time.time()
        if time_idx > 0 and not self.config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = self.initialize_optimizer(self.params, self.config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = self.params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = self.params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = self.config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, losses, self.variables = self.get_loss(self.params, self.variables, tracking_curr_data, iter_time_idx, \
                    self.config['tracking']['loss_weights'], self.config['tracking']['use_sil_for_loss'], self.config['tracking']['sil_thres'],
                    self.config['tracking']['use_l1'], self.config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                    plot_dir=self.eval_dir, visualize_tracking_loss=self.config['tracking']['visualize_tracking_loss'], tracking_iteration=iter)
                if self.config['use_wandb']:
                    # Report Loss
                    self.wandb_tracking_step = report_loss(losses, self.wandb_run, self.wandb_tracking_step, tracking=True)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = self.params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = self.params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if self.config['report_iter_progress']:
                        if self.config['use_wandb']:
                            report_progress(self.params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=self.config['tracking']['sil_thres'], tracking=True,
                                            wandb_run=self.wandb_run, wandb_step=self.wandb_tracking_step, wandb_save_qual=self.config['wandb']['save_qual'])
                        else:
                            report_progress(self.params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=self.config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                self.tracking_iter_time_sum += iter_end_time - iter_start_time
                self.tracking_frame_time_sum += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < self.config['tracking']['depth_loss_thres'] and self.config['tracking']['use_depth_loss_thres']:
                        break
                    elif self.config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if self.config['use_wandb']:
                            self.wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": self.wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                self.params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                self.params['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and self.config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                self.params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                self.params['cam_trans'][..., time_idx] = rel_w2c_tran
        
        # Update the runtime numbers
        tracking_end_time = time.time()
        self.tracking_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_frame_time_sum += 1

        if time_idx == 0 or (time_idx+1) % self.config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if self.config['use_wandb']:
                        report_progress(self.params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=self.config['tracking']['sil_thres'], tracking=True,
                                        wandb_run=self.wandb_run, wandb_step=self.wandb_time_step, wandb_save_qual=self.config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(self.params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=self.config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
                save_params_ckpt(self.params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

    def densify_and_map(self, time_idx, curr_data, curr_gt_w2c, color, depth, instseg, embeddings,\
            keyframe_list, num_iters_mapping, gt_w2c_all_frames, densify_curr_data):
        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % self.config['map_every'] == 0:
            # Densification
            self.densify(time_idx, curr_data, curr_gt_w2c, densify_curr_data)

            # select keyframes for mapping
            selected_keyframes = self.select_keyframes(time_idx, keyframe_list, depth)

            # Reset Optimizer & Learning Rates for Full Map Optimization
            self.map(num_iters_mapping, time_idx, selected_keyframes, color, depth, instseg, embeddings,\
                    keyframe_list, gt_w2c_all_frames, curr_data)

            if time_idx == 0 or (time_idx+1) % self.config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if self.config['use_wandb']:
                            report_progress(self.params, curr_data, 1, progress_bar, time_idx, sil_thres=self.config['mapping']['sil_thres'], 
                                            wandb_run=self.wandb_run, wandb_step=self.wandb_time_step, wandb_save_qual=self.config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(self.params, curr_data, 1, progress_bar, time_idx, sil_thres=self.config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
                    save_params_ckpt(self.params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        
        if time_idx == 0:
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                self.params['unnorm_rotations'][:, :, time_idx] = self.params['unnorm_rotations'][:, :, time_idx].detach()
                self.params['means3D'][:, :, time_idx] = self.params['means3D'][:, :, time_idx].detach()

    def densify(self, time_idx, curr_data, curr_gt_w2c, densify_curr_data):
        if self.config['mapping']['add_new_gaussians'] and time_idx > 0:
            # Setup Data for Densification
            if self.seperate_densification_res:
                # Load RGBD frames incrementally instead of all frames
                if not self.dynosplatam:
                    densify_color, densify_depth, _, _ = self.densify_dataset[time_idx]
                    density_instseg = None
                else:
                    densify_color, densify_depth, _, density_instseg = self.densify_dataset[time_idx]
                    density_instseg = density_instseg.permute(2, 0, 1)
                densify_color = densify_color.permute(2, 0, 1) / 255
                densify_depth = densify_depth.permute(2, 0, 1)
                densify_curr_data.update({
                    'im': densify_color, 'depth': densify_depth, 'id': time_idx, 'iter_gt_w2c_list': curr_gt_w2c, 'instseg': density_instseg})
            else:
                densify_curr_data = curr_data

            # Add new Gaussians to the scene based on the Silhouette
            self.add_new_gaussians(densify_curr_data, 
                                    self.config['mapping']['sil_thres'], time_idx,
                                    self.config['mean_sq_dist_method'], self.config['gaussian_distribution'])
            post_num_pts = self.params['means3D'].shape[0]
            if self.config['use_wandb']:
                self.wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                "Mapping/step": self.wandb_time_step})
        
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
            if len(keyframe_list) > 0:
                # Add last keyframe to the selected keyframes
                selected_time_idx.append(keyframe_list[-1]['id'])
                selected_keyframes.append(len(keyframe_list)-1)
            # Add current frame to the selected keyframes
            selected_time_idx.append(time_idx)
            selected_keyframes.append(-1)
            # Print the selected keyframes
            print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")
        
        return selected_keyframes

    def map(self, num_iters_mapping, time_idx, selected_keyframes, color, depth, instseg, embeddings, keyframe_list,\
            gt_w2c_all_frames, curr_data):
        
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
                iter_color = color
                iter_depth = depth
                iter_instseg = instseg
                iter_embeddings = embeddings
            else:
                # Use Keyframe Data
                iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                iter_instseg = keyframe_list[selected_rand_keyframe_idx]['instseg']
                iter_embeddings = keyframe_list[selected_rand_keyframe_idx]['embeddings']
            iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
            iter_data = {'cam': curr_data['cam'], 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                            'intrinsics': self.intrinsics, 'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c,
                            'instseg': iter_instseg, 'embeddings': iter_embeddings}

            # Loss for current frame
            loss, losses, self.variables = self.get_loss_dyno(self.params, self.variables, iter_data, iter_time_idx, \
                self.config['mapping']['loss_weights'], self.config['mapping']['use_sil_for_loss'], self.config['mapping']['sil_thres'],
                self.config['mapping']['use_l1'], self.config['mapping']['ignore_outlier_depth_loss'], cam_tracking=False,
                dyno=self.config['mapping']['dyno_losses'] and iter_time_idx>0)

            if self.config['use_wandb']:
                # Report Loss
                self.wandb_mapping_step = report_loss(losses, self.wandb_run, self.wandb_mapping_step, mapping=True)
            # Backprop
            loss.backward()

            with torch.no_grad():
                # Prune Gaussians
                if self.config['mapping']['prune_gaussians']:
                    self.params, self.variables = prune_gaussians(self.params, self.variables, optimizer, iter, self.config['mapping']['pruning_dict'])
                    if self.config['use_wandb']:
                        self.wandb_run.log({"Mapping/Number of Gaussians - Pruning": self.params['means3D'].shape[0],
                                        "Mapping/step": self.wandb_mapping_step})
                # Gaussian-Splatting's Gradient-based Densification
                if self.config['mapping']['use_gaussian_splatting_densification']:
                    self.params, self.variables = densify(self.params, self.variables, optimizer, iter, self.config['mapping']['densify_dict'])
                    if self.config['use_wandb']:
                        self.wandb_run.log({"Mapping/Number of Gaussians - Densification": self.params['means3D'].shape[0],
                                        "Mapping/step": self.wandb_mapping_step})
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
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
