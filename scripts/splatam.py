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

from utils.get_data import get_data, just_get_start_pix
from utils.common_utils import seed_everything, save_params_ckpt, save_params, load_params_ckpt
from utils.eval_helpers import report_loss, report_progress, eval
from utils.camera_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2depthplussilhouette,
    transform_to_frame,
    l1_loss_v1,
    quat_mult,
    l2_loss_v2,
    get_hook,
    dyno_losses,
    get_renderings
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify, normalize_quat, remove_points
from utils.neighbor_search import calculate_neighbors_seg, calculate_neighbors_between_pc
from utils.eval_traj import find_closest_to_start_pixels

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from torch_scatter import scatter_mean, scatter_add
import open3d as o3d
import imageio
import torchvision
from torchvision.transforms.functional import InterpolationMode

# Make deterministic
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from utils.render_trajectories import just_render
from utils.eval_traj import eval_traj, vis_grid_trajs


class RBDG_SLAMMER():
    def __init__(self, config):
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
    
    def lift(self, data, transform_pts, support_trajs, t_future=0):
        x_supp = support_trajs[t_future, :, 0]
        y_supp = support_trajs[t_future, :, 1]
        
        x_supp_3D = (x_supp - data['intrinsics'][0][2])/data['intrinsics'][0][0]
        y_supp_3D = (y_supp - data['intrinsics'][1][2])/data['intrinsics'][1][1]
        x_supp_3D = x_supp_3D.reshape(-1)
        y_supp_3D = y_supp_3D.reshape(-1)
        depth_supp = data['depth'][0][y_supp, x_supp].reshape(-1)

        # Initialize point cloud
        pts_cam = torch.stack((x_supp_3D * depth_supp, y_supp_3D * depth_supp, depth_supp), dim=-1)
        if transform_pts:
            pix_ones = torch.ones(depth_supp.shape[0], 1)
            pix_ones = pix_ones.cuda().float()
            pts4 = torch.cat((pts_cam, pix_ones), dim=1)
            c2w = torch.inverse(data['w2c'])
            pts = (c2w @ pts4.T).T[:, :3]
        else:
            pts = pts_cam
        color_supp = data['im'][:, y_supp, x_supp]

        support_params = {'means3D': pts, 'rgb_colors': color_supp.T}

        return support_params

    def lift_support_trajs_to_3D(self, curr_data, next_data, time_idx, transform_pts=True):
        # lift support trajectories into 3D      
        support_params_t0 = self.lift(
            curr_data, transform_pts, support_trajs=curr_data['support_trajs'], t_future=0)
        support_params_t1 = self.lift(
            next_data, transform_pts, support_trajs=curr_data['support_trajs'], t_future=1)

        # get trnaslation in 3D for support points
        support_trajs_trans = support_params_t1['means3D'] - support_params_t0['means3D']
        with torch.no_grad():
            # get nearest support points of points in pc 
            support_traj_neighbor_dics = calculate_neighbors_between_pc(
                self.params,
                time_idx,
                other_params=support_params_t0,
                num_knn=4,
                inflate=2,
                dist_to_use='rgb')

        # compute weighted nearest neighbor translation
        weight = support_traj_neighbor_dics['neighbor_weight_sm'].unsqueeze(1)
        support_trajs_trans = scatter_add(
                        weight * support_trajs_trans[support_traj_neighbor_dics['neighbor_indices']],
                        support_traj_neighbor_dics['self_indices'], dim=0)

        return support_trajs_trans
        
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
            prev_instseg=None,
            support_trajs=None,
            bg=None,
            start_pix=None):
        
        depth = self.pre_process_depth(depth, color)

        # Compute indices of pixels
        width, height = color.shape[2], color.shape[1]
        x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                        torch.arange(height).cuda().float(),
                                        indexing='xy')
        # TODO
        # Try +0.5 to get middle of pixel
        # x_grid += 0.5
        # y_grid += 0.5
        
        # downscale 
        color, depth, instseg, embeddings, mask, x_grid, y_grid = \
            self.resize_for_init(color, depth, instseg, embeddings, mask, x_grid, y_grid)
        width, height = color.shape[2], color.shape[1]

        # get pixel grid into 3D
        mask = mask.reshape(-1)
        xx = (x_grid + 0.5 - intrinsics[0][2])/intrinsics[0][0]
        yy = (y_grid + 0.5 - intrinsics[1][2])/intrinsics[1][1]
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
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
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
        
        if time_idx == 0:
            pcd = o3d.geometry.PointCloud()
            v3d = o3d.utility.Vector3dVector
            pcd.points = v3d(pts.cpu().numpy())
            o3d.io.write_point_cloud(filename=os.path.join(self.eval_dir, "init_pc.xyz"), pointcloud=pcd)

        if self.config["compute_normals"]:
            pcd = o3d.geometry.PointCloud()
            v3d = o3d.utility.Vector3dVector
            pcd.points = v3d(pts.cpu().numpy())
            pcd.estimate_normals()
            normals = np.asarray(pcd.normals)
            point_cld = torch.cat((point_cld, torch.from_numpy(normals).to(point_cld.device)), -1)

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
                bg = bg = bg.reshape(-1, 1)[mask]
        
        if start_pix is not None:
            means2D = torch.stack((x_grid.reshape(-1), y_grid.reshape(-1)), dim=-1)
            if mask is not None:
                means2D = means2D[mask]
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

        logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
        if gaussian_distribution == "isotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
        elif gaussian_distribution == "anisotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
        else:
            raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
        
        unnorm_rotations = torch.zeros((num_pts, 4, self.num_frames), dtype=torch.float32).cuda()
        unnorm_rotations[:, 0, :] = 1
        means3D = torch.zeros((num_pts, 3, self.num_frames), dtype=torch.float32).cuda()
        means3D[:, :, 0] = init_pt_cld[:, :3]
        moving = torch.zeros(num_pts, dtype=torch.float32).cuda()

        params = {
                'means3D': means3D,
                'rgb_colors': init_pt_cld[:, 3:6],
                'unnorm_rotations': unnorm_rotations,
                'logit_opacities': logit_opacities,
                'log_scales': log_scales,
                'moving': moving
            }
    
        params['instseg'] = init_pt_cld[:, 6].cuda().long()

        # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
        cam_rots = np.tile([1, 0, 0, 0], (1, 1))
        cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
        params['cam_unnorm_rots'] = cam_rots
        params['cam_trans'] = np.zeros((1, 3, num_frames))

        if self.dataset.load_embeddings:
            params['embeddings'] = init_pt_cld[:, 7:].cuda().float()

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

        instseg_mask = params["instseg"].long()
        if self.config['use_seg_for_nn']:
            instseg_mask = instseg_mask
        else: 
            instseg_mask = torch.ones_like(instseg_mask).long().to(instseg_mask.device)
        
        if bg is not None:
            variables['bg'] = bg

        if not self.config['neighbors_after_init']:
            with torch.no_grad():
                variables, to_remove = calculate_neighbors_seg(
                    params,
                    variables,
                    0,
                    instseg_mask,
                    num_knn=20,
                    dist_to_use=self.config['dist_to_use'])
        
        return params, variables, to_remove
        
    def initialize_optimizer(self, params, lrs_dict, tracking):
        lrs = lrs_dict
        param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
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
        variables, im, _, depth, _, mask, _, _, _, _, _, _, _, _, _ = \
            get_renderings(
                params,
                variables,
                iter_time_idx,
                curr_data,
                config,
                track_cam=True,
                mov_thresh=self.config['mov_thresh'],
                get_embeddings=self.config['data']['load_embeddings'])

        # Depth loss
        curr_gt_depth = curr_data['depth']
            
        if config['use_l1']:
            mask = mask.detach()
            losses['depth'] = l1_loss_v1(curr_gt_depth, depth, mask, reduction='sum')
        
        # RGB Loss
        curr_gt_im = curr_data['im']

        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_gt_im - im)[color_mask].sum()
        
        weighted_losses = {k: v * config['loss_weights'][k] for k, v in losses.items()}
        loss = sum(weighted_losses.values())
        
        weighted_losses['loss'] = loss

        return loss, weighted_losses, variables


    def get_loss_dyno(self,
                      params,
                      variables,
                      curr_data,
                      iter_time_idx,
                      iter=0,
                      config=None,
                      next_data=None,
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
            variables, im, radius, depth, instseg, mask, transformed_gaussians, prev_means2d, visible, weight, motion2d, _, _, _, embeddings = \
                get_renderings(
                    params,
                    variables,
                    iter_time_idx,
                    curr_data,
                    config,
                    disable_grads=init_next,
                    mov_thresh=self.config['mov_thresh'],
                    get_embeddings=self.config['data']['load_embeddings'])
    
        # intiialize next point cloud for delta tracking
        elif init_next:
            _, next_im, _, next_depth, next_instseg, next_mask, next_transformed_gaussians, next_means2d, _, _, motion2d, _, _, _, next_embeddings = \
                get_renderings(
                    params,
                    variables,
                    iter_time_idx,
                    next_data,
                    config,
                    mov_thresh=self.config['mov_thresh'],
                    prev_means2d=prev_means2d,
                    get_motion=config['use_flow']=='rendered',
                    get_embeddings=self.config['data']['load_embeddings'])

        if init_next and config['use_flow'] == 'pytorch':
            # https://arxiv.org/pdf/2403.12365.pdf
            h, w = weight.shape[1], weight.shape[2]
            weight = weight.detach().clone()
            visible = visible.detach().clone()
            
            # get only 10 most influential ones per pixel for speed up
            # needed cos weight / visibility per tile --> need to take
            # 1000 max influential to ensure some influence on every 
            # pixel in 16 x 16 block
            _weight = weight.sort(dim=0, descending=True)
            idx = _weight.indices[:max_contrib_flow, :, :]
            idx = idx.permute(1, 2, 0).flatten()
            h_idx = torch.arange(h).unsqueeze(1)
            h_idx = torch.tile(h_idx, (max_contrib_flow, 1, w)).flatten()
            w_idx = torch.arange(w)
            w_idx = w_idx.repeat(max_contrib_flow, 1).permute(1, 0).flatten().tile((1, h)).squeeze()
            visible = visible[idx, h_idx, w_idx]
            visible = visible.reshape(h, w, max_contrib_flow).permute(2, 0, 1)
            weight = _weight.values[:max_contrib_flow, :, :]
            
            # compute flow per pixel based on most influential ones
            weight = (weight / (weight.sum(dim=0)+1e-10)).unsqueeze(3)
            flow = (weight * (
                next_means2d[visible.long()] - prev_means2d[visible.long()])).sum(dim=0)

            # compute flow for sparse supervision points
            sparse_flow = flow[
                curr_data['support_trajs'][0, :, 1], curr_data['support_trajs'][0, :, 0], :]
            sparse_flow_gt = curr_data['support_trajs'][1, :] - curr_data['support_trajs'][0, :]
            losses['flow'] = l2_loss_v2(sparse_flow, sparse_flow_gt)
        
        elif init_next and config['use_flow'] == 'pytorch_simple':
            # https://arxiv.org/pdf/2403.12365.pdf

            # get values at positions of support
            h, w = weight.shape[1], weight.shape[2]
            weight = weight.detach().clone()
            visible = visible.detach().clone()

            # get pixels to supervise
            x = curr_data['support_trajs'][0, :, 1]
            y = curr_data['support_trajs'][0, :, 0]
            weight = weight[:, x, y]
            visible = visible[:, x, y]
            pix_ids = torch.arange(weight.shape[1]).unsqueeze(1)
            pix_ids = pix_ids.repeat((1, weight.shape[0])).to(visible.device)

            # keep only visible ones
            pix_ids = pix_ids.permute(1, 0)[visible != 0].flatten()
            weight = weight[visible != 0].flatten()
            visible = visible[visible != 0].flatten()
            weighted_delta = weight.unsqueeze(1) * (
                next_means2d[visible.long()] - prev_means2d[visible.long()])

            # add flow
            sparse_flow = scatter_add(weighted_delta, pix_ids, dim=0)
            sparse_flow_gt = curr_data['support_trajs'][1, :] - curr_data['support_trajs'][0, :]

            # compute loss
            losses['flow'] = l1_loss_v1(sparse_flow, sparse_flow_gt)
        
        elif init_next and config['use_flow'] == 'rendered':
            gt_flows = self.get_flow_supp(curr_data, only_curr=True, iter=iter)
            motion2d = motion2d.permute(1, 2, 0)
            # imageio.imwrite('arrow_flow.png', (norm_motion_y*255).astype(np.uint8))
            sparse_flow = motion2d[
                gt_flows[0, :, 1], gt_flows[0, :, 0], :2]
            
            if iter == 99 or iter == 0:
                import matplotlib.pyplot as plt
                # vis pred flow
                use_supp = True
                if not use_supp:
                    pix_x = torch.arange(motion2d.shape[1]).unsqueeze(1).repeat((1, motion2d.shape[0])).numpy().flatten()
                    pix_y = torch.arange(motion2d.shape[0]).unsqueeze(0).repeat((motion2d.shape[1], 1)).numpy().flatten()
                    flow_x = motion2d[:, :, 0].clone().detach().cpu().numpy().flatten()
                    flow_y = motion2d[:, :, 1].clone().detach().cpu().numpy().flatten()
                    stride = 50
                else:
                    pix_y = gt_flows[0, :, 1].clone().detach().cpu().numpy()
                    pix_x = gt_flows[0, :, 0].clone().detach().cpu().numpy()
                    flow_x = sparse_flow[:, 0].clone().detach().cpu().numpy().flatten()
                    flow_y = sparse_flow[:, 1].clone().detach().cpu().numpy().flatten()
                    stride = 1
                plt.imshow(curr_data['im'].cpu().permute(1, 2, 0).numpy())
                for i in range(0, flow_x.shape[0], stride):
                    # plt.arrow(supp[0, i, 0], supp[0, i, 1], flow[i, 1], flow[i, 0])
                    plt.arrow(pix_x[i], pix_y[i], flow_x[i], flow_y[i], width=0.2)
                plt.margins(0, 0)
                plt.axis('off')
                plt.savefig(f'flow_arrows_{iter}.png', bbox_inches="tight")
                plt.close()

                # vis GT FLOW
                pix_y = gt_flows[0, :, 1].clone().detach().cpu().numpy()
                pix_x = gt_flows[0, :, 0].clone().detach().cpu().numpy()
                flow_y = gt_flows[1, :, 1].clone().detach().cpu().numpy()-gt_flows[0, :, 1].clone().detach().cpu().numpy()
                flow_x = gt_flows[1, :, 0].clone().detach().cpu().numpy()-gt_flows[0, :, 0].clone().detach().cpu().numpy()
                stride = 1
                plt.imshow(curr_data['im'].cpu().permute(1, 2, 0).numpy())
                for i in range(0, flow_x.shape[0], stride):
                    # plt.arrow(supp[0, i, 0], supp[0, i, 1], flow[i, 1], flow[i, 0])
                    plt.arrow(pix_x[i], pix_y[i], flow_x[i], flow_y[i], width=0.2)
                plt.margins(0, 0)
                plt.axis('off')
                plt.savefig(f'flow_arrows_gt_{iter}.png', bbox_inches="tight")
                plt.close()            
            
            sparse_flow_gt = gt_flows[1, :] - gt_flows[0, :]
            losses['flow'] = l1_loss_v1(sparse_flow, sparse_flow_gt)
            self.update_flow_supp(iter=iter, only_curr=True)

        # Depth loss
        curr_gt_depth, next_gt_depth = curr_data['depth'], next_data['depth']
            
        if config['use_l1']:
            if not init_next:
                mask = mask.detach()
                losses['depth'] = l2_loss_v2(curr_gt_depth, depth, mask, reduction='mean')
            else:
                next_mask = next_mask.detach()
                losses['depth'] = l2_loss_v2(next_gt_depth, next_depth, next_mask, reduction='mean')
        
        # RGB Loss
        curr_gt_im, next_gt_im = curr_data['im'], next_data['im']

        if init_next or config['use_sil_for_loss'] or config['ignore_outlier_depth_loss']:
            if not init_next:
                color_mask = torch.tile(mask, (3, 1, 1))
                color_mask = color_mask.detach()
                losses['im'] = torch.abs(curr_gt_im - im)[color_mask].mean()
            else:
                next_color_mask = torch.tile(next_mask, (3, 1, 1))
                next_color_mask = next_color_mask.detach()
                losses['im'] = torch.abs(next_gt_im - next_im)[next_color_mask].mean()
        elif not config['calc_ssmi']:
            losses['im'] = torch.abs(curr_gt_im - im).mean()
        else:
            losses['im'] = 0.8 * l1_loss_v1(im, curr_gt_im) + 0.2 * (
                1.0 - calc_ssim(im, curr_data['im']))

        # EMBEDDING LOSS
        if self.dataset.load_embeddings:
            if not init_next and config['loss_weights']['embeddings'] != 0:
                losses['embeddings'] = l2_loss_v2(
                    torch.nn.functional.normalize(curr_data['embeddings'], p=2, dim=0).permute(1, 2, 0),
                    torch.nn.functional.normalize(embeddings, p=2, dim=0).permute(1, 2, 0),
                    mask.squeeze(),
                    reduction='mean')
            elif config['loss_weights']['embeddings'] != 0:
                losses['embeddings'] = l2_loss_v2(
                    torch.nn.functional.normalize(next_data['embeddings'], p=2, dim=0).permute(1, 2, 0),
                    torch.nn.functional.normalize(next_embeddings, p=2, dim=0).permute(1, 2, 0),
                    next_mask.squeeze(),
                    reduction='mean')

        # BG REG
        if config['bg_reg']:
            if not init_next and iter_time_idx > 0:
                bg_reg = torch.abs(
                    transformed_gaussians['means3D'][:, :] - params['means3D'][:, :, iter_time_idx-1].clone().detach())
                losses['bg_reg'] = bg_reg[variables['bg'][:, 0]].sum()
            elif iter_time_idx > 0:
                bg_reg = torch.abs(
                    next_transformed_gaussians['means3D'] - params['means3D'][:, :, iter_time_idx-1].clone().detach())
                losses['bg_reg'] = bg_reg[variables['bg'][:, 0]].sum()
                
        # ADD DYNO LOSSES LIKE RIGIDITY
        # DYNO LOSSES
        if config['dyno_losses'] and iter_time_idx > 0 and not init_next:
            dyno_losses_curr, self.variables['offset_0'] = dyno_losses(
                params,
                iter_time_idx,
                transformed_gaussians,
                variables,
                self.variables['offset_0'],
                iter,
                use_iso=True,
                update_iso=True)
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
                update_iso=False if iter_time_idx-1 != 0 else True)
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

        return loss, weighted_losses, variables, im, next_im, depth, next_depth, prev_means2d, visible, weight

    def get_flow_supp(self, curr_data, only_curr=True, iter=0):
        if not only_curr:
            flows = list()
            if iter==0:
                self.supervision_flow.append(curr_data['support_trajs'])
            for flow in self.supervision_flow:
                flows.append(flow[:2])
            flows = torch.cat(flows, dim=1)
        else:
            flows = curr_data['support_trajs'][:2]
        return flows

    def update_flow_supp(self, iter=0, only_curr=True):
        if iter == 0 and not only_curr:
            updated_flow = list()
            for flow in self.supervision_flow:
                if flow.shape[0] >= 2:
                    updated_flow.append(flow[1:])
            self.supervision_flow = updated_flow
            print('Updated Flow!!')

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

        logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
        if gaussian_distribution == "isotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
        elif gaussian_distribution == "anisotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
        else:
            raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")

        unnorm_rotations = torch.zeros((num_pts, 4, self.num_frames), dtype=torch.float32).cuda()
        unnorm_rotations[:, 0, :] = 1
        means3D = torch.zeros((num_pts, 3, self.num_frames), dtype=torch.float32).cuda()
        means3D[:, :, :time_idx] = new_pt_cld[:, :3].unsqueeze(2).repeat(1, 1, time_idx)
        moving = torch.zeros(num_pts, dtype=torch.float32).cuda()

        params = {
                'means3D': means3D,
                'rgb_colors': new_pt_cld[:, 3:6],
                'unnorm_rotations': unnorm_rotations,
                'logit_opacities': logit_opacities,
                'log_scales': log_scales,
                'moving': moving
            }

        params['instseg'] = new_pt_cld[:, 6].long().cuda()

        if self.dataset.load_embeddings:
            params['embeddings'] = new_pt_cld[:, 7:].cuda().float()
        
        if bg is not None:
            variables['bg'] = bg
        
        instseg_mask = params["instseg"].long()
        if self.config['use_seg_for_nn']:
            instseg_mask = instseg_mask
            existing_instseg_mask = self.params['instseg']
        else: 
            instseg_mask = torch.ones_like(instseg_mask).long().to(instseg_mask.device)
            existing_instseg_mask = torch.ones_like(self.params['instseg']).long().to(instseg_mask.device)

        if not self.config['neighbors_after_init']:
            with torch.no_grad():
                variables, to_remove = calculate_neighbors_seg(
                    params,
                    variables,
                    time_idx,
                    instseg_mask,
                    num_knn=20,
                    existing_params=self.params,
                    existing_instseg_mask=existing_instseg_mask,
                    dist_to_use=self.config['dist_to_use'])

        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

        return params, variables, to_remove

    def proj_means_to_2D(self, cam, time_idx, shape):
        # project existing params to instseg
        points_xy = cam.projmatrix.squeeze().T.matmul(torch.cat(
            [self.params['means3D'][:, :, time_idx], torch.ones(self.params['means3D'].shape[0], 1).cuda()], dim=1).T)
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
        to_remove = (existing_params_depth < existing_params_proj_depth) * (depth_error > factor*depth_error.median())
        if to_remove.sum():
            self.params, self.variables, self.support_trajs_trans = remove_points(
                to_remove.squeeze(),
                self.params,
                self.variables,
                optimizer=optimizer,
                support_trajs_trans=self.support_trajs_trans)
        print(f'Removed {to_remove.sum()} Gaussians based on depth error...')

    def add_new_gaussians(
            self,
            curr_data,
            sil_thres, 
            depth_error_factor,
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

        depth_sil, _, _, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)

        silhouette = depth_sil[1, :, :]
        non_presence_sil_mask = (silhouette < sil_thres)

        # Check for new foreground objects by using GT depth
        gt_depth = curr_data['depth'][0, :, :]
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
            curr_cam_rot = torch.nn.functional.normalize(self.params['cam_unnorm_rots'][..., time_idx].detach())
            curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
            curr_w2c = torch.eye(4).cuda().float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran
            valid_depth_mask = (curr_data['depth'][0, :, :] > 0)

            os.makedirs(os.path.join(self.eval_dir, 'presence_mask'), exist_ok=True)
            imageio.imwrite(
                os.path.join(self.eval_dir, 'presence_mask', 'gs_{:04d}.png'.format(time_idx)),
                (~non_presence_mask).cpu().numpy().astype(np.uint8)*255)

            # non_presence_mask = non_presence_mask & valid_depth_mask
            # os.makedirs(os.path.join(self.eval_dir, 'non_presence_mask'), exist_ok=True)
            # imageio.imwrite(
            #     os.path.join(self.eval_dir, 'non_presence_mask', 'gs_{:04d}.png'.format(time_idx)),
            #     non_presence_mask.cpu().numpy().astype(np.uint8)*255)

            new_pt_cld, mean3_sq_dist, bg, _ = self.get_pointcloud(
                curr_data['im'],
                curr_data['depth'],
                curr_data['intrinsics'], 
                curr_w2c,
                mask=non_presence_mask.unsqueeze(0),
                mean_sq_dist_method=mean_sq_dist_method,
                instseg=curr_data['instseg'],
                embeddings=curr_data['embeddings'],
                time_idx=time_idx,
                prev_instseg=prev_instseg,
                bg=curr_data['bg'])
            
            print(f"Added {new_pt_cld.shape[0]} Gaussians...")
            
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

            self.variables['means2D_gradient_accum'] = torch.zeros((num_gaussians), device="cuda").float()
            self.variables['denom'] = torch.zeros((num_gaussians), device="cuda").float()
            self.variables['max_2D_radius'] = torch.zeros((num_gaussians), device="cuda").float()

            new_timestep = time_idx*torch.ones(new_params['means3D'].shape[0], device="cuda").float()
            self.variables['timestep'] = torch.cat((self.variables['timestep'], new_timestep), dim=0)
            new_moving = torch.ones(new_params['means3D'].shape[0]).float().cuda()
            self.variables['moving'] = torch.cat((self.variables['moving'], new_moving), dim=0)
            if self.config["compute_normals"]:
                self.variables['normals'] = torch.cat((self.variables['normals'], new_variables['normals']), dim=0)
            
            if not self.config['neighbors_after_init']:
                self.variables['self_indices'] = torch.cat((self.variables['self_indices'], new_variables['self_indices']), dim=0)
                self.variables['neighbor_indices'] = torch.cat((self.variables['neighbor_indices'], new_variables['neighbor_indices']), dim=0)
                self.variables['neighbor_weight'] = torch.cat((self.variables['neighbor_weight'], new_variables['neighbor_weight']), dim=0)
                self.variables['neighbor_weight_sm'] = torch.cat((self.variables['neighbor_weight_sm'], new_variables['neighbor_weight_sm']), dim=0)
                self.variables['neighbor_dist'] = torch.cat((self.variables['neighbor_dist'], new_variables['neighbor_dist']), dim=0)

            if 'bg' in self.variables.keys():
                self.variables['bg'] = torch.cat((self.variables['bg'], new_variables['bg']), dim=0)

        return curr_data

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

                # determine velocity for moving
                if determine_mov == 'seg':
                    self.variables['moving'][mask] = torch.linalg.norm(seg_trans[mask], dim=1)
                elif determine_mov == 'kNN':
                    self.variables['moving'][mask] = torch.linalg.norm(kNN_trans[mask], dim=1)
                elif determine_mov == 'per_point':
                    self.variables['moving'] = torch.linalg.norm(point_trans, dim=1)
                elif determine_mov == 'support_trajs':
                    self.variables['moving'] = torch.linalg.norm(support_trajs_trans, dim=1)
                elif determine_mov == 'background':
                    pass

                return seg_trans, kNN_trans, point_trans
            else:
                return delta_tran, delta_tran, delta_tran
    
    def forward_propagate_camera(
            self,
            params,
            curr_time_idx,
            forward_prop):
        with torch.no_grad():
            if curr_time_idx > 1 and forward_prop:
                time_0 = curr_time_idx - 1 if curr_time_idx > 0 else curr_time_idx
                time_1 = curr_time_idx
                # Initialize the camera pose for the current frame based on a constant velocity model
                # Rotation
                prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., time_1].detach())
                prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., time_0].detach())
                new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
                params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
                # Translation
                prev_tran1 = params['cam_trans'][..., time_1].detach()
                prev_tran2 = params['cam_trans'][..., time_0].detach()
                new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
                params['cam_trans'][..., curr_time_idx] = new_tran.detach()
            else:
                # Initialize the camera pose for the current frame
                params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx].detach()
                params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx].detach()
        
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
                time_0 = curr_time_idx - 1 if curr_time_idx > 0 else curr_time_idx
                time_1 = curr_time_idx

                rot_1 = normalize_quat(params['unnorm_rotations'][:, :, time_1].detach())                                                                                                                                  
                rot_0 = normalize_quat(params['unnorm_rotations'][:, :, time_0].detach())
                rot_0_inv = rot_0
                rot_0_inv[:, 1:] = -1 * rot_0_inv[:, 1:]
                delta_rot = quat_mult(rot_1, rot_0_inv)

                tran_1 = params['means3D'][:, :, time_1]
                tran_0 = params['means3D'][:, :, time_0]
                delta_tran = tran_1 - tran_0                

                # make Gaussians static
                seg_trans, kNN_trans, point_trans = self.make_gaussians_static(
                    curr_time_idx, delta_rot, delta_tran, support_trajs_trans, determine_mov)
                
                # Get time mask 
                mask = (curr_time_idx - variables['timestep'] >= 0).squeeze()

                # add moving mask if to be used
                if mov_static_init:
                    mask = mask & (variables['moving'] > self.config['mov_thresh'])

                if make_bg_static:
                    mask = mask & ~variables['bg'].squeeze()
                
                # For moving objects set new rotation and translation
                if mov_init_by == 'sparse_flow' or mov_init_by == 'sparse_flow_simple' or mov_init_by == 'im_loss' or mov_init_by == 'rendered_flow':
                    new_rot = params['unnorm_rotations'][:, :, curr_time_idx+1].detach()
                else:
                    curr_rot = normalize_quat(params['unnorm_rotations'][:, :, curr_time_idx].detach())
                    new_rot = quat_mult(delta_rot, curr_rot)[mask]
                    new_rot = torch.nn.Parameter(new_rot.cuda().float().contiguous().requires_grad_(True))
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
                
                new_tran = torch.nn.Parameter(new_tran.cuda().float().contiguous().requires_grad_(True))
                params['means3D'][mask, :, curr_time_idx + 1] = new_tran

                # For static objects set new rotation and translation
                if mov_static_init or make_bg_static:
                    mask = (curr_time_idx - variables['timestep'] >= 0).squeeze()
                    if mov_static_init:
                        mask = mask & ~(variables['moving'] > self.config['mov_thresh'])
                    if make_bg_static:
                        mask = mask & variables['bg'].squeeze()
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
        w2c = torch.linalg.inv(pose)

        # Setup Camera
        if timestep == 0:
            self.cam = setup_camera(
                color.shape[2],
                color.shape[1],
                self.intrinsics.cpu().numpy(),
                w2c.detach().cpu().numpy())
            start_pix = just_get_start_pix(
                self.config,
                in_torch=True,
                normalized=False,
                h=self.cam.image_height,
                w=self.cam.image_width,
                rounded=True)
        else:
            start_pix = None
    
        if self.config['checkpoint'] != '' and timestep == 0:
            return

        # Get Initial Point Cloud (PyTorch CUDA Tensor)
        mask = (depth > 0) # Mask out invalid depth values
        init_pt_cld, mean3_sq_dist, bg, gauss_ids_to_track = self.get_pointcloud(
            color,
            depth,
            self.intrinsics,
            w2c,
            mask=mask,
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

    def get_data(self):
        # Load Dataset
        print("Loading Dataset ...")
        # Poses are relative to the first frame
        self.dataset = get_data(self.config)
        self.num_frames = self.config["data"]["num_frames"]
        if self.num_frames == -1:
            self.num_frames = len(self.dataset)
        
        # maybe load checkpoint
        ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
        if not os.path.isfile(os.path.join(ckpt_output_dir, f"temp_params.npz")):
            self.config['checkpoint'] = ''
        
        if self.config['checkpoint'] != '':
            self.params, self.variables = load_params_ckpt(ckpt_output_dir)
            first_frame_w2c = self.variables['first_frame_w2c']
            self.initialize_timestep(
                self.config['scene_radius_depth_ratio'],
                self.config['mean_sq_dist_method'],
                gaussian_distribution=self.config['gaussian_distribution'],
                timestep=0)
            time_idx = self.variables['last_time_idx'].item() + 1
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

    def rgbd_slam(self):        
        self.first_frame_w2c, start_time_idx = self.get_data()
        
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
        
        cam_data = {'cam': self.cam, 'intrinsics': self.intrinsics, 
            'w2c': self.first_frame_w2c}

        prev_data = None
        self.variables['prev_means2d'], self.variables['prev_weight'], self.variables['prev_visible'] = None, None, None
        if start_time_idx != 0:
            prev_data, gt_w2c_all_frames = self.make_data_dict(start_time_idx-1, gt_w2c_all_frames, cam_data)
            time_idx = start_time_idx
        print(f"Starting from time index {start_time_idx}...")
        
        # Iterate over Scan
        for time_idx in tqdm(range(start_time_idx, self.num_frames-1)):
            curr_data, gt_w2c_all_frames = self.make_data_dict(time_idx, gt_w2c_all_frames, cam_data)
            next_data, _ = self.make_data_dict(time_idx+1, gt_w2c_all_frames, cam_data)

            print(f"Optimizing time step {time_idx}...")
            keyframe_list, keyframe_time_indices = \
                self.optimize_time(
                    time_idx,
                    curr_data,
                    keyframe_list,
                    keyframe_time_indices,
                    next_data,
                    prev_data)
            prev_data = curr_data

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

        # Evaluate Final Parameters
        with torch.no_grad():
            self.variables['moving'] = self.variables['moving'] > self.config['mov_thresh']
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
                get_embeddings=self.config['data']['load_embeddings'])

        pts = self.params['means3D'][:, :, time_idx].detach()
        pcd = o3d.geometry.PointCloud()
        v3d = o3d.utility.Vector3dVector
        pcd.points = v3d(pts.cpu().numpy())
        path = os.path.join(self.eval_dir, "final_pc.xyz")
        o3d.io.write_point_cloud(filename=path, pointcloud=pcd)
        print(f"Stored final pc to {path}...")

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
        self.params['gauss_ids_to_track'] = self.variables['gauss_ids_to_track'].cpu().numpy()
        
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


    def optimize_time(
            self,
            time_idx,
            curr_data,
            keyframe_list,
            keyframe_time_indices,
            next_data,
            prev_data):
        
        # track cam
        if self.config['tracking_cam']['num_iters'] != 0 and time_idx > 0:
            self.track_cam(
                time_idx,
                curr_data)
            
        # Densification
        if (time_idx+1) % self.config['add_every'] == 0 and time_idx > 0:
            print('Densifying!')
            curr_data = self.densify(time_idx, curr_data, prev_data)
        
        self.support_trajs_trans = self.lift_support_trajs_to_3D(curr_data, next_data, time_idx)
    
        if self.config['tracking_obj']['num_iters'] != 0:
            optimizer = self.track_objects(
                time_idx,
                curr_data,
                next_data)
        
        if self.config['init_next']['num_iters'] != 0 and (self.config['mov_init_by'] == 'sparse_flow' or self.config['mov_init_by'] == 'im_loss' or self.config['mov_init_by'] == 'sparse_flow_simple' or self.config['mov_init_by'] == 'rendered_flow') :
            with torch.no_grad():
                self.params['means3D'][:, :, time_idx+1] = self.params['means3D'][:, :, time_idx]
                self.params['unnorm_rotations'][:, :, time_idx+1] = self.params['unnorm_rotations'][:, :, time_idx]
            _ = self.track_objects(
                time_idx+1,
                curr_data,
                next_data,
                init_next=True)
        
        # remove floating points based on depth
        if self.config['remove_gaussians']['remove']:
            self.remove_gaussians_with_depth(curr_data, time_idx, optimizer)
        
        # Increment WandB Time Step
        if self.config['use_wandb']:
            self.wandb_time_step += 1

        torch.cuda.empty_cache()
 
        if self.config['neighbors_after_init']:
            with torch.no_grad():
                self.variables, to_remove = calculate_neighbors_seg_after_init(
                    params,
                    variables,
                    time_idx,
                    instseg_mask,
                    num_knn=20,
                    dist_to_use=self.config['dist_to_use'])

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

        return keyframe_list, keyframe_time_indices

    def track_objects(
            self,
            time_idx,
            curr_data,
            next_data,
            init_next=False):
        if not init_next:
            config = self.config['tracking_obj']
            if self.config['use_wandb']:
                wandb_step = self.wandb_obj_tracking_step
        else:
            config = self.config['init_next']
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

        # Reset Optimizer & Learning Rates for tracking
        optimizer = self.initialize_optimizer(
            self.params,
            config['lrs'],
            tracking=True)

        if config['take_best_candidate']:
            # Keep Track of Best Candidate Rotation & Translation
            candidate_dyno_rot = self.params['unnorm_rotations'][:, :, time_idx].detach().clone()
            candidate_dyno_trans = self.params['means3D'][:, :, time_idx].detach().clone()
            current_min_loss = float(1e20)
            best_time_idx = 0

        # Tracking Optimization
        iter = 0
        best_iter = 0
        num_iters_tracking = config['num_iters']
        if not init_next:
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Object Tracking Time Step: {time_idx}")
        else:
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Initializing Time Step: {time_idx}")
        
        while iter <= num_iters_tracking:
            iter_start_time = time.time()
            # Loss for current frame
            loss, losses, self.variables, im, next_im, depth, next_depth, means2d, weight, visible = self.get_loss_dyno(
                self.params,
                self.variables,
                curr_data,
                time_idx,
                iter=iter,
                config=config,
                next_data=next_data,
                next_params=next_params,
                next_variables=next_variables,
                init_next=init_next)

            if config['disable_rgb_grads_old']:
                for k, p in self.params.items():
                    if 'cam' in k:
                        continue
                    if k not in ['rgb_colors', 'logit_opacities', 'log_scales'] :
                        continue
                    p.register_hook(get_hook(self.variables['timestep'] != time_idx))

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

        progress_bar.close()
        if config['take_best_candidate']:
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                self.params['unnorm_rotations'][:, :, time_idx] = candidate_dyno_rot
                self.params['means3D'][:, :, time_idx] = candidate_dyno_trans
                print(f'Best candidate at iteration {best_iter}!')
                if not init_next:
                    self.variables['prev_means2d'], self.variables['prev_weight'], self.variables['prev_visible'] = candidate_means2d, candidate_weight, candidate_visible
        elif not init_next:
            self.variables['prev_means2d'], self.variables['prev_weight'], self.variables['prev_visible'] = means2d.detach().clone(), weight.detach().clone(), visible.detach().clone()
        
        if self.config['use_wandb']:
            if not init_next:
                self.wandb_obj_tracking_step = wandb_step
            else:
                self.wandb_init_next_step = wandb_step

        # Update the runtime numbers
        tracking_end_time = time.time()
        self.tracking_obj_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_obj_frame_time_count += 1
        
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
                print(f'Best candidate at iteration {best_iter}!')

        # Update the runtime numbers
        tracking_end_time = time.time()
        self.tracking_cam_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_cam_frame_time_count += 1
        
        return optimizer

    def densify(self, time_idx, curr_data, prev_data):
        if self.config['add_gaussians']['add_new_gaussians']:
            if prev_data is not None:
                densify_prev_data = curr_data
            else:
                densify_prev_data = prev_data

            # Add new Gaussians to the scene based on the Silhouette
            curr_data = self.add_new_gaussians(curr_data, 
                                    self.config['add_gaussians']['sil_thres_gaussians'],
                                    self.config['add_gaussians']['depth_error_factor'],
                                    time_idx,
                                    self.config['mean_sq_dist_method'],
                                    self.config['gaussian_distribution'],
                                    densify_prev_data['instseg'],
                                    densify_prev_data['im'],
                                    self.params, 
                                    self.variables)

            post_num_pts = self.params['means3D'].shape[0]
            if self.config['use_wandb']:
                self.wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                "Mapping/step": self.wandb_time_step})
        return curr_data


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
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slammer = RBDG_SLAMMER(experiment.config)
    rgbd_slammer.rgbd_slam()


