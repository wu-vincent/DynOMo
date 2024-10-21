import os
import sys
import time

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import numpy as np
import torch
from tqdm import tqdm
import wandb
import copy

from src.utils.get_data import get_data, load_scene_data
from src.utils.common_utils import save_params_ckpt, save_params, load_params_ckpt
from src.evaluate.rendering_evaluator import RenderingEvaluator
from utils.losses import (
    l1_loss_v1,
    quat_mult,
    get_hook,
    physics_based_losses,
    get_rendered_losses,
    get_l1_losses
)
from src.model.renderer import RenderHelper
from utils.gaussian_utils import build_rotation, prune_gaussians, densify, normalize_quat, matrix_to_quaternion
from src.utils.neighbor_search import calculate_neighbors_seg_after_init

from torch_scatter import scatter_add
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

from src.evaluate.trajectory_evaluator import TrajEvaluator

from collections import OrderedDict
from typing import Dict, Callable

import gc

import json
from src.model.scene import GaussianScene
from src.model.logger import Logger
from src.model.optimization import OptimHandler


class DynOMo():
    def __init__(self, config):
        self.batch = 0
        # setup pytorch
        torch.set_num_threads(1)
        self.hook_list = list()
        self.device = config['primary_device']
        device = torch.device(self.device)
        torch.cuda.set_device(device)
        torch.cuda.seed()

        # update config
        if config['data']['start_from_complete_pc']:
            config['data']['load_embeddings'] = False
            config['add_gaussians']['add_new_gaussians'] = False
        self.config = config

        # Create Output Directories
        self.output_dir = os.path.join(config["workdir"], config["run_name"])
        self.eval_dir = os.path.join(self.output_dir, "eval")
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # Init WandB
        if config['use_wandb']:
            self.wandb_time_step = 0
            self.wandb_obj_tracking_step = 0
            self.wandb_run = wandb.init(project=config['wandb']['project'],
                                group=config['wandb']['group'],
                                name=config['wandb']['name'],
                                config=config)
        
        # initalize eval lists
        self.eval_pca = None

    def get_loss_cam(self,
                      curr_data,
                      iter_time_idx,
                      config=None,
                      use_gt_mask=False,
                      use_depth_error=False):
 
        # Initialize Loss Dictionary
        losses = {}
        self.scene.variables, im, _, depth, mask, _, _, _, time_mask, _, _, embeddings, bg, _, _ = \
            self.render_helper.get_renderings(
                self.scene.params,
                self.scene.variables,
                iter_time_idx,
                curr_data,
                config,
                track_cam=True,
                get_seg=True,
                get_embeddings=self.config['data']['load_embeddings'])
        
        bg_mask = bg.detach().clone() > 0.5
        bg_mask_gt = curr_data['bg'].detach().clone() > 0.5
        mask = (bg_mask & mask.detach()).squeeze()
        mask_gt = (bg_mask_gt & mask.detach()).squeeze()
        if use_gt_mask:
            mask = mask_gt
        
        if use_depth_error:
            depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
            error_depth = (depth > curr_data['depth']) * (depth_error > 25*depth_error.median())
            mask = mask | error_depth
        
        losses = get_rendered_losses(
            config,
            losses,
            curr_data,
            im,
            depth,
            embeddings,
            mask,
            self.dataset.load_embeddings,
            iter_time_idx,
            self.scene,
            self.device)

        weighted_losses = {k: v * config['loss_weights'][k] for k, v in losses.items()}
        loss = sum(weighted_losses.values())
        
        weighted_losses['loss'] = loss

        return loss, weighted_losses

    def get_loss_gaussians(self,
                      curr_data,
                      iter_time_idx,
                      num_iters,
                      iter=0,
                      config=None,
                      init_next=False,
                      early_stop_eval=False):

        # Initialize Loss Dictionary
        losses = {}
        # get renderings for current time
        self.scene.variables, im, radius, depth, mask, transformed_gaussians, visible, weight, time_mask, _, _, embeddings, bg, _ = \
            self.render_helper.get_renderings(
                self.scene.params,
                self.scene.variables,
                iter_time_idx,
                curr_data,
                config,
                get_seg=True,
                disable_grads=init_next,
                get_embeddings=self.config['data']['load_embeddings'])

        mask = mask.detach()

        losses = get_rendered_losses(
            config,
            losses,
            curr_data,
            im,
            depth,
            embeddings,
            mask,
            bg,
            self.dataset.load_embeddings,
            iter_time_idx,
            self.scene,
            self.device)
        
        losses = get_l1_losses(
            losses,
            config,
            iter_time_idx,
            self.scene,
            self.dataset.load_embeddings)
            
        # ADD DYNO LOSSES LIKE RIGIDITY
        # DYNO LOSSES
        if iter_time_idx > 0:
            # print(variables['timestep'])
            losses, offset_0 = physics_based_losses(
                self.scene.params,
                iter_time_idx,
                transformed_gaussians,
                self.scene.variables,
                self.scene.variables['offset_0'],
                iter,
                use_iso=True,
                update_iso=True, 
                device=self.device,
                losses=losses)
            self.scene.variables['offset_0'] = offset_0
        
        weighted_losses = {k: v * config['loss_weights'][k] for k, v in losses.items()}
        loss = sum(weighted_losses.values())

        seen = radius > 0
        self.scene.variables['max_2D_radius'][time_mask][seen] = torch.max(radius[seen], self.scene.variables['max_2D_radius'][time_mask][seen])
        self.scene.variables['seen'] = torch.zeros_like(self.scene.variables['max_2D_radius'], dtype=bool, device=self.device)
        self.scene.variables['seen'][time_mask] = True
        weighted_losses['loss'] = loss
    
        if (iter == num_iters - 1 or early_stop_eval) and self.config['eval_during']:
            self.eval_pca = self.rendering_evaluator.eval_during(
                curr_data,
                self.scene.params,
                iter_time_idx,
                self.eval_dir,
                im=im.detach().clone(),
                rastered_depth=depth.detach().clone(),
                rastered_sil=mask.detach().clone(),
                rastered_bg=bg.detach().clone(),
                rendered_embeddings=embeddings.detach().clone() if embeddings is not None else embeddings,
                pca=self.eval_pca,
                viz_config=self.config['viz'],
                num_frames=self.num_frames)

        return loss, weighted_losses, visible, weight

    def get_kNN_trans(
            self,
            curr_time_idx,
            delta_tran):
        
        mask = (curr_time_idx - self.scene.variables['timestep'] >= 1).squeeze()
        with torch.no_grad():
            if delta_tran[mask].shape[0]:
                # Gaussian kNN, point translation
                weight = self.scene.variables['neighbor_weight_sm'].unsqueeze(1)
                torch.use_deterministic_algorithms(False)
                kNN_trans = scatter_add(
                        weight * delta_tran[self.scene.variables['neighbor_indices']],
                        self.scene.variables['self_indices'], dim=0)
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
                prev_rot2 = normalize_quat(self.scene.params['cam_unnorm_rots'][:, :, time_2].detach())                                                                                                                                  
                prev_rot1 = normalize_quat(self.scene.params['cam_unnorm_rots'][:, :, time_1].detach())
                if not simple_rot:
                    prev_rot1_inv = prev_rot1.clone()
                    prev_rot1_inv[:, 1:] = -1 * prev_rot1_inv[:, 1:]
                    delta_rot = quat_mult(prev_rot2, prev_rot1_inv) # rot from 1 -> 2
                    new_rot = quat_mult(delta_rot, prev_rot2)
                else:
                    new_rot = torch.nn.functional.normalize(prev_rot2 + (prev_rot2 - prev_rot1))
                self.scene.params['cam_unnorm_rots'][..., curr_time_idx + 1] = new_rot.detach()
                
                # forward prop translation
                prev_tran2 = self.scene.params['cam_trans'][..., time_2].detach()
                prev_tran1 = self.scene.params['cam_trans'][..., time_1].detach()
                if not simple_trans:
                    delta_rot_mat = build_rotation(delta_rot).squeeze()
                    new_tran = torch.bmm(delta_rot_mat.unsqueeze(0), prev_tran2.unsqueeze(2)).squeeze() - \
                        torch.bmm(delta_rot_mat.unsqueeze(0), prev_tran1.unsqueeze(2)).squeeze() + prev_tran2.squeeze()
                else:
                    new_tran = prev_tran2 + (prev_tran2 - prev_tran1)

                self.scene.params['cam_trans'][..., curr_time_idx + 1] = new_tran.detach()
            else:
                # Initialize the camera pose for the current frame
                self.scene.params['cam_unnorm_rots'][..., curr_time_idx + 1] = self.scene.params['cam_unnorm_rots'][..., curr_time_idx].detach()
                self.scene.params['cam_trans'][..., curr_time_idx + 1] = self.scene.params['cam_trans'][..., curr_time_idx].detach()
    
    def forward_propagate_gaussians(
            self,
            curr_time_idx,
            forward_prop=False,
            simple_rot=False,
            simple_trans=True):

        # for all other timestamps moving
        with torch.no_grad():
            if forward_prop:
                if simple_rot or not simple_trans:
                    print(f'Using simple rot {simple_rot} and simple trans {simple_trans} for forward prop gauss!')
                # Get time mask 
                mask = (curr_time_idx - self.scene.variables['timestep'] >= 0).squeeze()
                if self.config['make_bg_static']:
                    mask = mask & ~self.scene.params['bg'].detach().clone().squeeze()

                # get time index
                time_1 = curr_time_idx - 1 if curr_time_idx > 0 else curr_time_idx
                time_2 = curr_time_idx
                
                # forward prop rotation
                rot_2 = normalize_quat(self.scene.variables['unnorm_rotations'][:, :, time_2].detach().clone())                                                                                                                                  
                rot_1 = normalize_quat(self.scene.variables['unnorm_rotations'][:, :, time_1].detach().clone())
                if not simple_rot:
                    rot_1_inv = rot_1.clone()
                    rot_1_inv[:, 1:] = -1 * rot_1_inv[:, 1:]
                    delta_rot = quat_mult(rot_2, rot_1_inv)
                    curr_rot = normalize_quat(self.scene.variables['unnorm_rotations'][:, :, curr_time_idx].detach().clone())
                    new_rot = quat_mult(delta_rot, curr_rot)[mask]
                    new_rot = torch.nn.Parameter(new_rot.to(self.device).float().contiguous().requires_grad_(True))
                else:
                    new_rot = torch.nn.functional.normalize(rot_2 + (rot_2 - rot_1))
                self.scene.params['unnorm_rotations'][mask, :] = new_rot

                # forward prop translation
                tran_2 = self.scene.variables['means3D'][:, :, time_2].detach().clone().to(self.device)
                tran_1 = self.scene.variables['means3D'][:, :, time_1].detach().clone().to(self.device)
                if not simple_trans:
                    delta_rot_mat = build_rotation(delta_rot).squeeze()
                    new_tran = torch.bmm(delta_rot_mat, tran_2.unsqueeze(2)).squeeze() - \
                        torch.bmm(delta_rot_mat, point_trans.unsqueeze(2)).squeeze() + tran_2.squeeze()         
                else:
                    delta_tran = tran_2 - tran_1       
                    kNN_trans, point_trans = self.get_kNN_trans(curr_time_idx, delta_tran)
                    curr_tran = self.scene.variables['means3D'][:, :, curr_time_idx].detach().clone().to(self.device)
                    if self.config['mov_init_by'] == 'kNN':
                        new_tran = (curr_tran + kNN_trans)[mask]
                    elif self.config['mov_init_by'] == 'per_point':
                        new_tran = (curr_tran + point_trans)[mask]
                    new_tran = torch.nn.Parameter(new_tran.to(self.device).float().contiguous().requires_grad_(True))
                    self.scene.params['means3D'][mask, :] = new_tran

                # For static objects set new rotation and translation
                if self.config['make_bg_static']:
                    mask = (curr_time_idx - self.scene.variables['timestep'] >= 0).squeeze()
                    mask = mask & self.scene.params['bg'].detach().clone().squeeze()
                    self.scene.params['unnorm_rotations'][mask, :] = curr_rot[mask]
                    self.scene.params['means3D'][mask, :] = curr_tran[mask]
            else:
                print('Not forward propagating gaussians.')

    def load_dataset(self):
        # Load Dataset
        print("Loading Dataset ...")
        # Poses are relative to the first frame
        self.dataset = get_data(self.config, stereo=False)
        self.config['data']['desired_image_height'] = self.dataset.desired_height
        self.config['data']['desired_image_width'] = self.dataset.desired_width
        print(f"Original image height {self.dataset.orig_height} and width {self.dataset.orig_width}...")
        print(f"Desired image height {self.dataset.desired_height} and width {self.dataset.desired_width}...")

        # self.dataset = DataLoader(get_data(self.config), batch_size=1, shuffle=False)
        self.num_frames = self.config["data"]["num_frames"]
        if self.num_frames == -1 or self.num_frames > len(self.dataset):
            self.num_frames = len(self.dataset)

    def init_Gaussian_scne(self):
        # init Gaussian scene 
        self.scene = GaussianScene(self.config)
        
        # maybe load checkpoint
        ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
        temp_params = os.path.isfile(os.path.join(ckpt_output_dir, f"temp_params.npz"))
        final_params = os.path.isfile(os.path.join(ckpt_output_dir, f"params.npz"))

        if self.config['checkpoint'] and temp_params and not final_params:
            self.scene.params, self.scene.variables = load_params_ckpt(ckpt_output_dir, device=self.device)
            first_frame_w2c = self.scene.variables['first_frame_w2c']
            self.scene.initialize_timestep(
                self.config['scene_radius_depth_ratio'],
                self.config['mean_sq_dist_method'],
                gaussian_distribution=self.config['gaussian_distribution'],
                timestep=0,
                w2c=first_frame_w2c)
            time_idx = min(self.num_frames, self.scene.variables['last_time_idx'].item() + 1)
        elif self.config['checkpoint'] and final_params:
            self.scene.params, _, _,first_frame_w2c = load_scene_data(self.config, ckpt_output_dir, device=self.device)
            self.scene.initialize_timestep(
                self.config['scene_radius_depth_ratio'],
                self.config['mean_sq_dist_method'],
                gaussian_distribution=self.config['gaussian_distribution'],
                timestep=0,
                w2c=first_frame_w2c)
            time_idx = self.num_frames
        else:
            self.config['checkpoint'] = False
            first_frame_w2c = self.scene.initialize_timestep(
                self.config['scene_radius_depth_ratio'],
                self.config['mean_sq_dist_method'],
                gaussian_distribution=self.config['gaussian_distribution'],
                timestep=0)
            self.scene.variables['first_frame_w2c'] = first_frame_w2c
            self.scene.variables['offset_0'] = None
            time_idx = 0

        return first_frame_w2c, time_idx
    
    def make_data_dict(self, time_idx):
        color, depth, self.intrinsics, gt_pose, embeddings, bg = \
             self.dataset[time_idx]
        
        # Process poses
        self.scene.variables['gt_w2c_all_frames'].append(torch.linalg.inv(gt_pose))

        curr_data = {
            'im': color,
            'depth': depth,
            'id': time_idx,
            'iter_gt_w2c_list': self.scene.variables['gt_w2c_all_frames'],
            'embeddings': embeddings,
            'cam': self.scene.cam,
            'intrinsics': self.intrinsics,
            'w2c': self.first_frame_w2c,
            'bg': bg}
        
        return curr_data
        
    def eval(self, novel_view_mode=None, eval_renderings=True, eval_traj=True, vis_trajs=True, vis_grid=False):
        self.load_dataset()
        self.first_frame_w2c, _ = self.init_Gaussian_scne()
        self.render_helper = RenderHelper()
        self.rendering_evaluator = RenderingEvaluator(
            self.device,
            self.wandb_run if self.config['use_wandb'] else None,
            save_frames=True,
            eval_dir=self.eval_dir,
            sil_thres=self.config['add_gaussians']['sil_thres_gaussians'],
            viz_config=self.config['viz'],
            get_embeddings=self.config['data']['load_embeddings'],
            config=self.config,
            render_helper=self.render_helper)
        
        self._eval(novel_view_mode, eval_renderings, eval_traj, vis_trajs, vis_grid)
    
    def _eval(self, novel_view_mode=None, eval_renderings=True, eval_traj=True, vis_trajs=False, vis_grid=False):
        # metrics empty dict
        metrics = dict()
        if eval_renderings:
            # Evaluate Final Parameters
            with torch.no_grad():
                self.rendering_evaluator.eval(
                    self.dataset,
                    self.scene.params,
                    self.num_frames,
                    variables=self.scene.params,
                    novel_view_mode=novel_view_mode)

        # eval traj
        evaluator = TrajEvaluator(
            self.config,
            self.scene.params,
            cam=self.scene.cam,
            results_dir=self.eval_dir, 
            vis_trajs=vis_trajs,
            queries_first_t=False if 'iphone' in self.eval_dir else True)
        
        if eval_traj:
            with torch.no_grad():
                metrics = evaluator.eval_traj()
                with open(os.path.join(self.eval_dir, f'traj_metrics.json'), 'w') as f:
                    json.dump(metrics, f)
                with open(os.path.join(self.eval_dir, f'traj_metrics.txt'), 'w') as f:
                    f.write(f"Trajectory metrics: {metrics}")
            print("Trajectory metrics: ",  metrics)
        
        if eval_traj and 'iphone' in self.eval_dir:
            with torch.no_grad():
                cam_metrics = evaluator.eval_cam_traj()
            print(f'Cam Traj Metrics: {cam_metrics}')

        if vis_grid:
            evaluator.vis_grid_trajs(
                mask=torch.from_numpy(self.dataset._load_bg(self.dataset.bg_paths[0])).to(self.device))
        
        if not novel_view_mode:
            self.logger.log_final_stats(
                self.rendering_evaluator.psnr_list,
                self.rendering_evaluator.rmse_list,
                self.rendering_evaluator.l1_list,
                self.rendering_evaluator.ssim_list,
                self.rendering_evaluator.lpips_list)
             
        return metrics

    def track(self):
        self.load_dataset()
        self.first_frame_w2c, start_time_idx = self.init_Gaussian_scne()
        self.render_helper = RenderHelper()
        self.optim_handler = OptimHandler(self.config)
        self.rendering_evaluator = RenderingEvaluator(
            self.device,
            self.wandb_run if self.config['use_wandb'] else None,
            save_frames=True,
            eval_dir=self.eval_dir,
            sil_thres=self.config['add_gaussians']['sil_thres_gaussians'],
            viz_config=self.config['viz'],
            get_embeddings=self.config['data']['load_embeddings'],
            config=self.config,
            render_helper=self.render_helper)
        
        # Init Variables to keep track of ground truth poses and runtimes
        self.scene.variables['gt_w2c_all_frames'] = []
        self.logger = Logger(self.config, self.wandb_run, self.eval_dir)

        if start_time_idx != 0:
            time_idx = start_time_idx
        print(f"Starting from time index {start_time_idx}...")

        sec_per_frame = list()
        # Iterate over Scan
        for time_idx in tqdm(range(start_time_idx, self.num_frames)):
            start = time.time()
            curr_data = self.make_data_dict(time_idx) 
            self.optimize_timestep(time_idx, curr_data)
            
            # Checkpoint every iteration
            if time_idx % self.config["checkpoint_interval"] == 0 and self.config['save_checkpoints'] and time_idx != 0:
                ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
                save_params_ckpt(self.scene.params, self.scene.variables, ckpt_output_dir, time_idx)
        
            end = time.time()
            sec_per_frame.append(end-start)

        print(f"Took {sum(sec_per_frame)}, i.e., {sum(sec_per_frame)/len(sec_per_frame)} sec per frame on average")
        duration = sum(sec_per_frame)
        sec_per_frame = sum(sec_per_frame)/len(sec_per_frame)

        if self.config['save_checkpoints']:
            ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
            save_params_ckpt(self.scene.params, self.scene.variables, ckpt_output_dir, time_idx)

        self.logger.log_time_stats()

        # Add Camera Parameters to Save them
        self.scene.update_params_for_saving(duration, sec_per_frame)
        # Save Parameters
        save_params(self.scene.params, self.output_dir)

        # eval renderings, traj and grid vis
        metrics = self._eval(
            eval_renderings=not self.config['eval_during'],
            vis_trajs=self.config['viz']['vis_trajs'],
            vis_grid=self.config['viz']['vis_grid'])

        # Close WandB Run
        if self.config['use_wandb']:
            self.wandb_run.log(metrics)
            wandb.finish()
        
        if self.config['save_checkpoints']:
            ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
            if os.path.isfile(os.path.join(ckpt_output_dir, "temp_params.npz")):
                os.remove(os.path.join(ckpt_output_dir, "temp_params.npz"))
                os.remove(os.path.join(ckpt_output_dir, "temp_variables.npz"))

    def optimize_timestep(self, time_idx, curr_data):
        # track cam
        if self.config['tracking_cam']['num_iters'] != 0 and time_idx > 0:
            self.optimize_camera(
                time_idx,
                curr_data)
        else:
            pass
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                print(curr_data[0]['iter_gt_w2c_list'][-1])
                print(self.first_frame_w2c)
                rel_w2c = curr_data[0]['iter_gt_w2c_list'][-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach().clone()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach().clone()
                # Update the camera parameters
                self.scene.params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                self.scene.params['cam_trans'][..., time_idx] = rel_w2c_tran
                print(self.scene.params['cam_trans'][..., time_idx])
                print(self.scene.params['cam_unnorm_rots'][..., time_idx])

        # Initialize Gaussian poses for the current frame in params

        # Densification
        if (time_idx+1) % self.config['add_every'] == 0 and time_idx > 0:
            self.densify(time_idx, curr_data)

        if self.config['tracking_obj']['num_iters'] != 0:
            _ = self.optimize_gaussians(
                time_idx,
                curr_data)
        
        # Increment WandB Time Step
        if self.config['use_wandb']:
            self.wandb_time_step += 1

        torch.cuda.empty_cache()
        gc.collect()
        
        if (time_idx < self.num_frames-1):
            with torch.no_grad():
                self.scene.variables, to_remove = calculate_neighbors_seg_after_init(
                    self.scene.params,
                    self.scene.variables,
                    time_idx,
                    num_knn=int(self.config['kNN']/self.config['stride']),
                    dist_to_use=self.config['dist_to_use'],
                    primary_device=self.device)

        if (time_idx < self.num_frames-1):
            # Initialize Gaussian poses for the next frame in params
            self.forward_propagate_gaussians(time_idx)            
            # initialize cam pos for next frame
            self.forward_propagate_camera(time_idx, forward_prop=self.config['tracking_cam']['forward_prop'])

        # reset hooks
        for k, p in self.scene.params.items():
            p._backward_hooks: Dict[int, Callable] = OrderedDict()

    def optimize_gaussians(
            self,
            time_idx,
            curr_data):

        config = self.config['tracking_obj']
        lrs = copy.deepcopy(config['lrs'])
        if self.config['use_wandb']:
            wandb_step = self.wandb_obj_tracking_step
        
        # get instance segementation mask for Gaussians
        tracking_start_time = time.time()
        
        # Reset Optimizer & Learning Rates for tracking
        optimizer = self.optim_handler.initialize_optimizer(
            self.scene.params,
            lrs,
            tracking=True)
        optimizer.zero_grad(set_to_none=True)

        if config['take_best_candidate']:
            # Keep Track of Best Candidate Rotation & Translation
            candidate_dyno_rot = self.scene.params['unnorm_rotations'][:, :].detach().clone()
            candidate_dyno_trans = self.scene.params['means3D'][:, :].detach().clone()
            current_min_loss = float(1e20)

        # Tracking Optimization
        iter = 0
        num_iters_tracking = config['num_iters'] if time_idx != 0 else config['num_iters_init']
        progress_bar = tqdm(range(num_iters_tracking), desc=f"Object Tracking Time Step: {time_idx}")

        self.get_hooks(config, time_idx)

        last_loss = 1000
        early_stop_count = 0
        early_stop_eval = False
        while iter <= num_iters_tracking:
            data_idx = random.choice(list(range(len(curr_data))))
            iter_start_time = time.time()
            # Loss for current frame
            loss, losses, visible, weight = self.get_loss_gaussians(
                curr_data[data_idx],
                time_idx,
                num_iters=num_iters_tracking,
                iter=iter,
                config=config)

            # Backprop
            loss.backward()

            if self.config['use_wandb']:
                # Report Loss
                wandb_step = self.logger.report_loss(
                    losses,
                    self.wandb_run,
                    wandb_step,
                    obj_tracking=True)

            with torch.no_grad():
                # Prune Gaussians
                if self.config['prune_densify']['prune_gaussians'] and time_idx > 0:
                    self.scene.params, self.scene.variables, means2d = prune_gaussians(
                        self.scene.params,
                        self.scene.variables,
                        optimizer, 
                        iter,
                        self.config['prune_densify']['pruning_dict'],
                        time_idx,
                        means2d)
                    if self.config['use_wandb']:
                        self.wandb_run.log({"Tracking Object/Number of Gaussians - Pruning": self.scene.params['means3D'].shape[0],
                                        "Mapping/step": self.wandb_mapping_step})
                # Gaussian-Splatting's Gradient-based Densification
                if self.config['prune_densify']['use_gaussian_splatting_densification']:
                    self.scene.params, self.scene.variables, means2d = densify(
                        self.scene.params,
                        self.scene.variables,
                        optimizer,
                        iter,
                        self.config['prune_densify']['densify_dict'],
                        time_idx,
                        means2d)
                    if self.config['use_wandb']:
                        self.wandb_run.log({"Tracking Object/Number of Gaussians - Densification": self.scene.params['means3D'].shape[0],
                                        "Tracking Object/step": self.wandb_mapping_step})

            # Optimizer Update
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if config['take_best_candidate']:
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_dyno_rot = self.scene.params['unnorm_rotations'][:, :, time_idx].detach().clone()
                        candidate_dyno_trans = self.scene.params['means3D'][:, :, time_idx].detach().clone()
            
            # Update the runtime numbers
            iter_end_time = time.time()
            self.logger.tracking_obj_iter_time_sum += iter_end_time - iter_start_time
            self.logger.tracking_obj_iter_time_count += 1
            # Check if we should stop tracking
            iter += 1
            if iter % 50 == 0:
                progress_bar.update(50)

            # early stopping
            early_stop_eval, early_stop_count, last_loss = self.optim_handler.early_check(
                early_stop_count, last_loss, loss, early_stop_eval)
            if early_stop_eval:
                break
        
        progress_bar.close()
        if config['take_best_candidate']:
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                self.scene.params['unnorm_rotations'] = candidate_dyno_rot
                self.scene.params['means3D']= candidate_dyno_trans
        
        # visibility
        for data_idx in range(len(curr_data)):
            _, _, visible, weight = self.get_loss_gaussians(
                curr_data[data_idx],
                time_idx,
                num_iters=num_iters_tracking,
                iter=iter,
                config=config,
                early_stop_eval=early_stop_eval,
                last=True)
            
            optimizer.zero_grad(set_to_none=True)
            self.scene.variables['visibility'][:, data_idx, time_idx] = \
                self.render_helper.compute_visibility(visible, weight, num_gauss=self.scene.params['means3D'].shape[0])
            
        # update params
        for k in ['rgb_colors', 'log_scales', 'means3D', 'unnorm_rotations']:
            self.scnee.update_params(k, time_idx)
        
        if self.config['use_wandb']:
            self.wandb_obj_tracking_step = wandb_step
              
        # Update the runtime numbers
        tracking_end_time = time.time()
        self.logger.tracking_obj_frame_time_sum += tracking_end_time - tracking_start_time
        self.logger.tracking_obj_frame_time_count += 1
        
        # update prev values for l1 losses
        self.scene.ema_update_all_prev()

        return optimizer

    def get_hooks(self, config, time_idx):
        if config['disable_rgb_grads_old'] or config['make_grad_bg_smaller']:
            # get list to turn of grads
            to_turn_off = []
            if not config['loss_weights']['l1_opacity']:
                to_turn_off.append('logit_opacities')          
            if not config['loss_weights']['l1_bg']:
                to_turn_off.append('bg')
            if not config['loss_weights']['l1_embeddings'] or config['make_grad_bg_smaller']:
                to_turn_off.append('embeddings')
            if not config['loss_weights']['l1_scale']:
                to_turn_off.append('log_scales')
            if not config['loss_weights']['l1_rgb']:
                to_turn_off.append('rgb_colors')
            if config['make_grad_bg_smaller']:
                to_turn_off.append('means3D')
            if time_idx == 0:
                print(f"Turning off {to_turn_off}.")
            
            # remove old hooks
            if len(self.hook_list):
                for h in self.hook_list:
                    h.remove()
                self.hook_list = list()
            
            # add hooks
            for k, p in self.scene.params.items():
                if 'cam' in k:
                    continue
                if k not in to_turn_off:
                    continue
                if config['make_grad_bg_smaller'] and k in ["means3D"]:
                    self.hook_list.append(p.register_hook(get_hook(
                        (self.scene.params['bg'].clone().detach() > 0.5).squeeze() & (self.scene.variables['timestep'] != time_idx).squeeze(),
                        grad_weight=config['make_grad_bg_smaller_weight'])))
                else:
                    self.hook_list.append(p.register_hook(get_hook(
                        self.scene.variables['timestep'] != time_idx)))

    def optimize_camera(
            self,
            time_idx,
            curr_data):

        assert len(curr_data) == 1, "Cam tracking currently not implemented for stereo..."

        # get instance segementation mask for Gaussians
        tracking_start_time = time.time()

        # Reset Optimizer & Learning Rates for tracking
        optimizer = self.optim_handler.initialize_optimizer(
            self.scene.params,
            self.config['tracking_cam']['lrs'],
            tracking=True)

        if self.config['tracking_cam']['take_best_candidate']:
            # Keep Track of Best Candidate Rotation & Translation
            candidate_dyno_rot = self.scene.params['cam_unnorm_rots'][:, :, time_idx].detach().clone()
            candidate_dyno_trans = self.scene.params['cam_trans'][:, :, time_idx].detach().clone()
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
            loss, losses, self.scene.variables = self.get_loss_cam(
                curr_data[0],
                time_idx,
                config=self.config['tracking_cam'])

            # Backprop
            loss.backward()

            if self.config['use_wandb']:
                # Report Loss
                self.wandb_obj_tracking_step = self.logger.report_loss(
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
                        candidate_dyno_rot = self.scene.params['cam_unnorm_rots'][:, :, time_idx].detach().clone()
                        candidate_dyno_trans = self.scene.params['cam_trans'][:, :, time_idx].detach().clone()

            # Update the runtime numbers
            iter_end_time = time.time()
            self.logger.tracking_cam_iter_time_sum += iter_end_time - iter_start_time
            self.logger.tracking_cam_iter_time_count += 1
            # Check if we should stop tracking
            iter += 1
            if iter % 50 == 0:
                progress_bar.update(50)

            if iter == num_iters_tracking and loss >= 0.5 and \
                    not restarted_tracking and self.config['tracking_cam']['restart_if_fail']:
                iter = 2
                restarted_tracking = True
                with torch.no_grad():
                    self.scene.params['cam_unnorm_rots'][:, :, time_idx] = self.scene.params[
                        'cam_unnorm_rots'][:, :, time_idx-1].detach().clone()
                    self.scene.params['cam_trans'][:, :, time_idx] = self.scene.params[
                        'cam_trans'][:, :, time_idx-1].detach().clone()

                if self.config['tracking_cam']['take_best_candidate']:
                    current_min_loss = float(1e20)
                    candidate_dyno_rot = self.scene.params['cam_unnorm_rots'][:, :, time_idx].detach().clone()
                    candidate_dyno_trans = self.scene.params['cam_trans'][:, :, time_idx].detach().clone()
            
            # early stopping
            early_stop_eval, early_stop_count, last_loss = self.optim_handler.early_check(
                early_stop_count, last_loss, loss, early_stop_eval)
            if early_stop_eval:
                break

        progress_bar.close()
        if self.config['tracking_cam']['take_best_candidate']:
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                self.scene.params['cam_unnorm_rots'][:, :, time_idx] = candidate_dyno_rot
                self.scene.params['cam_trans'][:, :, time_idx] = candidate_dyno_trans

        # Update the runtime numbers
        tracking_end_time = time.time()
        self.logger.tracking_cam_frame_time_sum += tracking_end_time - tracking_start_time
        self.logger.tracking_cam_frame_time_count += 1
        
        return optimizer

    def densify(self, time_idx, curr_data):
        if self.config['add_gaussians']['add_new_gaussians']:
            for i in  range(len(curr_data)):
            # Add new Gaussians to the scene based on the Silhouette
                pre_num_pts = self.scene.params['means3D'].shape[0]
                self.scene.add_new_gaussians(curr_data[i], 
                                        self.config['add_gaussians']['sil_thres_gaussians'],
                                        self.config['add_gaussians']['depth_error_factor'],
                                        time_idx,
                                        self.config['mean_sq_dist_method'],
                                        self.config['gaussian_distribution'],
                                        self.scene.params, 
                                        self.scene.variables)

                post_num_pts = self.scene.params['means3D'].shape[0]
                if self.config['use_wandb']:
                    self.wandb_run.log({"Adding/Number of Gaussians": post_num_pts-pre_num_pts,
                                    "Adding/step": self.wandb_time_step})





