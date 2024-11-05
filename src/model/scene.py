import torch
import numpy as np
import os
from utils.camera_helpers import setup_camera
from utils.gaussian_utils import build_rotation
from utils.neighbor_search import torch_3d_knn
from diff_gaussian_rasterization_w_dwv import GaussianRasterizer as Renderer


class GaussianScene():
    def __init__(self, config, render_helper, load_embeddings, num_frames, device):
        self.config = config
        self.render_helper = render_helper
        self.load_embeddings = load_embeddings
        self.num_frames = num_frames
        self.device = device
    
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
            bg=None):

        # downscale 
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
        xx = (x_grid - intrinsics[0][2])/intrinsics[0][0]
        yy = (y_grid - intrinsics[1][2])/intrinsics[1][1]

        xx = xx[::self.config['stride'], ::self.config['stride']]
        yy = yy[::self.config['stride'], ::self.config['stride']]
        mask = mask[:, ::self.config['stride'], ::self.config['stride']]

        # get pixel grid into 3D
        mask = mask.reshape(-1)
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
            scale_gaussian = depth_z / ((intrinsics[0][0] + intrinsics[1][1])/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

        # Colorize point cloud
        cols = color.permute(1, 2, 0) # (C, H, W) -> (H, W, C) -> (H * W, C)
        cols = cols[::self.config['stride'], ::self.config['stride']].reshape(-1, 3)
        point_cld = torch.cat((pts, cols), -1)
        if instseg is not None:
            instseg = instseg.permute(1, 2, 0)
            instseg = instseg[::self.config['stride'], ::self.config['stride']].reshape(-1, 1) # (C, H, W) -> (H, W, C) -> (H * W, C)
            point_cld = torch.cat((point_cld, instseg), -1)
        if embeddings is not None:
            channels = embeddings.shape[0]
            embeddings = torch.permute(embeddings, (1, 2, 0))
            embeddings = embeddings[::self.config['stride'], ::self.config['stride']].reshape(-1, channels) # (C, H, W) -> (H, W, C) -> (H * W, C)
            point_cld = torch.cat((point_cld, embeddings), -1)

        # mask background points and incoming points
        if mask is not None:
            point_cld = point_cld[mask]
            mean3_sq_dist = mean3_sq_dist[mask]
            if bg is not None:
                bg = bg[mask]

        return point_cld, mean3_sq_dist, bg

    def get_complete_pointcloud(self):
        
        params = np.load(os.path.join(
            '../Dynamic3DGaussians/experiments/output_first/exp1/',
            os.path.dirname(os.path.dirname(self.config['data']['sequence'])),
            'params.npz'))
        point_cld = torch.from_numpy(params['means3D']).to(self.device)
        color = torch.from_numpy(params['rgb_colors']).to(self.device)
        seg_colors = torch.from_numpy(params['seg_colors']).to(self.device)[:, 2].unsqueeze(-1)
        point_cld = torch.cat([point_cld, color, seg_colors], dim=-1)
        transformed_pts = point_cld[:, :3]
        
        dist, _ = torch_3d_knn(point_cld[:, :3].contiguous().float(), num_knn=4)
        dist = dist[:, 1:]
        mean3_sq_dist = dist.mean(-1).clip(min=0.0000001)
        point_cld = point_cld[mean3_sq_dist<0.01]
        transformed_pts = transformed_pts[mean3_sq_dist<0.01]
        mean3_sq_dist = mean3_sq_dist[mean3_sq_dist<0.01]

        point_cld[:, -1] = point_cld[:, -1]
        bg = point_cld[:, 6].unsqueeze(1)

        return point_cld, mean3_sq_dist, bg
    
    def initialize_params(
            self,
            init_pt_cld,
            mean3_sq_dist,
            gaussian_distribution,
            bg):
        
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
        cam_rots = np.tile(cam_rots[:, :, None], (1, 1, self.num_frames))
        params['cam_unnorm_rots'] = cam_rots
        params['cam_trans'] = np.zeros((1, 3, self.num_frames))
        # params['cam_dfx_dfy_dcx_dcy'] = np.zeros((1, 4))

        if self.load_embeddings:
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
            "means3D": means3D.cpu(),
            "unnorm_rotations": unnorm_rotations.cpu()}
        
        self.params = params
        self.variables = variables


    def initialize_timestep(self, scene_radius_depth_ratio, \
            mean_sq_dist_method, gaussian_distribution=None, timestep=0, w2c=None, data=None):
        # Get RGB-D Data & Camera Parameters
        embeddings = None
        color, depth, self.intrinsics, pose, embeddings, bg, instseg = data

        # Process Camera Parameters
        self.intrinsics = self.intrinsics[:3, :3]

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
        if self.config['data']['start_from_complete_pc']:
            init_pt_cld, mean3_sq_dist, bg = self.get_complete_pointcloud()
        else:
            init_pt_cld, mean3_sq_dist, bg = self.get_pointcloud(
                color,
                depth,
                self.intrinsics,
                w2c,
                mean_sq_dist_method=mean_sq_dist_method,
                embeddings=embeddings,
                bg=bg,
                instseg=instseg)

        # Initialize Parameters
        self.initialize_params(
            init_pt_cld,
            mean3_sq_dist,
            gaussian_distribution,
            bg)
        
        # Initialize an estimate of scene radius for Gaussian-Splatting Densification
        self.variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio
        return w2c
    
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

        if self.load_embeddings:
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
            'means3D': torch.zeros((num_pts, 3, self.num_frames), dtype=torch.float32),
            'unnorm_rotations': params['unnorm_rotations'].detach().clone().unsqueeze(2).repeat(1, 1, self.num_frames).cpu()}
        variables['means3D'][:, :, :time_idx+1] = params['means3D'].detach().clone().unsqueeze(2).repeat(1, 1, time_idx+1).cpu()

        return params, variables
    
    def get_depth_for_new_gauss(
            self,
            time_idx,
            curr_data,
            gauss_time_idx=None):
        
        # Silhouette Rendering        
        _, _, _, depth, _, _, _, _, _, _, silhouette, _, _, _ = \
            self.render_helper.get_renderings(
                self.params,
                self.variables,
                time_idx,
                curr_data,
                get_rgb=False,
                get_bg=False,
                get_depth=True,
                get_embeddings=False,
                disable_grads=True,
                track_cam=False,
                gauss_time_idx=gauss_time_idx)
        
        gt_depth = curr_data['depth'][0, :, :]

        return depth, silhouette, gt_depth
    
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
            do_store_vis=False):
        
        # get depth for adding gaussians and rescaled cam
        render_depth, silhouette, gt_depth = self.get_depth_for_new_gauss(
            time_idx,
            curr_data)

        non_presence_sil_mask = (silhouette < sil_thres)
        if do_store_vis:
            self.store_vis(time_idx, silhouette, 'presence_mask')
        # Check for new foreground objects by u sing GT depth

        if self.config['add_gaussians']['use_depth_error_for_adding_gaussians']:
            depth_error = (torch.abs(gt_depth - render_depth)/gt_depth) * (gt_depth > 0)
            non_presence_depth_mask =  (depth_error > depth_error_factor*depth_error.median()) # * (render_depth > gt_depth)
            non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
            
            if do_store_vis:
                # depth error
                depth_error_store = (depth_error-depth_error.min())/(depth_error.max()-depth_error.min())
                self.store_vis(time_idx, depth_error_store, 'depth_error')
                # non presence depth mask
                self.store_vis(time_idx, non_presence_depth_mask, 'non_presence_depth_mask')

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
            curr_c02c = torch.eye(4).to(self.device).float()
            curr_c02c[:3, :3] = build_rotation(curr_cam_rot)
            curr_c02c[:3, 3] = curr_cam_tran
            # TODO CODE CLEAN UP --> check this!! 
            # curr_w2c = curr_data['w2c'] @ curr_c2c0
            curr_w2c = curr_data['w2c'] @ curr_c02c

            new_pt_cld, mean3_sq_dist, bg = self.get_pointcloud(
                curr_data['im'],
                curr_data['depth'],
                curr_data['intrinsics'], 
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

    def update_params_for_saving(self, duration, sec_per_frame, first_frame_w2c, orig_width, orig_height, desired_width, desired_height):
        # Add Camera Parameters to Save them
        self.params['timestep'] = self.variables['timestep']
        self.params['intrinsics'] = self.intrinsics.detach().cpu().numpy()
        self.params['w2c'] = first_frame_w2c.detach().cpu().numpy()
        self.params['orig_width'] = orig_width
        self.params['orig_height'] = orig_height
        self.params['desired_width'] = desired_width
        self.params['desired_height'] = desired_height
        self.params['gt_w2c_all_frames'] = torch.stack(self.variables['gt_w2c_all_frames']).detach().cpu().numpy()

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

    def update_params(self, k, time_idx):
        if k in self.variables.keys():
            if len(self.variables[k].shape) == 3:
                self.variables[k][:, :, time_idx] = self.params[k].detach().clone().squeeze()
            else:
                self.variables[k][:, time_idx] = self.params[k].detach().clone().squeeze()
    
    def ema_update_all_prev(self):
        for key in ['prev_bg', 'prev_embeddings', 'prev_rgb_colors', 'prev_log_scales', 'prev_logit_opacities']:
            try:
                self.variables[key] = self.ema_update(key, key[5:])
            except:
                if key != 'prev_embeddings' or 'embeddings' in self.params.keys():
                    self.variables[key] = self.params[key[5:]]

    def ema_update(self, key, prev_key):
        return (1-self.config['ema']) * self.params[key].detach().clone() \
            + self.config['ema'] * self.variables[prev_key]