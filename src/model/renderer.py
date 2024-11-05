from utils.gaussian_utils import build_rotation
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
# from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization_w_dwv import GaussianRasterizer as Renderer
from utils.gaussian_utils import quat_mult


class RenderHelper():

    def mask_timestamp(
            self,
            rendervar,
            timestamp,
            first_occurance,
            strictly_less=False):
        
        if strictly_less:
            time_mask = first_occurance < timestamp
        else:
            time_mask = first_occurance <= timestamp

        masked_rendervar = dict()
        for k, v in rendervar.items():
            masked_rendervar[k] = v[time_mask]
        return masked_rendervar, time_mask

    def transform_to_frame(
            self,
            params,
            time_idx,
            gaussians_grad,
            camera_grad=False,
            gauss_time_idx=None):
        """
        Function to transform Isotropic or Anisotropic Gaussians from world frame to camera frame.
        
        Args:
            params: dict of parameters
            time_idx: time index to transform to
            gaussians_grad: enable gradients for Gaussians
            camera_grad: enable gradients for camera pose
        
        Returns:
            transformed_gaussians: Transformed Gaussians (dict containing means3D & unnorm_rotations)
        """
        all_times = len(params['means3D'].shape) == 3

        if gauss_time_idx is None:
            gauss_time_idx = time_idx

        # Get Frame Camera Pose
        if camera_grad:
            cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx])
            cam_tran = params['cam_trans'][..., time_idx]
        else:
            cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
            cam_tran = params['cam_trans'][..., time_idx].detach()

        rel_w2c = torch.eye(4, device=params['means3D'].device).float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran

        # Check if Gaussians need to be rotated (Isotropic or Anisotropic)
        if params['log_scales'].shape[1] == 1:
            transform_rots = False # Isotropic Gaussians
        else:
            transform_rots = True # Anisotropic Gaussians
        
        # Get Centers and Unnorm Rots of Gaussians in World Frame
        if all_times:
            pts = params['means3D'][:, :, gauss_time_idx]
            unnorm_rots = params['unnorm_rotations'][:, :, gauss_time_idx]
        else:
            pts = params['means3D']
            unnorm_rots = params['unnorm_rotations']
        if not gaussians_grad:
            pts = pts.detach()
            unnorm_rots = unnorm_rots.detach()

        transformed_gaussians = {}
        # Transform Centers of Gaussians to Camera Frame
        pts_ones = torch.ones(pts.shape[0], 1, device=params['means3D'].device).float()
        pts4 = torch.cat((pts, pts_ones), dim=1)
        transformed_pts = (rel_w2c @ pts4.T).T[:, :3]
        transformed_gaussians['means3D'] = transformed_pts

        # Transform Rots of Gaussians to Camera Frame
        if transform_rots:
            norm_rots = F.normalize(unnorm_rots)
            transformed_rots = quat_mult(cam_rot, norm_rots)
            transformed_gaussians['unnorm_rotations'] = transformed_rots
        else:
            transformed_gaussians['unnorm_rotations'] = unnorm_rots

        return transformed_gaussians

    def get_depth_and_silhouette(
            self,
            pts_3D,
            w2c):
        """
        Function to compute depth and silhouette for each gaussian.
        These are evaluated at gaussian center.
        """
        # Depth of each gaussian center in camera frame
        pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
        pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
        depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
        depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]

        # Depth and Silhouette
        depth_silhouette = torch.zeros((pts_3D.shape[0], 3), device=pts_3D.device).float()
        depth_silhouette[:, 0] = depth_z.squeeze(-1)
        depth_silhouette[:, 1] = 1.0
        depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)
        
        return depth_silhouette


    def get_bg(
            self,
            tensor_shape,
            bg):
        """
        Function to compute depth and silhouette for each gaussian.
        These are evaluated at gaussian center.
        """
        # Depth and Silhouette
        depth_silhouette = torch.zeros((tensor_shape, 3), device=bg.device).float()
        depth_silhouette[:, 0] = bg.float().squeeze()
        depth_silhouette[:, 1] = 1
        depth_silhouette[:, 2] = 1
        return depth_silhouette
    
    def get_log_scales(
            self,
            params,
            time_idx):
        """
        Function to get log scales depending on if iso or anisotropic
        """
        if len(params['log_scales'].squeeze().shape) == 1:
            log_scales = params['log_scales'] 
        elif len(params['log_scales'].squeeze().shape) == 2 and params['log_scales'].squeeze().shape[1] == 3:
            log_scales = params['log_scales']
        elif len(params['log_scales'].squeeze().shape) == 2:
            log_scales = params['log_scales'][:, time_idx].unsqueeze(1)
        else:
            log_scales = params['log_scales'][:, :, time_idx]

        if log_scales.shape[1] == 1:
            log_scales = torch.tile(log_scales, (1, 3))
        else:
            log_scales = log_scales
        
        return log_scales

    def get_renderings(
            self,
            params,
            variables,
            iter_time_idx,
            data,
            config={'use_sil_for_loss': False, 'sil_thres': 0.5}, 
            disable_grads=False,
            track_cam=False,
            get_rgb=True,
            get_bg=True,
            get_depth=True,
            get_embeddings=True,
            do_compute_visibility=False,
            last=False,
            gauss_time_idx=None):
        """
        Function to compute all renderings needed for loss computation and 
        visualizations
        """

        transformed_gaussians = self.transform_to_frame(params, iter_time_idx,
                                            gaussians_grad=True if not disable_grads else False,
                                            camera_grad=track_cam,
                                            gauss_time_idx=iter_time_idx if gauss_time_idx is None else gauss_time_idx)
        
        log_scales = self.get_log_scales(params, iter_time_idx)
        rendervar = {
            'means3D': transformed_gaussians['means3D'],
            'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(log_scales),
            'means2D': torch.zeros_like(transformed_gaussians['means3D'], requires_grad=True, device=params['means3D'].device) + 0
        } 
        rendervar, time_mask =  self.mask_timestamp(rendervar, iter_time_idx, variables['timestep'])
        
        if get_rgb:
            # RGB Rendering
            rgb = params['rgb_colors'] if len(params['rgb_colors'].shape) == 2 else params['rgb_colors'][:, :, iter_time_idx]
            rendervar['colors_precomp'] = rgb[time_mask]
            if not disable_grads and not last:
                rendervar['means2D'].retain_grad()
            im, radius, _, weight, visible = Renderer(raster_settings=data['cam'])(**rendervar) 
            variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
            if do_compute_visibility:
                visibility = self.compute_visibility(visible, weight, num_gauss=params['means3D'].shape[0])
            else:
                visibility = None
        else:
            im, radius, weight, visible, visibility = None, None, None, None, None

        if get_depth:
            # Depth & Silhouette Rendering
            rendervar['colors_precomp'] = self.get_depth_and_silhouette(transformed_gaussians['means3D'], data['w2c'])[time_mask]
            depth_sil, _, _, _, _  = Renderer(raster_settings=data['cam'])(**rendervar)

            # silouette
            silhouette = depth_sil[1, :, :]
            presence_sil_mask = (silhouette > config['sil_thres'])
            # depth 
            depth = depth_sil[0, :, :].unsqueeze(0)
            uncertainty = (depth_sil[2, :, :].unsqueeze(0) - depth**2).detach()
            # Mask with valid depth values (accounts for outlier depth values)
            nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
            mask = (data['depth'] > 0) & nan_mask

            # Mask with presence silhouette mask (accounts for empty space)
            if config['use_sil_for_loss']:
                mask = mask & presence_sil_mask
        else:
            mask, depth, silhouette = None, None, None

        if get_bg:
            # BG rendering
            rendervar['colors_precomp'] = self.get_bg(transformed_gaussians['means3D'].shape[0], params['bg'])[time_mask]
            bg, _, _, _, _ = Renderer(raster_settings=data['cam'])(**rendervar)
            # instseg 
            bg = bg[0, :, :].unsqueeze(0)
        else:
            bg = None
        
        if get_embeddings:
            rendered_embeddings = list()
            for emb_idx in range(0, params['embeddings'].shape[1], 3):
                max_idx = min(params['embeddings'].shape[1]-emb_idx, 3)
                embs = params['embeddings'][:, emb_idx:emb_idx+max_idx]
                if max_idx < 3:
                    embs = torch.cat((embs, torch.ones((embs.shape[0], 1), device=embs.device)), dim=-1).float()
                rendervar['colors_precomp'] = embs[time_mask]
                _embeddings, _, _, _, _ = Renderer(raster_settings=data['cam'])(**rendervar)
                rendered_embeddings.append(_embeddings[:max_idx])
            rendered_embeddings = torch.cat(rendered_embeddings, dim=0)
        else:
            rendered_embeddings = None
        
        return variables, im, radius, depth, mask, transformed_gaussians, visible, weight, time_mask, None, silhouette, rendered_embeddings, bg, visibility

    def compute_visibility(
            self,
            visible,
            weight,
            visibility_modus='thresh',
            num_gauss=0):
        """
        Function to compute visibility needed for trajectory evaluation
        """
        w, h, contrib = visible.shape[2], visible.shape[1], visible.shape[0]

        # get max visible gauss per pix
        max_gauss_weight, max_gauss_idx = weight.detach().clone().max(dim=0)
        x_grid, y_grid = torch.meshgrid(torch.arange(w).to(visible.device).long(), 
                                        torch.arange(h).to(visible.device).long(),
                                        indexing='xy')
        max_gauss_id = visible.detach().clone()[
            max_gauss_idx.flatten(),
            y_grid.flatten(),
            x_grid.flatten()].int().reshape(max_gauss_idx.shape)
        
        if visibility_modus == 'max':
            visibility = torch.zeros(num_gauss, dtype=bool)
            visibility[max_gauss_id.flatten().long()] = True
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
        weight_sum_per_gauss = torch.zeros(num_gauss).to(weight_pix_flat.device)
        weight_sum_per_gauss[torch.unique(vis_pix_flat)] = \
            scatter_add(weight_pix_flat, vis_pix_flat)[torch.unique(vis_pix_flat)]

        visibility = weight_sum_per_gauss
        
        return visibility
    
