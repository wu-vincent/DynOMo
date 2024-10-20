from utils.slam_external import build_rotation
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from diff_gaussian_rasterization import GaussianRasterizer as Renderer


class RenderHelper():

    def mask_timestamp(self, rendervar, timestamp, first_occurance, moving_mask=None, strictly_less=False):
        if strictly_less:
            time_mask = first_occurance < timestamp
        else:
            time_mask = first_occurance <= timestamp

        if moving_mask is not None:
            time_mask = time_mask & moving_mask

        masked_rendervar = dict()
        for k, v in rendervar.items():
            masked_rendervar[k] = v[time_mask]
        return masked_rendervar, time_mask

    def transformed_params2rendervar(self, params, transformed_gaussians, time_idx, first_occurance, log_scales):
        rgb = params['rgb_colors'] if len(params['rgb_colors'].shape) == 2 else params['rgb_colors'][:, :, time_idx]

        # Initialize Render Variables
        rendervar = {
            'means3D': transformed_gaussians['means3D'],
            'colors_precomp': rgb,
            'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(log_scales),
            'means2D': torch.zeros_like(transformed_gaussians['means3D'], requires_grad=True, device=params['means3D'].device) + 0
        }
        rendervar, time_mask =  self.mask_timestamp(rendervar, time_idx, first_occurance)
        return rendervar, time_mask

    def transformed_params2depthplussilhouette(self, params, w2c, transformed_gaussians, time_idx, first_occurance, log_scales):
        # Initialize Render Variables
        rendervar = {
            'means3D': transformed_gaussians['means3D'],
            'colors_precomp': self.get_depth_and_silhouette(transformed_gaussians['means3D'], w2c),
            'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(log_scales),
            'means2D': torch.zeros_like(transformed_gaussians['means3D'], requires_grad=True, device=params['means3D'].device) + 0
        }
        rendervar, time_mask = self.mask_timestamp(rendervar, time_idx, first_occurance)
        return rendervar, time_mask

    def transformed_params2instsegbg(
            self,
            params,
            transformed_gaussians,
            time_idx,
            variables,
            log_scales):
        # Initialize Render Variables
        rendervar = {
            'means3D': transformed_gaussians['means3D'],
            'colors_precomp': self.get_instsegbg(transformed_gaussians['means3D'].shape[0], params['instseg'], params['bg']),
            'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(log_scales),
            'means2D': torch.zeros_like(transformed_gaussians['means3D'], requires_grad=True, device=params['means3D'].device) + 0,
        }
        rendervar, time_mask = self.mask_timestamp(rendervar, time_idx, variables['timestep'])
        return rendervar, time_mask

    def transformed_params2emb(
            self,
            params,
            transformed_gaussians,
            time_idx,
            variables,
            emb_idx,
            max_idx,
            log_scales):
        # Initialize Render Variables
        # embs = torch.ones([params['embeddings'].shape[0], 3], device=params['embeddings'].device)
        embs = params['embeddings'][:, emb_idx:emb_idx+max_idx]
        if max_idx < 3:
            embs = torch.cat((embs, torch.ones((embs.shape[0], 1), device=embs.device)), dim=-1).float()

        rendervar = {
            'means3D': transformed_gaussians['means3D'],
            'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(log_scales),
            'means2D': torch.zeros_like(transformed_gaussians['means3D'], requires_grad=True, device=params['means3D'].device) + 0,
            'colors_precomp': embs
        }

        rendervar, time_mask = self.mask_timestamp(rendervar, time_idx, variables['timestep'])

        return rendervar, time_mask

    def transform_to_frame(
            self,
            params,
            time_idx,
            gaussians_grad,
            camera_grad=False,
            gauss_time_idx=None,
            delta=0,
            variables=None):
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

    def get_depth_and_silhouette(self, pts_3D, w2c):
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


    def get_instsegbg(self, tensor_shape, instseg, bg):
        """
        Function to compute depth and silhouette for each gaussian.
        These are evaluated at gaussian center.
        """
        # Depth and Silhouette
        depth_silhouette = torch.zeros((tensor_shape, 3), device=bg.device).float()
        depth_silhouette[:, 0] = bg.float().squeeze()
        depth_silhouette[:, 1] = 1
        depth_silhouette[:, 2] = instseg.squeeze()
        return depth_silhouette
    
    def get_log_scales(self, params, time_idx):
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
            config, 
            disable_grads=False,
            track_cam=False,
            get_rgb=True,
            get_depth=True,
            get_seg=False,
            get_embeddings=True,
            delta=False,
            do_compute_visibility=False):

        transformed_gaussians = self.transform_to_frame(params, iter_time_idx,
                                            gaussians_grad=True if not disable_grads and not track_cam else False,
                                            camera_grad=track_cam,
                                            delta=delta)
        
        log_scales = self.get_log_scales(params, iter_time_idx)    

        if get_rgb:
            # RGB Rendering
            rendervar, time_mask = self.transformed_params2rendervar(
                params,
                transformed_gaussians,
                iter_time_idx,
                first_occurance=variables['timestep'],
                log_scales=log_scales)
            if not disable_grads:
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
            depth_sil_rendervar, _ = self.transformed_params2depthplussilhouette(
                params,
                data['w2c'],
                transformed_gaussians,
                iter_time_idx,
                first_occurance=variables['timestep'])
            depth_sil, _, _, _, _  = Renderer(raster_settings=data['cam'])(**depth_sil_rendervar)

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

        if get_seg:
            # Instseg rendering
            seg_rendervar, _ = self.transformed_params2instsegbg(
                params,
                transformed_gaussians,
                iter_time_idx,
                variables)
            instseg, _, _, _, _ = Renderer(raster_settings=data['cam'])(**seg_rendervar)
            # instseg 
            bg = instseg[0, :, :].unsqueeze(0)
        else:
            bg = None
        
        if get_embeddings:
            rendered_embeddings = list()
            for emb_idx in range(0, params['embeddings'].shape[1], 3):
                max_idx = min(params['embeddings'].shape[1]-emb_idx, 3)
                emb_rendervar, _ = self.transformed_params2emb(
                    params,
                    transformed_gaussians,
                    iter_time_idx,
                    variables,
                    emb_idx,
                    max_idx)
                _embeddings, _, _, _, _ = Renderer(raster_settings=data['cam'])(**emb_rendervar)
                rendered_embeddings.append(_embeddings[:max_idx])
            rendered_embeddings = torch.cat(rendered_embeddings, dim=0)
        else:
            rendered_embeddings = None
        return variables, im, radius, depth, mask, transformed_gaussians, visible, weight, time_mask, None, silhouette, rendered_embeddings, bg, visibility

    def compute_visibility(self, visible, weight, visibility_modus='thresh', get_norm_pix_pos=False, thresh=0.5, num_gauss=0):
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
        weight_sum_per_gauss = torch.zeros(num_gauss).to(weight_pix_flat.device)
        weight_sum_per_gauss[torch.unique(vis_pix_flat)] = \
            scatter_add(weight_pix_flat, vis_pix_flat)[torch.unique(vis_pix_flat)]

        if visibility_modus == 'thresh':
            visibility = weight_sum_per_gauss
            if not get_norm_pix_pos:
                return visibility

        weighted_norm_x, weighted_norm_y = self.get_weighted_pix_pos(
            params,
            w,
            h,
            pix_id,
            vis_pix_flat,
            weight_pix_flat,
            weight_sum_per_gauss,
            num_gauss)
        
        return visibility, weighted_norm_x, weighted_norm_y
        
    def get_weighted_pix_pos(self, w, h, pix_id, vis_pix_flat, weight_pix_flat, weight_sum_per_gauss, num_gauss):
        # normalize weights per Gaussian for normalized pix position
        weight_pix_flat_norm = weight_pix_flat/(weight_sum_per_gauss[vis_pix_flat])
        
        x_grid, y_grid = torch.meshgrid(torch.arange(w).to(vis_pix_flat.device).float(), 
                                        torch.arange(h).to(vis_pix_flat.device).float(),
                                        indexing='xy')
        x_grid = (x_grid.flatten()+1)/w
        y_grid = (y_grid.flatten()+1)/h

        # initializing necessary since last gaussian might be invisible
        weighted_norm_x = torch.zeros(num_gauss).to(weight_pix_flat.device)
        weighted_norm_y = torch.zeros(num_gauss).to(weight_pix_flat.device)
        weighted_norm_x[torch.unique(vis_pix_flat)] = \
            scatter_add(x_grid[pix_id] * weight_pix_flat_norm, vis_pix_flat)[torch.unique(vis_pix_flat)]
        weighted_norm_y[torch.unique(vis_pix_flat)] = \
            scatter_add(y_grid[pix_id] * weight_pix_flat_norm, vis_pix_flat)[torch.unique(vis_pix_flat)]
        
        weighted_norm_x[0] = 1/w
        weighted_norm_y[0] = 1/h

        return weighted_norm_x, weighted_norm_y
