import cv2
import os
import torch
from tqdm import tqdm
import numpy as np
from src.utils.camera_helpers import setup_camera
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
import imageio
import open3d as o3d
from sklearn.decomposition import PCA
import copy
from src.utils.viz_utils import make_vid, get_cam_poses


class RenderingEvaluator():
    def __init__(self, device, wandb_run, save_frames=True, eval_dir='', sil_thres=0.5, viz_config=None, get_embeddings=True, config=None, render_helper=None):
        self.device = device
        self.loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True)
        self.loss_fn_alex = self.loss_fn_alex.to(self.device)
        self.save_frames = save_frames
        self.wandb_run = wandb_run
        self.psnr_list, self.rmse_list, self.l1_list, self.lpips_list, self.ssim_list = list(), list(), list(), list(), list()
        self.eval_dir = eval_dir
        self.sil_thres = sil_thres
        self.viz_config = viz_config
        self.get_embeddings = get_embeddings
        self.config = config
        self.render_helper = render_helper
        self.pca = None
        self.vmin_depth = 0
        if 'panoptic' in self.eval_dir:
            self.vmax_depth = 20
            self.fps = 30
        elif'iphone' in self.eval_dir:
            self.vmax_depth = 5
            self.fps = 30 if self.config['data']['sequence'] not in ['haru-sit', 'mochi-high-five'] else 60
        else:
            self.vmax_depth = 80
            self.fps = 24
        
    @staticmethod
    def save_pc(final_params_time, save_dir, time_idx, time_mask):
        pcd = o3d.geometry.PointCloud()
        v3d = o3d.utility.Vector3dVector
        pcd.points = v3d(final_params_time['means3D'][:, :, time_idx][time_mask].cpu().numpy())
        o3d.io.write_point_cloud(filename=os.path.join(save_dir, "pc_{:04d}_all.xyz".format(time_idx)), pointcloud=pcd)

    def save_normalized(self, img, save_dir, time_idx, vmin=None, vmax=None, num_frames=100):
        img = img[0].detach().cpu().numpy()
        if vmin is None:
            vmax, vmin = img.max(), img.min()
        normalized = np.clip((img - vmin) / (vmax - vmin + 1e-10), 0, 1)
        colormap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_dir, "gs_{:04d}.png".format(time_idx)), colormap)
        if time_idx == num_frames - 1: 
            make_vid(save_dir, self.fps)

    def save_rgb(self, img, save_dir, time_idx, num_frames):
        viz_gt_im = torch.clamp(img, 0, 1)
        viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
        cv2.imwrite(os.path.join(save_dir, "gt_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))
        if time_idx == num_frames - 1: 
            make_vid(save_dir, self.fps)

    def save_pca_downscaled(self, features, save_dir, time_idx, num_frames):
        features = features.permute(1, 2, 0).detach().cpu().numpy()
        shape = features.shape
        if shape[2] != 3:
            if self.pca is None:
                self.pca = PCA(n_components=3)
                self.pca.fit(features.reshape(-1, shape[2]))
            features = self.pca.transform(
                features.reshape(-1, shape[2]))
            features = features.reshape(
                (shape[0], shape[1], 3))
            self.vmax_emb, self.vmin_emb = features.max(), features.min()
        normalized_features = np.clip((features - self.vmin_emb) / (self.vmax_emb - self.vmin_emb + 1e-10), 0, 1)
        normalized_features_colormap = (normalized_features * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(save_dir, "gs_{:04d}.png".format(time_idx)), normalized_features_colormap)
        if time_idx == num_frames - 1: 
            make_vid(save_dir, self.fps)
            
    @staticmethod
    def calc_psnr(img1, img2):
        mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def eval(
            self,
            dataset,
            final_params,
            num_frames,
            variables=None,
            novel_view_mode=None,):
        
        if final_params['log_scales'].shape[1] == num_frames and len(final_params['log_scales'].shape) == 3:
            final_params['log_scales'] = final_params['log_scales'].permute(0, 2, 1)
        
        print("Evaluating Final Parameters ...")
        if novel_view_mode is not None:
            poses, name = get_cam_poses(novel_view_mode, dataset, self.config, num_frames, final_params['means3D'].device, final_params)
            torch.save(poses.cpu(), os.path.join(self.eval_dir, f'poses_{name}.pth'))
            print('Store to', os.path.join(self.eval_dir, f'poses_{name}.pth'))
            print(f"Evaluating novel view in mode {novel_view_mode}!!")
        else:
            name = ''

        if self.save_frames:
            dir_names = self.make_dirs(name=name)
        
        pca = None
        visibilities = list()
        for time_idx in tqdm(range(num_frames)):
            final_params_time = copy.deepcopy(final_params)
            # Get RGB-D Data & Camera Parameters
            color, depth, intrinsics, pose, embeddings, bg, instseg = dataset[time_idx]

            # Process Camera Parameters
            intrinsics = intrinsics[:3, :3]
            if novel_view_mode is not None:
                pose = poses[time_idx]
                w2c = torch.linalg.inv(pose)
            else:
                w2c = final_params_time['w2c']
            
            if not isinstance(w2c, torch.Tensor):
                w2c = torch.from_numpy(w2c).to(final_params['means3D'].device)

            # Setup Camera
            cam = setup_camera(
                color.shape[2],
                color.shape[1],
                intrinsics.cpu().numpy(),
                w2c.detach().cpu().numpy(),
                device=final_params_time['means3D'].device)
            
            # Define current frame data
            curr_data = {
                'cam': cam,
                'im': color,
                'depth': depth,
                'id': time_idx,
                'intrinsics': intrinsics,
                'w2c': w2c,
                'embeddings': embeddings,
                'bg': bg,
                'instseg': instseg,
                'iter_gt_w2c_list': variables['gt_w2c_all_frames']}
            
            variables, im, radius, rastered_depth, _, _, _, _, time_mask, _, rastered_sil, rendered_embeddings, rastered_bg, visibility = self.render_helper.get_renderings(
                final_params_time,
                variables,
                time_idx,
                curr_data,
                {'sil_thres': self.sil_thres, 'use_sil_for_loss': False, 'use_flow': 'rendered', 'depth_cam': 'cam', 'embedding_cam': 'cam'},
                disable_grads=True,
                track_cam=False,
                do_compute_visibility=True)

            visibilities.append(visibility)
            
            valid_depth_mask = (curr_data['depth'] > 0)
            
            if novel_view_mode is None:
                # Render RGB and Calculate PSNR
                weighted_im = im * valid_depth_mask
                weighted_gt_im = curr_data['im'] * valid_depth_mask
                psnr = self.calc_psnr(weighted_im, weighted_gt_im).mean()
                try:
                    ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                                data_range=1.0, size_average=True)
                except:
                    ssim = torch.tensor(0)
                lpips_score = self.loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                            torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

                self.psnr_list.append(psnr.cpu().numpy())
                self.ssim_list.append(ssim.cpu().numpy())
                self.lpips_list.append(lpips_score)

                # Compute Depth RMSE
                masked_rastered_depth = rastered_depth * valid_depth_mask
                diff_depth_rmse = torch.sqrt((((masked_rastered_depth - curr_data['depth'])) ** 2))
                diff_depth_rmse = diff_depth_rmse * valid_depth_mask
                rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
                diff_depth_l1 = torch.abs((masked_rastered_depth - curr_data['depth']))
                diff_depth_l1 = diff_depth_l1 * valid_depth_mask
                depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
                self.rmse_list.append(rmse.cpu().numpy())
                self.l1_list.append(depth_l1.cpu().numpy())

            if self.save_frames:
                if self.viz_config['vis_all']:
                    # Save Rendered RGB and Depth
                    self.save_rgb(im, dir_names['render_rgb_dir'], time_idx, num_frames)

                if self.viz_config['vis_all']:
                    # depth
                    self.save_normalized(rastered_depth.detach(), dir_names['render_depth_dir'], time_idx, self.vmin_depth, self.vmax_depth, num_frames)

                # bg
                if self.viz_config['vis_all']:
                    self.save_normalized(rastered_bg, dir_names['render_bg_dir'], time_idx, num_frames=num_frames)

                # embeddings
                if rendered_embeddings is not None and self.viz_config['vis_all']:
                    self.save_pca_downscaled(rendered_embeddings, dir_names['render_emb_dir'], time_idx, num_frames)
                    self.save_pca_downscaled(curr_data['embeddings'], dir_names['render_emb_gt_dir'], time_idx, num_frames)

                # if self.viz_config['vis_all']:
                #     # silouette
                #     self.save_normalized(rastered_sil.unsqueeze(0).detach().float(), dir_names['render_sil_dir'], time_idx, num_frames=num_frames)

                if self.viz_config['vis_gt'] and novel_view_mode is None:
                    # Save GT RGB and Depth
                    self.save_rgb(curr_data['im'], dir_names['rgb_dir'], time_idx, num_frames)
                    # depth
                    self.save_normalized(curr_data['depth'], dir_names['depth_dir'], time_idx, self.vmin_depth, self.vmax_depth, num_frames)
                    # instseg
                    self.save_normalized(curr_data['instseg'], dir_names['instseg_dir'], time_idx, num_frames=num_frames)                
                    # bg 
                    self.save_normalized(curr_data['bg'].float(), dir_names['bg_dir'], time_idx, num_frames=num_frames)

            if self.viz_config['save_pc']:
                self.save_pc(final_params_time, dir_names['pc_dir'], time_idx, time_mask)

        return visibilities

    def make_dirs(self, name=''):
        dir_names = dict()
        if self.viz_config['vis_all']:
            dir_names['render_rgb_dir'] = os.path.join(self.eval_dir, f"rendered_rgb_{name}")
            os.makedirs(dir_names['render_rgb_dir'], exist_ok=True)
        
        if self.viz_config['vis_all']:
            dir_names['render_depth_dir'] = os.path.join(self.eval_dir, f"rendered_depth_{name}")
            os.makedirs(dir_names['render_depth_dir'], exist_ok=True)

        if self.viz_config['vis_all']:
            dir_names['render_emb_dir'] = os.path.join(self.eval_dir, f"pca_emb_{name}")
            dir_names['render_emb_gt_dir'] = os.path.join(self.eval_dir, f"pca_emb_gt_{name}")
            os.makedirs(dir_names['render_emb_dir'], exist_ok=True)
            os.makedirs(dir_names['render_emb_gt_dir'], exist_ok=True)

        if self.viz_config['vis_all']:
            dir_names['render_sil_dir'] = os.path.join(self.eval_dir, f"rendered_sil_{name}")
            os.makedirs(dir_names['render_sil_dir'], exist_ok=True)

        if self.viz_config['save_pc']:
            dir_names['pc_dir'] = os.path.join(self.eval_dir, "pc")
            os.makedirs(dir_names['pc_dir'], exist_ok=True)

        if self.viz_config['vis_all']:
            dir_names['render_bg_dir'] = os.path.join(self.eval_dir, f"rendered_bg_{name}")
            os.makedirs(dir_names['render_bg_dir'], exist_ok=True)
        
        if self.viz_config['vis_gt']:
            dir_names['rgb_dir'] = os.path.join(self.eval_dir, "rgb")
            dir_names['depth_dir'] = os.path.join(self.eval_dir, "depth")
            dir_names['instseg_dir'] = os.path.join(self.eval_dir, "instseg")
            dir_names['bg_dir'] = os.path.join(self.eval_dir, "bg")  
            os.makedirs(dir_names['rgb_dir'], exist_ok=True)
            os.makedirs(dir_names['depth_dir'], exist_ok=True)
            os.makedirs(dir_names['instseg_dir'], exist_ok=True)
            os.makedirs(dir_names['bg_dir'], exist_ok=True)

        return dir_names

    def eval_during(
            self,
            curr_data,
            time_idx,
            im,
            rastered_depth,
            rastered_sil,
            rastered_bg,
            rendered_embeddings,
            pca=None,
            num_frames=-1):

        if self.save_frames:
            dir_names = self.make_dirs()
        
        valid_depth_mask = (curr_data['depth'] > 0)
        rastered_depth = rastered_depth * valid_depth_mask
        rastered_sil = rastered_sil.unsqueeze(0)

        # Render RGB and Calculate PSNR
        weighted_im = im * valid_depth_mask
        weighted_gt_im = curr_data['im'] * valid_depth_mask
        psnr = self.calc_psnr(weighted_im, weighted_gt_im).mean()
        try:
            ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                        data_range=1.0, size_average=True)
        except:
            ssim = torch.tensor(0)
        lpips_score = self.loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                    torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

        diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth'])) ** 2))
        diff_depth_rmse = diff_depth_rmse * valid_depth_mask
        rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
        diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']))
        diff_depth_l1 = diff_depth_l1 * valid_depth_mask
        depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        if self.save_frames:
            if self.viz_config['vis_all']:
                # Save Rendered RGB and Depth
                self.save_rgb(im, dir_names['render_rgb_dir'], time_idx, num_frames)

            if self.viz_config['vis_all']:
                # depth
                self.save_normalized(rastered_depth.detach(), dir_names['render_depth_dir'], time_idx, self.vmin_depth, self.vmax_depth, num_frames)

            # bg
            if self.viz_config['vis_all']:
                self.save_normalized(rastered_bg, dir_names['render_bg_dir'], time_idx, num_frames=num_frames)

            # embeddings
            if rendered_embeddings is not None and self.viz_config['vis_all']:
                self.save_pca_downscaled(rendered_embeddings, dir_names['render_emb_dir'], time_idx, num_frames)
                self.save_pca_downscaled(curr_data['embeddings'], dir_names['render_emb_gt_dir'], time_idx, num_frames)

            if self.viz_config['vis_gt']:
                # Save GT RGB and Depth
                self.save_rgb(curr_data['im'], dir_names['rgb_dir'], time_idx, num_frames=num_frames)
                # depth
                self.save_normalized(curr_data['depth'], dir_names['depth_dir'], time_idx, self.vmin_depth, self.vmax_depth, num_frames=num_frames)
                # instseg
                self.save_normalized(curr_data['instseg'], dir_names['instseg_dir'], time_idx, num_frames=num_frames)                
                # bg 
                self.save_normalized(curr_data['bg'].float(), dir_names['bg_dir'], time_idx, num_frames=num_frames)

        self.psnr_list.append(psnr.cpu().numpy())
        self.rmse_list.append(rmse.cpu().numpy())
        self.l1_list.append(depth_l1.cpu().numpy())
        self.ssim_list.append(ssim.cpu().numpy())
        self.lpips_list.append(lpips_score)

