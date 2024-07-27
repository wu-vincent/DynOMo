import cv2
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from datasets.gradslam_datasets.geometryutils import relative_transformation
from utils.camera_helpers import setup_camera
from utils.slam_external import build_rotation, calc_psnr
from utils.slam_helpers import (
    three2two,
    get_renderings,
)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import imageio.v2 as iio
import numpy as np

import imageio
import glob
import open3d as o3d
import wandb
from sklearn.decomposition import PCA
import copy
import json
import random

  
def make_vid(input_path): 
    images = list()
    img_paths = glob.glob(f'{input_path}/*')
    for f in sorted(img_paths):
        if 'mp4' in f:
            continue
        images.append(imageio.imread(f))

    imageio.mimwrite(os.path.join(input_path, 'vid_trails.mp4'), np.stack(images), quality=8, fps=10)
    for f in img_paths:
        if 'mp4' in f:
            continue
        os.remove(f)

def report_loss(
        losses,
        wandb_run,
        wandb_step,
        cam_tracking=False,
        obj_tracking=False,
        delta_optim=False,
        refine=False):

    # Update loss dict
    if cam_tracking:
        tracking_loss_dict = {}
        for k, v in losses.items():
            tracking_loss_dict[f"Per Iteration Cam Tracking/{k}"] = v.item()
        tracking_loss_dict['Per Iteration Cam Tracking/step'] = wandb_step
        wandb_run.log(tracking_loss_dict)
    elif obj_tracking:
        tracking_loss_dict = {}
        for k, v in losses.items():
            tracking_loss_dict[f"Per Iteration Object Tracking/{k}"] = v.item()
        tracking_loss_dict['Per Iteration Object Tracking/step'] = wandb_step
        wandb_run.log(tracking_loss_dict)
    elif refine:
        tracking_loss_dict = {}
        for k, v in losses.items():
            tracking_loss_dict[f"Per Iteration Refine/{k}"] = v.item()
        tracking_loss_dict['Per Iteration Refine/step'] = wandb_step
        wandb_run.log(tracking_loss_dict)
    elif delta_optim:
        delta_loss_dict = {}
        for k, v in losses.items():
            delta_loss_dict[f"Per Iteration Delta Optim/{k}"] = v.item()
        delta_loss_dict['Per Iteration Delta Optim/step'] = wandb_step
        wandb_run.log(delta_loss_dict)
    
    # Increment wandb step
    wandb_step += 1
    return wandb_step


def tensor2param(tensor, device):
    if not isinstance(tensor, torch.Tensor):
        return torch.nn.Parameter(torch.tensor(tensor).to(device).float().contiguous().requires_grad_(True))
    else:
        return torch.nn.Parameter(tensor.to(device).float().contiguous().requires_grad_(True))

def param2tensor(param, device):
    return torch.nn.Tensor(param.to(device).float())


def save_pc(final_params_time, save_dir, time_idx, time_mask):
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(final_params_time['means3D'][:, :, time_idx][time_mask].cpu().numpy())
    o3d.io.write_point_cloud(filename=os.path.join(save_dir, "pc_{:04d}_all.xyz".format(time_idx)), pointcloud=pcd)


def save_normalized(img, save_dir, time_idx, vmin=None, vmax=None, num_frames=100):
    img = img[0].detach().cpu().numpy()
    if vmin is None:
        vmax, vmin = img.max(), img.min()
    normalized_depth = np.clip((img - vmin) / (vmax - vmin + 1e-10), 0, 1)
    depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(save_dir, "gs_{:04d}.png".format(time_idx)), depth_colormap)
    if time_idx == num_frames - 1: 
        make_vid(save_dir)


def save_rgb(img, save_dir, time_idx, num_frames):
    viz_gt_im = torch.clamp(img, 0, 1)
    viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
    cv2.imwrite(os.path.join(save_dir, "gt_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))
    if time_idx == num_frames - 1: 
        make_vid(save_dir)


def save_pca_downscaled(features, save_dir, pca, time_idx, num_frames):
    features = features.permute(1, 2, 0).detach().cpu().numpy()
    shape = features.shape
    if shape[2] != 3:
        if pca is None:
            pca = PCA(n_components=3)
            pca.fit(features.reshape(-1, shape[2]))
        features = pca.transform(
            features.reshape(-1, shape[2]))
        features = features.reshape(
            (shape[0], shape[1], 3))
    vmax, vmin = features.max(), features.min()
    normalized_features = np.clip((features - vmax) / (vmax - vmin + 1e-10), 0, 1)
    normalized_features_colormap = (normalized_features * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(save_dir, "gs_{:04d}.png".format(time_idx)), normalized_features_colormap)
    if time_idx == num_frames - 1: 
        make_vid(save_dir)

def numpy_and_save(save_path, input_list):
    input_list = np.array(input_list)
    np.savetxt(save_path, input_list)
    return input_list


def get_w2c(y_angle=0., center_dist=2.4, cam_height=1.3):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    return w2c

def get_circle(num_frames, device, avg_w2c, rots=1, rads=0.5, zrate=0):
    w2cs = torch.stack([avg_w2c] * num_frames)
    thetas = torch.linspace(
        0, 2 * torch.pi * rots, num_frames + 1, device=device)[:-1]
    positions = (
        torch.stack(
            [
                torch.cos(thetas),
                -torch.sin(thetas),
                -torch.sin(thetas * zrate),
            ],
            dim=-1,
        )
        * rads
    )
    positions += rads*0.5
    w2cs[:, :3, -1] += positions
    return w2cs
    

def get_cam_poses(novel_view_mode, dataset, config, num_frames, device, params):
    # w2c = torch.linalg.inv(pose)
    train_c2ws = dataset.transformed_poses.to(device)
    if novel_view_mode == 'circle':
        # params['means3D'] = (N, T, 3)
        avg_w2c = torch.linalg.inv(train_c2ws[0])

        # zoom out a bit
        scene_center = params['means3D'][:, :, :].reshape(-1, 3).mean(dim=0)
        lookat = scene_center - avg_w2c[:3, -1]
        if avg_w2c.sum() == 4:
            if 'DAVIS' in config['data']['basedir']:
                lookat = torch.tensor([0, 0, -10]).to(device)
            else:
                lookat = torch.tensor([0, 0, -2]).to(device)

        avg_w2c[:3, -1] -= 1 * lookat
        w2cs = get_circle(num_frames, device, avg_w2c, rads=0.3, rots=3)
        poses = torch.linalg.inv(w2cs)
        name = 'circle'
        
    elif novel_view_mode == 'test_cam':
        meta_path = os.path.join(
            config['data']['basedir'],
            os.path.dirname(os.path.dirname(config['data']['sequence'])),
            'test_meta.json')
        with open(meta_path, 'r') as jf:
            data = json.load(jf)

        # [0, 10, 15, 30] --> 10 (above) or 0 (side) most interesting I'd say
        test_cam = 10 # random.choice(data['cam_id'][0])
        print(f"Using test cam {test_cam}.")
        test_cam_idx = data['cam_id'][0].index(test_cam)
        test_cam_w2c = torch.tensor(data['w2c'][0][test_cam_idx]).float()

        # if point cloud was not transformed to world but is in train cam
        # if this is the case, train cam is actually world
        if not config['data']['do_transform']:
            print('Transforming!')
            meta_path = os.path.join(
                config['data']['basedir'],
                os.path.dirname(os.path.dirname(config['data']['sequence'])),
                'train_meta.json')
            with open(meta_path, 'r') as jf:
                data = json.load(jf)
            cam_id = int(os.path.basename(config['data']['sequence']))
            idx = data['cam_id'][0].index(cam_id)
            w2c = torch.tensor(data['w2c'][0][idx]).float()
            c2w = torch.linalg.inv(w2c)

            # right multiply
            test_cam_w2c = test_cam_w2c @ c2w

        test_cam_c2w = torch.linalg.inv(test_cam_w2c)
        poses = [test_cam_c2w.to(device)] * num_frames
        name = f'{test_cam}'
     
    return poses, name


def eval(
        dataset,
        final_params,
        num_frames,
        eval_dir,
        sil_thres, 
        wandb_run=None,
        save_frames=True,
        variables=None,
        save_pc=False,
        vis_gt=False,
        rendered_silhouette=False,
        rendered_instseg=False,
        get_embeddings=False,
        rendered_bg=False,
        time_window=1,
        save_depth=False,
        save_rendered_embeddings=False,
        remove_close=False,
        novel_view_mode=None,
        config=None):

    print("Evaluating Final Parameters ...")
    psnr_list, rmse_list, l1_list, lpips_list, ssim_list = list(), list(), list(), list(), list()

    if novel_view_mode is not None:
        poses, name = get_cam_poses(novel_view_mode, dataset, config, num_frames, final_params['means3D'].device, final_params)
        print(f"Evaluating novel view in mode {novel_view_mode}!!")
    else:
        name = ''

    if save_frames:
        dir_names = make_dirs(eval_dir, rendered_instseg, rendered_silhouette, save_pc, rendered_bg, vis_gt, save_depth, save_rendered_embeddings, name=name)
    
    pca = None
    loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True)
    loss_fn_alex = loss_fn_alex.to(final_params['means3D'].device)

    visibilities = list()
    for time_idx in tqdm(range(num_frames)):
        final_params_time = copy.deepcopy(final_params)
         # Get RGB-D Data & Camera Parameters
        data = dataset[time_idx]
        color, depth, intrinsics, pose, instseg, embeddings, support_trajs, bg = data

        if novel_view_mode is not None:
            pose = poses[time_idx]

        # Process Camera Parameters
        intrinsics = intrinsics[:3, :3]
        w2c = torch.linalg.inv(pose)

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
            'instseg': instseg,
            'embeddings': embeddings,
            'support_trajs': support_trajs,
            'bg': bg}
        
        variables, im, _, rastered_depth, rastered_inst, _, _, _, _, time_mask, _, rastered_sil, rendered_embeddings, rastered_bg, visibility, _ = get_renderings(
            final_params_time,
            variables,
            time_idx,
            curr_data,
            {'sil_thres': sil_thres, 'use_sil_for_loss': False, 'use_flow': 'rendered', 'depth_cam': 'cam', 'embedding_cam': 'cam'},
            disable_grads=True,
            track_cam=False,
            get_seg=True,
            get_embeddings=get_embeddings,
            time_window=time_window,
            do_compute_visibility=True,
            remove_close=remove_close)

        visibilities.append(visibility)
        
        valid_depth_mask = (curr_data['depth'] > 0)
        
        if novel_view_mode is None:
            # Render RGB and Calculate PSNR
            weighted_im = im * valid_depth_mask
            weighted_gt_im = curr_data['im'] * valid_depth_mask
            psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
            try:
                ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                            data_range=1.0, size_average=True)
            except:
                ssim = torch.tensor(0)
            lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                        torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

            psnr_list.append(psnr.cpu().numpy())
            ssim_list.append(ssim.cpu().numpy())
            lpips_list.append(lpips_score)

            # Compute Depth RMSE
            masked_rastered_depth = rastered_depth * valid_depth_mask
            diff_depth_rmse = torch.sqrt((((masked_rastered_depth - curr_data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((masked_rastered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
            rmse_list.append(rmse.cpu().numpy())
            l1_list.append(depth_l1.cpu().numpy())

        if save_frames:
            # Save Rendered RGB and Depth
            save_rgb(im, dir_names['render_rgb_dir'], time_idx, num_frames)

            if save_depth:
                # depth
                vmin = min(curr_data['depth'].min().item(), rastered_depth.min().item())
                vmax = max(curr_data['depth'].max().item(), rastered_depth.max().item()) + 1e-10
                save_normalized(rastered_depth.detach(), dir_names['render_depth_dir'], time_idx, vmin, vmax, num_frames)

            # bg
            if rendered_bg:
                save_normalized(rastered_bg, dir_names['render_bg_dir'], time_idx, num_frames=num_frames)

            # embeddings
            if rendered_embeddings is not None and save_rendered_embeddings:
                save_pca_downscaled(rendered_embeddings, dir_names['render_emb_dir'], pca, time_idx, num_frames)
                save_pca_downscaled(curr_data['embeddings'], dir_names['render_emb_dir'], pca, time_idx, num_frames)

            if rendered_silhouette:
                # silouette
                rastered_sil = torch.clamp(rastered_sil.unsqueeze(0) , 0, 1)[0].detach().cpu().numpy()
                sil_colormap = (rastered_sil * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(dir_names['render_sil_dir'], "gs_{:04d}.png".format(time_idx)), sil_colormap)

            if rendered_instseg:
                # instseg
                save_normalized(rastered_inst.detach(), dir_names['render_instseg_dir'], time_idx, num_frames=num_frames)

            if vis_gt and novel_view_mode is None:
                # Save GT RGB and Depth
                save_rgb(curr_data['im'], dir_names['rgb_dir'], time_idx, num_frames)
                # depth
                vmin = min(curr_data['depth'].min().item(), rastered_depth.min().item())
                vmax = max(curr_data['depth'].max().item(), rastered_depth.max().item())
                save_normalized(curr_data['depth'], dir_names['depth_dir'], time_idx, vmin, vmax, num_frames)
                # instseg
                save_normalized(curr_data['instseg'], dir_names['instseg_dir'], time_idx, num_frames)                
                # bg 
                save_normalized(curr_data['bg'], dir_names['bg_dir'], time_idx, num_frames)

        if save_pc:
            save_pc(final_params_time, dir_names['pc_dir'], time_idx, time_mask)
    
    if novel_view_mode is None:
        # Compute Average Metrics
        psnr_list = numpy_and_save(os.path.join(eval_dir, "psnr.txt"), psnr_list)
        rmse_list = numpy_and_save(os.path.join(eval_dir, "rmse.txt"), rmse_list)
        l1_list = numpy_and_save(os.path.join(eval_dir, "l1.txt"), l1_list)
        ssim_list = numpy_and_save(os.path.join(eval_dir, "ssim.txt"), ssim_list)
        lpips_list = numpy_and_save(os.path.join(eval_dir, "lpips.txt"), lpips_list)


        print("Average PSNR: {:.2f}".format(psnr_list.mean()))
        print("Average Depth RMSE: {:.2f} cm".format(rmse_list.mean()*100))
        print("Average Depth L1: {:.2f} cm".format(l1_list.mean()*100))
        print("Average MS-SSIM: {:.3f}".format(ssim_list.mean()))
        print("Average LPIPS: {:.3f}".format(lpips_list.mean()))

        if wandb_run is not None:
            wandb_run.log({"Final Stats/Average PSNR": psnr_list.mean(), 
                        "Final Stats/Average Depth RMSE": rmse_list.mean(),
                        "Final Stats/Average Depth L1": l1_list.mean(),
                        "Final Stats/Average MS-SSIM": ssim_list.mean(), 
                        "Final Stats/Average LPIPS": lpips_list.mean(),
                        "Final Stats/step": 1})

    return visibilities

def make_dirs(eval_dir, rendered_instseg, rendered_silhouette, save_pc, rendered_bg, vis_gt, save_depth, save_rendered_embeddings, name=''):
    dir_names = dict()
    dir_names['render_rgb_dir'] = os.path.join(eval_dir, f"rendered_rgb_{name}")
    os.makedirs(dir_names['render_rgb_dir'], exist_ok=True)
    
    if save_depth:
        dir_names['render_depth_dir'] = os.path.join(eval_dir, f"rendered_depth_{name}")
        os.makedirs(dir_names['render_depth_dir'], exist_ok=True)

    if save_rendered_embeddings:
        dir_names['render_emb_dir'] = os.path.join(eval_dir, f"pca_emb_{name}")
        dir_names['render_emb_gt_dir'] = os.path.join(eval_dir, f"pca_emb_gt_{name}")
        os.makedirs(dir_names['render_emb_dir'], exist_ok=True)
        os.makedirs(dir_names['render_emb_gt_dir'], exist_ok=True)

    if rendered_instseg:
        dir_names['render_instseg_dir'] = os.path.join(eval_dir, f"rendered_instseg_{name}")
        os.makedirs(dir_names['render_instseg_dir'], exist_ok=True)

    if rendered_silhouette:
        dir_names['render_sil_dir'] = os.path.join(eval_dir, f"rendered_sil_{name}")
        os.makedirs(dir_names['render_sil_dir'], exist_ok=True)

    if save_pc:
        dir_names['pc_dir'] = os.path.join(eval_dir, "pc")
        os.makedirs(dir_names['pc_dir'], exist_ok=True)

    if rendered_bg:
        dir_names['render_bg_dir'] = os.path.join(eval_dir, f"rendered_bg_{name}")
        os.makedirs(dir_names['render_bg_dir'], exist_ok=True)
    
    if vis_gt:
        dir_names['rgb_dir'] = os.path.join(eval_dir, "rgb")
        dir_names['depth_dir'] = os.path.join(eval_dir, "depth")
        dir_names['instseg_dir'] = os.path.join(eval_dir, "instseg")
        dir_names['bg_dir'] = os.path.join(eval_dir, "bg")  
        os.makedirs(dir_names['rgb_dir'], exist_ok=True)
        os.makedirs(dir_names['depth_dir'], exist_ok=True)
        os.makedirs(dir_names['instseg_dir'], exist_ok=True)
        os.makedirs(dir_names['bg_dir'], exist_ok=True)

    return dir_names

def eval_during(
        curr_data,
        curr_params,
        time_idx,
        eval_dir,
        sil_thres,
        im,
        rastered_depth,
        rastered_sil,
        rastered_bg,
        rendered_embeddings,
        pca=None,
        wandb_run=None,
        wandb_save_qual=False, 
        save_frames=True,
        variables=None,
        save_pc=False,
        save_videos=False,
        mov_thresh=0.001,
        vis_gt=False,
        rendered_motion=False,
        rendered_mov=False,
        rendered_silhouette=False,
        rendered_instseg=False,
        get_embeddings=False,
        save_rendered_embeddings=False,
        rendered_bg=False,
        save_depth=False,
        time_window=1,
        batch=0):

    psnr_list = []
    rmse_list = []
    l1_list = []
    lpips_list = []
    ssim_list = []
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if save_frames:
        dir_names = make_dirs(eval_dir, rendered_mov, rendered_instseg, rendered_silhouette, save_pc, rendered_motion, rendered_bg, vis_gt, save_depth, save_rendered_embeddings)
    
    gt_w2c_list = []
    loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True)
    loss_fn_alex = loss_fn_alex.to(curr_params['means3D'].device)
        # Get RGB-D Data & Camera Parameters

    valid_depth_mask = (curr_data['depth'] > 0)
    rastered_depth_viz = rastered_depth.detach()
    rastered_sil_vis = rastered_sil.detach()
    rastered_depth = rastered_depth * valid_depth_mask
    rastered_sil = rastered_sil.unsqueeze(0)

    # Render RGB and Calculate PSNR
    weighted_im = im * valid_depth_mask
    weighted_gt_im = curr_data['im'] * valid_depth_mask
    psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
    try:
        ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                    data_range=1.0, size_average=True)
    except:
        ssim = torch.tensor(0)
    lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

    diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth'])) ** 2))
    diff_depth_rmse = diff_depth_rmse * valid_depth_mask
    rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
    diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']))
    diff_depth_l1 = diff_depth_l1 * valid_depth_mask
    depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

    if save_frames:
        # Save Rendered RGB and Depth
        viz_render_im = torch.clamp(im, 0, 1)
        viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
        cv2.imwrite(os.path.join(render_rgb_dir, "gs_{:04d}_{:04d}.png".format(batch, time_idx)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
        if time_idx == num_frames - 1: 
            make_vid(render_rgb_dir)

        if save_depth:
            # depth
            viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
            vmin = 0
            vmax = viz_render_depth.max()
            normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_depth_dir, "gs_{:04d}_{:04d}.png".format(batch, time_idx)), depth_colormap)
            if time_idx == num_frames - 1: 
                make_vid(render_depth_dir)
        # bg
        if rendered_bg:
            rastered_bg = rastered_bg[0].detach().cpu().numpy()
            smax, smin = rastered_bg.max(), rastered_bg.min()
            normalized_bg = np.clip((rastered_bg - smin) / (smax - smin), 0, 1)
            bg_colormap = cv2.applyColorMap((normalized_bg * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_bg_dir, "gs_{:04d}_{:04d}.png".format(batch, time_idx)), bg_colormap)
            if time_idx == num_frames - 1: 
                make_vid(render_bg_dir)

        # embeddings
        if rendered_embeddings is not None and save_rendered_embeddings:
            rendered_embeddings = rendered_embeddings.permute(1, 2, 0).detach().cpu().numpy()
            shape = rendered_embeddings.shape
            if shape[2] != 3:
                if pca is None:
                    pca = PCA(n_components=3)
                    pca.fit(rendered_embeddings.reshape(-1, shape[2]))
                rendered_embeddings = pca.transform(
                    rendered_embeddings.reshape(-1, shape[2]))
                rendered_embeddings = rendered_embeddings.reshape(
                    (shape[0], shape[1], 3))
            smax, smin = rendered_embeddings.max(), rendered_embeddings.min()
            normalized_emb = np.clip((rendered_embeddings - smin) / (smax - smin), 0, 1)
            emb_colormap = (normalized_emb * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(render_emb_dir, "gs_{:04d}_{:04d}.png".format(batch, time_idx)), emb_colormap)
            if time_idx == num_frames - 1: 
                make_vid(render_emb_dir)

            rendered_embeddings = curr_data['embeddings'].permute(1, 2, 0).detach().cpu().numpy()
            if shape[2] != 3:
                shape = rendered_embeddings.shape
                rendered_embeddings = pca.transform(
                    rendered_embeddings.reshape(-1, shape[2]))
                rendered_embeddings = rendered_embeddings.reshape(
                    (shape[0], shape[1], 3))
            smax, smin = rendered_embeddings.max(), rendered_embeddings.min()
            normalized_emb = np.clip((rendered_embeddings - smin) / (smax - smin), 0, 1)
            emb_colormap = (normalized_emb * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(render_emb_gt_dir, "gs_{:04d}_{:04d}.png".format(batch, time_idx)), emb_colormap)
            if time_idx == num_frames - 1: 
                make_vid(render_emb_gt_dir)

        if rendered_silhouette:
            # silouette
            rastered_sil_vis = torch.clamp(rastered_sil_vis , 0, 1)[0].detach().cpu().numpy()
            sil_colormap = (rastered_sil_vis * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(render_sil_dir, "gs_{:04d}_{:04d}.png".format(batch, time_idx)), sil_colormap)
            if time_idx == num_frames - 1: 
                make_vid(render_sil_dir)

        if rendered_instseg:
            # instseg
            viz_render_instseg = rastered_inst_viz[0].detach().cpu().numpy()
            smax, smin = viz_render_instseg.max(), viz_render_instseg.min()
            normalized_instseg = np.clip((viz_render_instseg - smin) / (smax - smin), 0, 1)
            instseg_colormap = cv2.applyColorMap((normalized_instseg * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_instseg_dir, "gs_{:04d}_{:04d}.png".format(batch, time_idx)), instseg_colormap)
            if time_idx == num_frames - 1: 
                make_vid(render_instseg_dir)

        if vis_gt:
            # Save GT RGB and Depth
            viz_gt_im = torch.clamp(curr_data['im'], 0, 1)
            viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
            # depth
            vmin = 0 # viz_gt_depth.min() # 0
            vmax = 6 # viz_gt_depth.max() # 6
            viz_gt_depth = torch.clamp(curr_data['depth'], 0, vmax)
            viz_gt_depth = viz_gt_depth[0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_gt_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # instseg
            viz_gt_instseg = curr_data['instseg'][0].detach().cpu().numpy()
            smax, smin = viz_gt_instseg.max(), viz_gt_instseg.min()
            normalized_instseg = np.clip((viz_gt_instseg - smin) / (smax - smin), 0, 1)
            instseg_colormap = cv2.applyColorMap((normalized_instseg * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # bg 
            viz_gt_bg = curr_data['bg'][0].detach().cpu().float().numpy()
            smax, smin = viz_gt_bg.max(), viz_gt_bg.min()
            normalized_bg = np.clip((viz_gt_bg - smin) / (smax - smin), 0, 1)
            bg_colormap = cv2.applyColorMap((normalized_bg * 255).astype(np.uint8), cv2.COLORMAP_JET)

            cv2.imwrite(os.path.join(rgb_dir, "gt_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))
            if time_idx == num_frames - 1: 
                make_vid(rgb_dir)
            cv2.imwrite(os.path.join(depth_dir, "gt_{:04d}.png".format(time_idx)), depth_colormap)
            if time_idx == num_frames - 1: 
                make_vid(depth_dir)
            cv2.imwrite(os.path.join(instseg_dir, "gt_{:04d}.png".format(time_idx)), instseg_colormap)
            if time_idx == num_frames - 1: 
                make_vid(instseg_dir)
            cv2.imwrite(os.path.join(bg_dir, "gt_{:04d}.png".format(time_idx)), bg_colormap)
            if time_idx == num_frames - 1: 
                make_vid(bg_dir)

    if save_pc:
        print('all', curr_params['means3D'][:, :, time_idx][time_mask].shape)
        pcd = o3d.geometry.PointCloud()
        v3d = o3d.utility.Vector3dVector
        pcd.points = v3d(curr_params['means3D'][:, :, time_idx][time_mask].cpu().numpy())
        o3d.io.write_point_cloud(filename=os.path.join(pc_dir, "pc_{:04d}_all.xyz".format(time_idx)), pointcloud=pcd)
    
    return psnr, rmse, depth_l1, ssim, lpips_score, pca

