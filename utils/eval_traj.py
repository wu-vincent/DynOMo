import os
from utils.get_data import get_gt_traj, load_scene_data, get_cam_data
import numpy as np
import torch
from utils.camera_helpers import get_projection_matrix
from utils.two2threeD_helpers import three2two, unnormalize_points, normalize_points
from utils.tapnet_utils_viz import vis_tracked_points
from utils.camera_helpers import setup_camera
from utils.slam_helpers import transform_to_frame
import copy


def get_gs_traj_pts(proj_matrix, params, first_occurance, w, h, start_pixels, start_pixels_normalized=True, gauss_ids=None, norm=False):
    means3D = params['means3D'][first_occurance==first_occurance.min().item()]

    # assign and get 3D trajectories
    means3D_t0 = means3D[:, :, 0]
    if not norm:
        means2D_t0 = three2two(proj_matrix, means3D_t0, w, h, do_round=True)
        if start_pixels_normalized:
            start_pixels = unnormalize_points(start_pixels, h, w, do_round=True)
            
    else:
        means2D_t0 = three2two(proj_matrix, means3D_t0, w, h, do_normalize=True).float()
        if not start_pixels_normalized:
            start_pixels = normalize_points(start_pixels.float(), h, w)
        start_pixels = start_pixels.to(means2D_t0.device).float()

    if gauss_ids is None:
        gauss_ids = find_closest_to_start_pixels(
            means2D_t0,
            start_pixels,
            norm=norm)

    gs_traj_3D, logit_opacities, logit_sclaes, rgb_colors, unnorm_rotations = get_3D_trajs_for_track(
        gauss_ids, params, return_all=True)
    print(logit_opacities.squeeze())
    
    params_gs_traj_3D = copy.deepcopy(params)
    params_gs_traj_3D['means3D'] = gs_traj_3D
    params_gs_traj_3D['unnorm_rotations'] = unnorm_rotations
    # get 2D trajectories
    gs_traj_2D = list()
    for time in range(gs_traj_3D.shape[-1]):
        if gs_traj_3D[:, :, time].sum() == 0:
            continue
        transformed_gs_traj_3D = transform_to_frame(
                params_gs_traj_3D,
                time,
                gaussians_grad=False,
                camera_grad=False,
                delta=0)
        gs_traj_2D.append(
            three2two(proj_matrix, transformed_gs_traj_3D['means3D'], w, h, do_normalize=False))
    gs_traj_2D = torch.stack(gs_traj_2D).permute(1, 0, 2)
    gs_traj_3D = gs_traj_3D.permute(0, 2, 1)

    return gs_traj_2D, gs_traj_3D, gauss_ids


def find_closest(means2D, pix):
    for d_x in [0, -1, 0, 1, 0, -1, 1, -1, 1, -2, 2, 0, 0, -2, 2, -2, 2, -1, 1, 1, -1. -2, 2, -2, 2]:
        for d_y in [0, 0, -1, 0, 1, -1, -1, 1, 1, 0, 0, -2, 2, -1, 1, 1, -1, -2, 2, -2, 2, -2, -2, 2, 2]:
            pix_mask =  torch.logical_and(
                means2D[:, 0] == pix[0] + d_x,
                means2D[:, 1] == pix[1] + d_y)
            if pix_mask.sum() != 0:
                return torch.nonzero(pix_mask)[0]
    return None


def find_closest_norm(means2D, pix):
    dist = torch.cdist(pix.unsqueeze(0).unsqueeze(0), means2D.unsqueeze(0)).squeeze()
    gauss_id = dist.argmin()
    return gauss_id


def find_closest_to_start_pixels(means2D, start_pixels, norm=False):
    gauss_ids = list()
    gs_traj_3D = list()
    for i, pix in enumerate(start_pixels):
        if norm:
            gauss_id = find_closest_norm(means2D, pix).unsqueeze(0)
        else:
            gauss_id = find_closest(means2D, pix)

        if gauss_id is not None:
            gauss_ids.append(gauss_id)
        else:
            gauss_ids.append(torch.tensor([0]).to(means2D.device))

    return gauss_ids
        

def get_3D_trajs_for_track(gauss_ids, params, return_all=False):
    gs_traj_3D = list()
    logit_opacities = list()
    logit_scales = list()
    rgb_colors = list()
    unnorm_rotations = list()

    for gauss_id in gauss_ids:
        if gauss_id != -1 and params['bg'][gauss_id] < 0.5:
            gs_traj_3D.append(
                    params['means3D'][gauss_id].squeeze())
            unnorm_rotations.append(params['unnorm_rotations'][gauss_id].squeeze())
            logit_opacities.append(params['logit_opacities'][gauss_id].squeeze())
            rgb_colors.append(params['rgb_colors'][gauss_id])
            logit_scales.append(params['log_scales'][gauss_id])
        else:
            gs_traj_3D.append(
                    (torch.ones_like(params['means3D'][0]).squeeze()*-1).to(params['means3D'].device))
            logit_opacities.append(torch.tensor(-1).to(params['means3D'].device))
            rgb_colors.append(torch.tensor([[-1, -1, -1]]).to(params['means3D'].device))
            logit_scales.append(torch.tensor([[-1]]).to(params['means3D'].device))
            unnorm_rotations.append(
                (torch.ones_like(params['unnorm_rotations'][0]).squeeze()*-1).to(params['means3D'].device))

    if return_all:
        return torch.stack(gs_traj_3D), torch.stack(logit_opacities), torch.stack(logit_scales), torch.stack(rgb_colors), torch.stack(unnorm_rotations)
    else:
        return torch.stack(gs_traj_3D)

def _eval_traj(
        params,
        first_occurance,
        data,
        h=None,
        w=None,
        proj_matrix=None,
        default_to_proj=True,
        use_only_pred=True,
        vis_trajs=False,
        results_dir=None,
        gauss_ids_to_track=None):

    if params['means3D'][:, :, -1].sum() == 0:
        params['means3D'] = params['means3D'][:, :, :-1]
        params['unnorm_rotations'] = params['unnorm_rotations'][:, :, :-1]

    # use projected 3D trajectories if no ground truth
    if data['points'].sum() == 0 and default_to_proj:
        gt_traj_2D = data['points_projected']
        occluded = data['occluded'] - 1 
    else:
        gt_traj_2D = data['points']
        occluded = data['occluded']
    valids = 1-occluded.float()
    
    # get trajectories of Gaussians
    gs_traj_2D, gs_traj_3D, gauss_ids = get_gs_traj_pts(
        proj_matrix,
        params,
        first_occurance,
        w,
        h,
        gt_traj_2D[:, 0].clone(),
        gauss_ids=gauss_ids_to_track)

    # unnormalize gt to image pixels
    gt_traj_2D = unnormalize_points(gt_traj_2D, h, w)

    # make timesteps after predicted len invalid
    if use_only_pred:
        gt_traj_2D = gt_traj_2D[:, :gs_traj_2D.shape[1], :]
        valids = valids[:, :gs_traj_2D.shape[1]]
        occluded = occluded[:, :gs_traj_2D.shape[1]]

    # unsqueeze to batch dimension
    if len(gt_traj_2D.shape) == 3:
        gt_traj_2D = gt_traj_2D.unsqueeze(0)
        gs_traj_2D = gs_traj_2D.unsqueeze(0)
        valids = valids.unsqueeze(0)
        occluded = occluded.unsqueeze(0)

    # mask by valid ids
    gs_traj_2D, gt_traj_2D, valids, occluded = mask_valid_ids(
        valids,
        gs_traj_2D,
        gt_traj_2D,
        occluded)

    # compute metrics
    metrics = compute_metrics(
        h,
        w,
        gs_traj_2D.to(gt_traj_2D.device),
        gt_traj_2D,
        valids
    )
    if vis_trajs:
        data['points'] = normalize_points(gs_traj_2D, h, w).squeeze()
        data['occluded'] = occluded.squeeze()
        data = {k: v.detach().clone().cpu().numpy() for k, v in data.items()}
        vis_tracked_points(
            results_dir,
            data)

    return metrics


def mask_valid_ids(valids, gs_traj_2D, gt_traj_2D, occluded):
    # only keep points that are visible at time 0
    vis_ok = valids[:, :, 0] > 0

    # flatten along along batch * num points and mask
    shape = gs_traj_2D.shape
    vis_ok = vis_ok.reshape(shape[0]*shape[1])
    gs_traj_2D = gs_traj_2D.reshape(
        shape[0]*shape[1], shape[2], shape[3])[vis_ok].reshape(
            shape[0], -1, shape[2], shape[3])
    gt_traj_2D = gt_traj_2D.reshape(
        shape[0]*shape[1], shape[2], shape[3])[vis_ok].reshape(
            shape[0], -1, shape[2], shape[3])
    valids = valids.reshape(
        shape[0]*shape[1], shape[2])[vis_ok].reshape(
            shape[0], -1, shape[2])
    occluded = occluded.reshape(
        shape[0]*shape[1], shape[2])[vis_ok].reshape(
            shape[0], -1, shape[2])
    
    return gs_traj_2D, gt_traj_2D, valids, occluded


def compute_metrics(
        H,
        W,
        gs_traj_2D,
        gt_traj_2D,
        valids,
        sur_thr=16,
        thrs=[1, 2, 4, 8, 16],
        norm_factor=256):
    B, N, S = gt_traj_2D.shape[0], gt_traj_2D.shape[1], gt_traj_2D.shape[2]
    
    # permute number of points and seq len
    gs_traj_2D = gs_traj_2D.permute(0, 2, 1, 3)
    gt_traj_2D = gt_traj_2D.permute(0, 2, 1, 3)
    valids = valids.permute(0, 2, 1)

    # get metrics
    metrics = dict()
    d_sum = 0.0
    sc_pt = torch.tensor(
        [[[W/norm_factor, H/norm_factor]]]).float().to(gs_traj_2D.device)
    for thr in thrs:
        # note we exclude timestep0 from this eval
        d_ = (torch.norm(
            gs_traj_2D[:,1:]/sc_pt - gt_traj_2D[:,1:]/sc_pt, dim=-1) < thr).float() # B,S-1,N
        d_ = reduce_masked_mean(d_, valids[:,1:]).item()*100.0
        d_sum += d_
        metrics['d_%d' % thr] = d_
    d_avg = d_sum / len(thrs)
    metrics['d_avg'] = d_avg
    
    dists = torch.norm(gs_traj_2D/sc_pt - gt_traj_2D/sc_pt, dim=-1) # B,S,N
    dist_ok = 1 - (dists > sur_thr).float() * valids # B,S,N
    survival = torch.cumprod(dist_ok, dim=1) # B,S,N
    metrics['survival'] = torch.mean(survival).item()*100.0

    # get the median l2 error for each trajectory
    dists_ = dists.permute(0,2,1).reshape(B*N,S)
    valids_ = valids.permute(0,2,1).reshape(B*N,S)
    median_l2 = reduce_masked_median(dists_, valids_, keep_batch=True)
    metrics['median_l2'] = median_l2.mean().item()

    return metrics


def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    for (a,b) in zip(x.size(), mask.size()):
        # if not b==1: 
        assert(a==b) # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x*mask

    if dim is None:
        numer = torch.sum(prod)
        denom = 1e-10+torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = 1e-10+torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer/denom
    return mean


def reduce_masked_median(x, mask, keep_batch=False):
    # x and mask are the same shape
    assert(x.size() == mask.size())
    device = x.device

    B = list(x.shape)[0]
    x = x.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    if keep_batch:
        x = np.reshape(x, [B, -1])
        mask = np.reshape(mask, [B, -1])
        meds = np.zeros([B], np.float32)
        for b in list(range(B)):
            xb = x[b]
            mb = mask[b]
            if np.sum(mb) > 0:
                xb = xb[mb > 0]
                meds[b] = np.median(xb)
            else:
                meds[b] = np.nan
        meds = torch.from_numpy(meds).to(device)
        return meds.float()
    else:
        x = np.reshape(x, [-1])
        mask = np.reshape(mask, [-1])
        if np.sum(mask) > 0:
            x = x[mask > 0]
            med = np.median(x)
        else:
            med = np.nan
        med = np.array([med], np.float32)
        med = torch.from_numpy(med).to(device)
        return med.float()


def eval_traj(
        config,
        params=None,
        results_dir='out',
        cam=None,
        vis_trajs=True,
        gauss_ids_to_track=None):
    # get projectoin matrix
    if cam is None:
        params, _, k, w2c = load_scene_data(config, results_dir)
        h, w = config["data"]["desired_image_height"], config["data"]["desired_image_width"]
        proj_matrix = get_projection_matrix(w, h, k, w2c).squeeze()
        results_dir = os.path.join(results_dir, 'eval')
        '''if 'gauss_ids_to_track' in params.keys():
            gauss_ids_to_track = params['gauss_ids_to_track'].long()
        if gauss_ids_to_track.sum() == 0:
            gauss_ids_to_track = None'''
        print(params['bg'])
    else:
        proj_matrix = cam.projmatrix.squeeze()
        h = cam.image_height
        w = cam.image_width

    # get gt data
    data = get_gt_traj(config, in_torch=True)
    if "jono_data" in config["data"]["gradslam_data_cfg"]:
        data['points'][:, 0] = (data['points'][:, 0] * w - 1)/w
        data['points'][:, 1] = (data['points'][:, 1] * h - 1)/h

    # get metrics
    metrics = _eval_traj(
        params,
        params['timestep'],
        data,
        proj_matrix=proj_matrix,
        h=h,
        w=w,
        results_dir=os.path.join(results_dir, 'tracked_points_vis'),
        vis_trajs=vis_trajs,
        gauss_ids_to_track=gauss_ids_to_track)

    return metrics


def meshgrid2d(B, Y, X, stack=False, norm=False, device='cuda', on_chans=False):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2d(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        if on_chans:
            grid = torch.stack([grid_x, grid_y], dim=1)
        else:
            grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


def get_xy_grid(H, W, N=1024, B=1):
    # pick N points to track; we'll use a uniform grid
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
    xy0 = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2

    return xy0

def vis_grid_trajs(config, params=None, cam=None, results_dir=None, orig_image_size=False):
    # get projectoin matrix
    if cam is None:
        params, _, k, w2c = load_scene_data(config, results_dir)
        if orig_image_size:
            k, pose, h, w = get_cam_data(config, orig_image_size=orig_image_size)
            w2c = torch.linalg.inv(pose)
            proj_matrix = get_projection_matrix(w, h, k, w2c).squeeze()
        else:
            h, w = config["data"]["desired_image_height"], config["data"]["desired_image_width"]
            proj_matrix = get_projection_matrix(w, h, k, w2c).squeeze()
        results_dir = os.path.join(results_dir, 'eval')
    elif orig_image_size:
        k, pose, h, w = get_cam_data(config, orig_image_size=orig_image_size)
        w2c = torch.linalg.inv(pose)
        proj_matrix = get_projection_matrix(w, h, k, w2c).squeeze()
    else:
        proj_matrix = cam.projmatrix.squeeze()
        h = cam.image_height
        w = cam.image_width

    # get trajectories to track
    start_pixels = get_xy_grid(h, w).squeeze().long()
    gs_traj_2D, gs_traj_3D, gauss_ids = get_gs_traj_pts(
        proj_matrix,
        params,
        params['timestep'],
        w,
        h,
        start_pixels,
        start_pixels_normalized=False)

    # get gt data for visualization (actually only need rgb here)
    data = get_gt_traj(config, in_torch=True)

    data['points'] = normalize_points(gs_traj_2D, h, w).squeeze()
    data['occluded'] = torch.zeros(data['points'].shape[:-1]).to(data['points'].device)
    data = {k: v.detach().clone().cpu().numpy() for k, v in data.items()}
    vis_tracked_points(
        os.path.join(results_dir, 'grid_points_vis'),
        data)

