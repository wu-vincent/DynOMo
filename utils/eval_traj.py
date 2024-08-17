import os
from utils.get_data import get_gt_traj, load_scene_data, get_cam_data
import numpy as np
import torch
from utils.camera_helpers import get_projection_matrix
from utils.two2threeD_helpers import three2two, unnormalize_points, normalize_points
from utils.tapnet_utils_viz import vis_tracked_points, vis_trail
from utils.camera_helpers import setup_camera
from utils.slam_helpers import transform_to_frame, get_renderings
import copy
import glob
import json 
import imageio
import cv2
import pdb


def format_start_pix(start_pixels, use_norm_pix, start_pixels_normalized, use_round_pix, h, w, do_scale, params):
    if not use_norm_pix:
        if use_round_pix:
            if start_pixels_normalized:
                start_pixels = unnormalize_points(start_pixels, h, w, do_round=True)
        else:
            if start_pixels_normalized:
                start_pixels = unnormalize_points(start_pixels, h, w, do_round=False)
            start_pixels = start_pixels.to(params['means3D'].device).float()
    else:
        if not start_pixels_normalized:
            start_pixels = normalize_points(start_pixels.float(), h, w)
        start_pixels = start_pixels.to(params['means3D'].device).float()
    
    return start_pixels


def gauss_wise3D_track(search_fg_only, params, use_norm_pix, use_round_pix, proj_matrix, w, h, do_scale, start_pixels_normalized, thresh, best_x, gt_colors, color_thresh, no_bg, gauss_ids, first_occurance, start_pixels):
    if gauss_ids is None or gauss_ids == np.array(None):
        if search_fg_only:
            fg_mask = (params['bg'] < 0.5).squeeze()
            num_gauss = params['means3D'].shape[0]
            first_occurance = first_occurance[fg_mask]
            for k in params.keys():
                try:
                    if params[k].shape[0] == num_gauss:
                        params[k] = params[k][fg_mask]
                except:
                    params[k] = params[k]

        means3D = params['means3D'][first_occurance==first_occurance.min().item()]
        # assign and get 3D trajectories
        means3D_t0 = means3D[:, :, 0]

        if not use_norm_pix:
            if use_round_pix:
                means2D_t0 = three2two(proj_matrix, means3D_t0, w, h, do_round=True)
            else:
                means2D_t0 = three2two(proj_matrix, means3D_t0, w, h, do_round=False).float()
        else:
            means2D_t0 = three2two(proj_matrix, means3D_t0, w, h, do_normalize=True).float()
            
        gauss_ids = find_closest_to_start_pixels(
            means2D_t0,
            start_pixels,
            means3D=means3D_t0,
            opacity=params['logit_opacities'],
            not_round=use_norm_pix or not use_round_pix,
            visibility=params['visibility'] if "visibility" in params.keys() else torch.ones_like(params['logit_opacities']),
            thresh=thresh,
            best_x=best_x,
            gt_colors=gt_colors,
            gs_colors=params['rgb_colors'][:, :, 0] if len(params['rgb_colors'].shape) == 3 else params['rgb_colors'],
            color_thresh=color_thresh)

    gs_traj_3D, logit_opacities, rgb_colors, unnorm_rotations, visibility = get_3D_trajs_for_track(
        gauss_ids, params, return_all=True, no_bg=no_bg)
    return gs_traj_3D, logit_opacities, rgb_colors, unnorm_rotations, visibility


def get_2D_track_from_3D(params, gs_traj_3D, unnorm_rotations, proj_matrix, w, h):
    params_gs_traj_3D = copy.deepcopy(params)
    params_gs_traj_3D['means3D'] = gs_traj_3D

    params_gs_traj_3D['unnorm_rotations'] = unnorm_rotations
    gs_traj_2D = list()
    for time in range(gs_traj_3D.shape[-1]):
        if gs_traj_3D[:, :, time].sum() == 0:
            continue
        transformed_gs_traj_3D, _ = transform_to_frame(
                params_gs_traj_3D,
                time,
                gaussians_grad=False,
                camera_grad=False,
                delta=0)
        gs_traj_2D.append(
            three2two(proj_matrix, transformed_gs_traj_3D['means3D'], w, h, do_normalize=False))
    gs_traj_2D = torch.stack(gs_traj_2D).permute(1, 0, 2)
    gs_traj_3D = gs_traj_3D.permute(0, 2, 1)

    return gs_traj_2D


def get_2D_track_from_3D_for_vis(params, gs_traj_3D, unnorm_rotations, proj_matrices, w, h):
    params_gs_traj_3D = copy.deepcopy(params)
    params_gs_traj_3D['means3D'] = gs_traj_3D

    params_gs_traj_3D['unnorm_rotations'] = unnorm_rotations
    gs_traj_2D_per_time = list()
    for cam_time in range(gs_traj_3D.shape[-1]):
        if cam_time % 50 == 0:
            print(f"Processing step {cam_time}...")
        gs_traj_2D = list()
        proj_matrix = proj_matrices[cam_time] if len(proj_matrices.shape) == 3 else proj_matrices
        for gauss_time in range(gs_traj_3D.shape[-1]):
            if gs_traj_3D[:, :, gauss_time].sum() == 0:
                continue
            transformed_gs_traj_3D, _ = transform_to_frame(
                    params_gs_traj_3D,
                    cam_time,
                    gaussians_grad=False,
                    camera_grad=False,
                    gauss_time_idx=gauss_time,
                    delta=0)
            gs_traj_2D.append(
                three2two(proj_matrix, transformed_gs_traj_3D['means3D'], w, h, do_normalize=False))
        gs_traj_2D = torch.stack(gs_traj_2D).permute(1, 0, 2)
        gs_traj_2D_per_time.append(gs_traj_2D)
    gs_traj_2D_per_time = torch.stack(gs_traj_2D_per_time)
    return gs_traj_2D_per_time


def get_2D_and_3D_from_sum(params, cam, start_pixels, proj_matrix, w, h, visuals):
    with torch.no_grad():
        _, im, _, _, _, _, _, visible, weight, time_mask, _, _, _, _, _, _ = get_renderings(
            params,
            params,
            0,
            data={'cam': cam},
            config=None, 
            disable_grads=True,
            track_cam=False,
            get_depth=False,
            get_embeddings=False)
    visible_means_start_pix = visible[:, torch.round(start_pixels).long()[:, 1],torch.round(start_pixels).long()[:, 0]]
    weight_means_start_pix = weight[:, torch.round(start_pixels).long()[:, 1],torch.round(start_pixels).long()[:, 0]]
    all_trajs_3D = list()
    all_trajs_2D = list()
    all_visibilities = list()
    gs_traj_2D_per_time = list()
    for i in range(start_pixels.shape[0]):
        visible_means = visible_means_start_pix[:, i][torch.nonzero(visible_means_start_pix[:, i])].squeeze().long()
        weight_means = weight_means_start_pix[:, i][torch.nonzero(visible_means_start_pix[:, i])].squeeze()
        traj_3D = list()
        traj_2D = list()
        visibility = list()
        gs_traj_2D_per_time_per_start_pix = list()
        start_pix_params = copy.deepcopy(params)
        for cam_time in range(params['means3D'].shape[2]):
            gs_traj_2D_per_time_per_start_pix_per_cam_time = list()
            for gauss_time in range(params['means3D'].shape[2]):
                if not visuals and cam_time != gauss_time:
                    continue
                loc_3D = ((weight_means/weight_means.sum()).unsqueeze(1) * params['means3D'][visible_means, :, gauss_time]).sum(dim=0).unsqueeze(0).unsqueeze(-1)
                start_pix_params['means3D'] = loc_3D
                start_pix_params['unnorm_rotations'] = torch.zeros(1, 4, 1).to(params['means3D'].device)
                transformed_loc_3D, _ = transform_to_frame(
                        start_pix_params,
                        cam_time,
                        gaussians_grad=False,
                        camera_grad=False,
                        gauss_time_idx=0,
                        delta=0)
                loc_2D = three2two(proj_matrix, transformed_loc_3D['means3D'], w, h, do_normalize=False).float()
                if cam_time == gauss_time:
                    visibility.append(((weight_means/weight_means.sum()) * params['visibility'][visible_means, gauss_time].squeeze()).sum())
                    traj_3D.append(loc_3D)
                    traj_2D.append(loc_2D)
                gs_traj_2D_per_time_per_start_pix_per_cam_time.append(loc_2D)

            gs_traj_2D_per_time_per_start_pix_per_cam_time = torch.stack(gs_traj_2D_per_time_per_start_pix_per_cam_time)
            gs_traj_2D_per_time_per_start_pix.append(gs_traj_2D_per_time_per_start_pix_per_cam_time)

        traj_2D = torch.stack(traj_2D).squeeze()
        traj_3D = torch.stack(traj_3D).squeeze()
        visibility = torch.stack(visibility)
        gs_traj_2D_per_time_per_start_pix = torch.stack(gs_traj_2D_per_time_per_start_pix)

        all_trajs_2D.append(traj_2D)
        all_trajs_3D.append(traj_3D)
        all_visibilities.append(visibility)
        gs_traj_2D_per_time.append(gs_traj_2D_per_time_per_start_pix)

    all_trajs_2D = torch.stack(all_trajs_2D).squeeze()
    all_trajs_3D = torch.stack(all_trajs_3D).squeeze()
    all_visibilities = torch.stack(all_visibilities).squeeze()
    gs_traj_2D_per_time = torch.stack(gs_traj_2D_per_time)
    if visuals:
        gs_traj_2D_per_time = gs_traj_2D_per_time.squeeze().permute(1, 0, 2, 3)
    else:
        gs_traj_2D_per_time = None
    return all_trajs_2D, all_trajs_3D, all_visibilities, gs_traj_2D_per_time
        

def gauss_wise3D_track_from_3D(
        search_fg_only,
        params, 
        thresh,
        best_x,
        gt_colors,
        color_thresh,
        no_bg,
        gauss_ids,
        first_occurance,
        start_3D,
        start_pixels):
    if gauss_ids is None or gauss_ids == np.array(None):
        if search_fg_only:
            fg_mask = (params['bg'] < 0.5).squeeze()
            num_gauss = params['means3D'].shape[0]
            first_occurance = first_occurance[fg_mask]
            for k in params.keys():
                try:
                    if params[k].shape[0] == num_gauss:
                        params[k] = params[k][fg_mask]
                except:
                    params[k] = params[k]

        means3D = params['means3D'][first_occurance==first_occurance.min().item()]
        # assign and get 3D trajectories
        means3D_t0 = means3D[:, :, 0]
        gauss_ids = find_closest_to_start_pixels(
            means3D_t0,
            start_3D,
            means3D=means3D_t0,
            opacity=params['logit_opacities'],
            not_round=True,
            visibility=params['visibility'] if "visibility" in params.keys() else torch.ones_like(params['logit_opacities']),
            thresh=thresh,
            best_x=best_x,
            gt_colors=gt_colors,
            gs_colors=params['rgb_colors'][:, :, 0] if len(params['rgb_colors'].shape) == 3 else params['rgb_colors'],
            color_thresh=color_thresh,
            topk=min(5000, means3D_t0.shape[0]), 
            start_pixels_2=start_pixels)

    gs_traj_3D, logit_opacities, rgb_colors, unnorm_rotations, visibility = get_3D_trajs_for_track(
        gauss_ids, params, return_all=True, no_bg=no_bg)

    return gs_traj_3D, logit_opacities, rgb_colors, unnorm_rotations, visibility


def get_gs_traj_pts(
        proj_matrix,
        params,
        first_occurance,
        w,
        h,
        start_pixels,
        start_pixels_normalized=True,
        gauss_ids=None,
        use_norm_pix=False,
        use_round_pix=True,
        do_scale=True,
        no_bg=False,
        search_fg_only=False,
        w2c=None,
        thresh=10000,
        best_x=1,
        gt_colors=None,
        color_thresh=10000,
        cam=None,
        get_gauss_wise3D_track=True,
        get_from3D=False,
        start_3D=None,
        visuals=False,
        proj_matrices=None):
    
    if proj_matrices is None:
        proj_matrices = proj_matrix

    # get start pixels in right format
    start_pixels = format_start_pix(
        start_pixels,
        use_norm_pix=False if not get_gauss_wise3D_track else use_norm_pix,
        start_pixels_normalized=start_pixels_normalized,
        use_round_pix=False if not get_gauss_wise3D_track else use_round_pix,
        h=h,
        w=w,
        do_scale=do_scale,
        params=params)

    gs_traj_2D_for_vis = None
    if get_gauss_wise3D_track and get_from3D and start_3D is not None:
        gs_traj_3D, logit_opacities, rgb_colors, unnorm_rotations, visibility = \
            gauss_wise3D_track_from_3D(
                search_fg_only,
                params,
                thresh,
                best_x,
                gt_colors,
                color_thresh,
                no_bg,
                gauss_ids,
                first_occurance,
                start_3D,
                start_pixels)
        if visuals:
            gs_traj_2D_for_vis = get_2D_track_from_3D_for_vis(
                copy.deepcopy(params),
                copy.deepcopy(gs_traj_3D),
                copy.deepcopy(unnorm_rotations),
                proj_matrices,
                w,
                h,
            )
        gs_traj_2D = get_2D_track_from_3D(
            params, gs_traj_3D, unnorm_rotations, proj_matrix, w, h)
        gs_traj_3D = gs_traj_3D.permute(0, 2, 1)
    elif get_gauss_wise3D_track:
        gs_traj_3D, logit_opacities, rgb_colors, unnorm_rotations, visibility = \
            gauss_wise3D_track(
                search_fg_only,
                copy.deepcopy(params),
                use_norm_pix,
                use_round_pix,
                proj_matrix,
                w,
                h,
                do_scale,
                start_pixels_normalized,
                thresh,
                best_x,
                gt_colors,
                color_thresh,
                no_bg, gauss_ids,
                first_occurance,
                start_pixels)
        
        if visuals:
            gs_traj_2D_for_vis = get_2D_track_from_3D_for_vis(
                copy.deepcopy(params),
                copy.deepcopy(gs_traj_3D),
                copy.deepcopy(unnorm_rotations),
                proj_matrices,
                w,
                h,
            )

        gs_traj_2D = get_2D_track_from_3D(
            params, gs_traj_3D, unnorm_rotations, proj_matrix, w, h)
        
        gs_traj_3D = gs_traj_3D.permute(0, 2, 1)
    else:
        gs_traj_2D, gs_traj_3D, visibility, gs_traj_2D_for_vis = get_2D_and_3D_from_sum(
            params, cam, start_pixels, proj_matrix, w, h, visuals)
    return gs_traj_2D, gs_traj_3D, visibility, gs_traj_2D_for_vis


def find_closest(means2D, pix, means3D=None, opacity=None):
    # print("USING use_min_z_dist", use_min_z_dist)
    count = 0
    for d_x, d_y in zip([0, 1, 0, -1, 0, 1, -1, 1, -1, -2, 2, 0, 0, -2, 2, -2, 2, -1, 1, 1, -1. -2, 2, -2, 2], [0, 0, 1, 0, -1, 1, -1, -1, 1, 0, 0, -2, 2, -1, 1, 1, -1, -2, 2, -2, 2, -2, -2, 2, 2]):
        pix_mask =  torch.logical_and(
            means2D[:, 0] == pix[0] + d_x,
            means2D[:, 1] == pix[1] + d_y)
        if pix_mask.sum() != 0:
            possible_ids = torch.nonzero(pix_mask)
            gauss_id = possible_ids[0]
            return gauss_id
        count += 1
    return None


def find_closest_not_round(means2D, pix, visibility=None, thresh=0.5, from_closest=0, gt_colors=None, gs_colors=None, color_thresh=100, topk=30):
    dist = torch.cdist(pix.unsqueeze(0).unsqueeze(0), means2D.unsqueeze(0)).squeeze()
    dist_top_k = dist.topk(largest=False, k=topk)
    dist_color = torch.cdist(gt_colors.unsqueeze(0).unsqueeze(0).float(), 255*gs_colors[dist_top_k.indices[from_closest:]].unsqueeze(0).float()).squeeze()
    for i, k in enumerate(dist_top_k.indices[from_closest:]):
        if visibility[k, 0] >= thresh and dist_color[i] < color_thresh:
            return k
    if pix[0] != 0:
        print('did not find visible point using min dist\n')
    return dist_top_k.indices[0]


def find_closest_to_start_pixels(
        means2D,
        start_pixels,
        means3D=None,
        opacity=None,
        not_round=False,
        visibility=None,
        thresh=0.5,
        best_x=1,
        gt_colors=None,
        gs_colors=None,
        color_thresh=100,
        topk=30,
        start_pixels_2=None):
    gauss_ids = list()
    gs_traj_3D = list()
    for j, pix in enumerate(start_pixels):
        # print(pix, start_pixels_2[j])
        if not_round:
            best_x_gauss_ids = list()
            for i in range(best_x):
                gauss_id = find_closest_not_round(
                    means2D,
                    pix,
                    visibility,
                    thresh,
                    from_closest=i,
                    gt_colors=gt_colors[j],
                    gs_colors=gs_colors,
                    color_thresh=color_thresh,
                    topk=topk).unsqueeze(0)
                best_x_gauss_ids.append(gauss_id)
        else:
            best_x_gauss_ids = [find_closest(means2D, pix, means3D, opacity)]

        if best_x_gauss_ids[0] is not None:
            gauss_ids.extend(best_x_gauss_ids)
        else:
            gauss_ids.extend([torch.tensor([0]).to(means2D.device)]*best_x)

    return gauss_ids
        

def get_3D_trajs_for_track(gauss_ids, params, return_all=False, no_bg=False):
    gs_traj_3D = list()
    logit_opacities = list()
    logit_scales = list()
    rgb_colors = list()
    unnorm_rotations = list()
    visibility = list()

    for gauss_id in gauss_ids:
        if no_bg and 'bg' in params.keys():
            fg = params['bg'][gauss_id] < 0.5
        else:
            fg = 1 # torch.ones_like(params['logit_opacities']).bool() # params['bg'][gauss_id] < 1000

        if gauss_id != -1 and fg:
            gs_traj_3D.append(
                    params['means3D'][gauss_id].squeeze())
            unnorm_rotations.append(params['unnorm_rotations'][gauss_id].squeeze())
            logit_opacities.append(params['logit_opacities'][gauss_id].squeeze())
            if len(params['rgb_colors'].shape) == 2:
                rgb_colors.append(params['rgb_colors'][gauss_id])
            else:
                rgb_colors.append(params['rgb_colors'][:, :, 0][gauss_id])

            if 'visibility' in params.keys():
                visibility.append(params['visibility'][gauss_id].squeeze())
            else:
                visibility.append(
                    (torch.zeros(params['means3D'].shape[2])).to(params['means3D'].device))
        elif fg:
            gs_traj_3D.append(
                    (torch.ones_like(params['means3D'][0]).squeeze()*-1).to(params['means3D'].device))
            logit_opacities.append(torch.tensor(-1).to(params['means3D'].device))
            rgb_colors.append(torch.tensor([[-1, -1, -1]]).to(params['means3D'].device))
            unnorm_rotations.append(
                (torch.ones_like(params['unnorm_rotations'][0]).squeeze()*-1).to(params['means3D'].device))
            visibility.append(
                    (torch.zeros(params['means3D'].shape[2])).to(params['means3D'].device))
    # print(gs_traj_3D, logit_opacities, rgb_colors, unnorm_rotations, visibility)
    if return_all:
        return torch.stack(gs_traj_3D), torch.stack(logit_opacities), torch.stack(rgb_colors), torch.stack(unnorm_rotations), torch.stack(visibility)
    else:
        return torch.stack(gs_traj_3D)
    

def get_start_color(img, start_pix, mode='bilinear'):
    start_pix[:, 0] *= img.shape[1]
    start_pix[:, 1] *= img.shape[0]
    if mode == 'bilinear':
        floor = start_pix.floor().long()
        ceil = start_pix.ceil().long()
        colors = \
            img[floor[:, 1], floor[:, 0]] * (ceil[:, 1]-start_pix[:, 1]).unsqueeze(1) * (ceil[:, 0]-start_pix[:, 0]).unsqueeze(1) \
                + img[ceil[:, 1], floor[:, 0]] * (start_pix[:, 1]-floor[:, 1]).unsqueeze(1) * (ceil[:, 0]-start_pix[:, 0]).unsqueeze(1) \
                    + img[floor[:, 1], ceil[:, 0]] * (ceil[:, 1]-start_pix[:, 1]).unsqueeze(1) * (start_pix[:, 0]-floor[:, 0]).unsqueeze(1) \
                        + img[ceil[:, 1], ceil[:, 0]] * (start_pix[:, 1]-floor[:, 1]).unsqueeze(1) * (start_pix[:, 0]-floor[:, 0]).unsqueeze(1)
        colors /= ((ceil[:, 0] - floor[:, 0]) * (ceil[:, 1] - floor[:, 1])).unsqueeze(1)
        colors = torch.nan_to_num(colors, nan=0.0)
    else:
        rounded = torch.round(start_pix).long()
        colors = img[rounded[:, 1], rounded[:, 0]].float()
    return colors


def transform_to_world(config, gs_traj_3D):
    meta_path = os.path.join(
        config['data']['basedir'],
        os.path.dirname(os.path.dirname(config['data']['sequence'])),
        'meta.json')
    cam_id = int(os.path.basename(config['data']['sequence']))
    with open(meta_path, 'r') as jf:
        data = json.load(jf)
    idx = data['cam_id'][0].index(cam_id)
    w2c = torch.tensor(data['w2c'][0][idx]).to(gs_traj_3D.device).float()

    B, N, T, D = gs_traj_3D.shape
    # gs_traj_3D = gs_traj_3D.squeeze()
    gs_traj_3D = gs_traj_3D.view(B*N, T, D)

    pts_cam = gs_traj_3D
    pix_ones = torch.ones(pts_cam.shape[0], pts_cam.shape[1], 1).to(gs_traj_3D.device).float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=2)
    c2w = torch.inverse(w2c)
    gs_traj_3D = (c2w @ pts4.view(-1, D+1).T).T[:, :3]
    # gs_traj_3D = gs_traj_3D.view(N, T, D).unsqueeze(0)
    gs_traj_3D = gs_traj_3D.view(B, N, T, D)

    return gs_traj_3D


def _eval_traj(
        params,
        first_occurance,
        data,
        h=None,
        w=None,
        proj_matrix=None,
        use_only_pred=True,
        vis_trajs=False,
        results_dir=None,
        gauss_ids_to_track=None,
        dataset='jono',
        use_norm_pix=False,
        use_round_pix=False,
        do_scale=False,
        w2c=None,
        clip=True,
        use_gt_occ=False,
        vis_thresh=0.5,
        vis_thresh_start=100000,
        best_x=1, # 5,
        traj_len=10,
        color_thresh=1000,
        no_bg=False,
        cam=None,
        config=None,
        do_transform=False,
        get_gauss_wise3D_track=True,
        get_best_jaccard=True,
        get_from3D=False,
        vis_trajs_best_x=False,
        novel_view_mode=None,
    ):
    if params['means3D'][:, :, -1].sum() == 0:
        params['means3D'] = params['means3D'][:, :, :-1]
        params['unnorm_rotations'] = params['unnorm_rotations'][:, :, :-1]
    
    # get GT
    gt_traj_2D = data['points']
    if 'trajs' in data.keys():
        gt_traj_3D = data['trajs']
    else:
        gt_traj_3D = None

    gt_colors = get_start_color(data['video'][0], gt_traj_2D[:, 0, :].clone()).to(params['means3D'].device)
    occluded = data['occluded']
    if dataset == 'jono':
        gt_traj_2D[:, :, 0] = ((gt_traj_2D[:, :, 0] * w) - 1)/w
        gt_traj_2D[:, :, 1] = ((gt_traj_2D[:, :, 1] * h) - 1)/h

    if dataset == "jono":
        search_fg_only = True
    else:
        search_fg_only = False

    valids = 1-occluded.float()
    if novel_view_mode is not None:
        proj_matrices = torch.load(os.path.join(
            results_dir, f'poses_{novel_view_mode}.pth')).to(params['means3D'].device)
    else:
        proj_matrices = proj_matrix

    # params['visibility'] = (params['visibility'] > vis_thresh).float()
    # get trajectories of Gaussians
    gs_traj_2D, gs_traj_3D, pred_visibility, gs_traj_2D_for_vis = get_gs_traj_pts(
        proj_matrix,
        params,
        first_occurance,
        w,
        h,
        gt_traj_2D[:, 0].clone(),
        start_pixels_normalized=True,
        gauss_ids=gauss_ids_to_track,
        use_norm_pix=use_norm_pix,
        use_round_pix=use_round_pix,
        do_scale=do_scale,
        no_bg=no_bg,
        search_fg_only=search_fg_only, 
        w2c=w2c,
        thresh=vis_thresh_start,
        best_x=best_x,
        gt_colors=gt_colors,
        color_thresh=color_thresh,
        cam=cam,
        get_gauss_wise3D_track=get_gauss_wise3D_track,
        start_3D=gt_traj_3D[:, 0].clone().cuda() if gt_traj_3D is not None else gt_traj_3D,
        get_from3D=get_from3D,
        visuals=vis_trajs_best_x or vis_trajs,
        proj_matrices=proj_matrices)
    
    pred_visibility = (pred_visibility > vis_thresh).float()

    if best_x > 1 and vis_trajs_best_x:
        print('Visualizeing tracked points')
        data['points'] = normalize_points(gs_traj_2D_for_vis, h, w).squeeze()
        data['occluded'] = occluded.squeeze()
        data = {k: v.detach().clone().cpu().numpy() for k, v in data.items()}
        vis_trail(
            os.path.join(results_dir, 'tracked_points_vis'),
            data,
            pred_visibility=pred_visibility.squeeze(),
            vis_traj=True if traj_len > 0 else False)

    # N*best_x, T, D 
    gs_traj_2D = gs_traj_2D.reshape(-1, best_x, gs_traj_2D.shape[1], gs_traj_2D.shape[2]).permute(1, 0, 2, 3)
    gs_traj_3D = gs_traj_3D.reshape(-1, best_x, gs_traj_3D.shape[1], gs_traj_3D.shape[2]).permute(1, 0, 2, 3)
    pred_visibility = pred_visibility.reshape(-1, best_x, pred_visibility.shape[1]).permute(1, 0, 2)
    if vis_trajs_best_x or vis_trajs:
        gs_traj_2D_for_vis = gs_traj_2D_for_vis.reshape(gs_traj_2D_for_vis.shape[2], -1, best_x, gs_traj_2D_for_vis.shape[2], gs_traj_2D_for_vis.shape[3]).permute(0, 2, 1, 3, 4)

    # if point cloud defined in camera = world space
    # if gt_traj_3D is not None and do_transform:
    #     print("TRANSFORMNING!!!")
    #     gs_traj_3D = transform_to_world(config, gs_traj_3D)

    if dataset != "iphone" and get_best_jaccard:
        samples = sample_queries_first(
            occluded.cpu().bool().numpy().squeeze(),
            unnormalize_points(copy.deepcopy(gt_traj_2D.cpu().numpy().squeeze()), h, w),
            ignore_invalid=True)
        samples = {k: v.repeat(best_x, axis=0) for k, v in samples.items()}
        min_idx = torch.from_numpy(get_smallest_AJ(
            samples['query_points'],
            samples['occluded'],
            samples['target_points'],
            (1-(pred_visibility > vis_thresh).float()).cpu().numpy(),
            gs_traj_2D.cpu().numpy(),
            W=w,
            H=h,
            use_gt_occ=use_gt_occ)).to(gs_traj_2D.device)
    elif dataset != "iphone":
        # Take the best of several Gaussials
        min_idx = get_smallest_l2(
            gs_traj_2D,
            pred_visibility,
            gt_traj_2D.unsqueeze(0).repeat(best_x, 1, 1, 1).to(gs_traj_2D.device),
            valids.unsqueeze(0).repeat(best_x, 1, 1).to(gs_traj_2D.device),
            W=w, H=h,
            median=True)
    else:
        min_idx = 0

    pred_visibility = pred_visibility[min_idx, torch.arange(pred_visibility.shape[1]), :]

    gs_traj_2D = gs_traj_2D[min_idx, torch.arange(gs_traj_2D.shape[1]), :, :]
    gs_traj_3D = gs_traj_3D[min_idx, torch.arange(gs_traj_3D.shape[1]), :, :]
    if vis_trajs_best_x or vis_trajs:
        gs_traj_2D_for_vis = gs_traj_2D_for_vis[:, min_idx, torch.arange(gs_traj_2D_for_vis.shape[2]), :, :]

    # make predicted visinbility bool
    pred_visibility = (pred_visibility > vis_thresh).float()

    # unnormalize gt to image pixels
    gt_traj_2D = unnormalize_points(gt_traj_2D, h, w)

    # unsqueeze to batch dimension
    if len(gt_traj_2D.shape) == 3:
        gt_traj_2D = gt_traj_2D.unsqueeze(0)
        gs_traj_2D = gs_traj_2D.unsqueeze(0)
        valids = valids.unsqueeze(0)
        occluded = occluded.unsqueeze(0)
        pred_visibility = pred_visibility.unsqueeze(0)
        if gt_traj_3D is not None:
            gt_traj_3D = gt_traj_3D.unsqueeze(0)
            gs_traj_3D = gs_traj_3D.unsqueeze(0)

    if dataset != "iphone":
        # mask by valid ids
        if valids.sum() != 0:
            gs_traj_2D, gt_traj_2D, valids, occluded, pred_visibility, gt_traj_3D, gs_traj_3D, gs_traj_2D_for_vis = mask_valid_ids(
                valids,
                gs_traj_2D,
                gt_traj_2D,
                occluded,
                pred_visibility,
                gt_traj_3D,
                gs_traj_3D,
                gs_traj_2D_for_vis=gs_traj_2D_for_vis if vis_trajs_best_x or vis_trajs else None)

        # compute metrics from pips
        pips_metrics = compute_metrics(
            h,
            w,
            gs_traj_2D.to(gt_traj_2D.device),
            gt_traj_2D,
            valids
        )
        metrics = {'pips': pips_metrics}

        if 'trajs' in data.keys():
            # multiply by 100 because cm evaluation
            metrics3D = compute_metrics(
                None,
                None,
                (gs_traj_3D*100).to(gt_traj_2D.device),
                gt_traj_3D*100,
                valids,
                sur_thr=50,
                norm_factor=None)
            metrics.update({'pips_3D': {f'{k}_3D': v for k, v in metrics3D.items()}})
            
        if (1-occluded.long()).sum() != 0:
            # compute metrics from tapvid
            samples = sample_queries_first(
                occluded.cpu().bool().numpy().squeeze(),
                gt_traj_2D.cpu().numpy().squeeze())
            tapvid_metrics = compute_tapvid_metrics(
                samples['query_points'],
                samples['occluded'],
                samples['target_points'],
                (1-pred_visibility).cpu().numpy(),
                gs_traj_2D.cpu().numpy(),
                W=w,
                H=h,
                use_gt_occ=use_gt_occ)
            metrics.update({'tapvid': tapvid_metrics})

    else:
        metrics = dict()
        _pred_visibility = pred_visibility[0, :, data['time_ids']].cpu().permute(1, 0).numpy()
        _gs_traj_2D = gs_traj_2D[0, :, data['time_ids']].cpu().permute(1, 0, 2).numpy()
        _gs_traj_3D = gs_traj_3D[0, :, data['time_ids']].cpu().permute(1, 0, 2).numpy()
        metrics.update(compute_metrics_iphone(data, _pred_visibility, _gs_traj_2D))
        metrics.update(compute_metrics_iphone3D(data, _gs_traj_3D))

    if vis_trajs:
        print('Visualizeing tracked points')
        data['points'] = normalize_points(gs_traj_2D_for_vis, h, w).squeeze()
        data['occluded'] = occluded.squeeze()
        data = {k: v.detach().clone().cpu().numpy() for k, v in data.items()}
        if novel_view_mode is not None:
            data['video'] = np.stack(imageio.mimread(
                os.path.join(results_dir, f'rendered_rgb_{novel_view_mode}/vid_trails.mp4')))
        vis_trail(
            os.path.join(results_dir, 'tracked_points_vis', f'_{novel_view_mode}'),
            data,
            pred_visibility=pred_visibility.squeeze(),
            vis_traj=True if traj_len > 0 else False)
        
    return metrics


def get_smallest_l2(gs_traj_2D, pred_visibility, gt_traj_2D, valids, W, H, norm_factor=256, median=False):
    B, N, S = gt_traj_2D.shape[0], gt_traj_2D.shape[1], gt_traj_2D.shape[2]
    
    # permute number of points and seq len
    gs_traj_2D = gs_traj_2D.permute(0, 2, 1, 3)
    gt_traj_2D = gt_traj_2D.permute(0, 2, 1, 3)

    # get metrics
    metrics = dict()
    d_sum = 0.0
    sc_pt = torch.tensor(
        [[[W/norm_factor, H/norm_factor]]]).float().to(gs_traj_2D.device)
    
    dists = torch.linalg.norm(gs_traj_2D/sc_pt - gt_traj_2D/sc_pt, dim=-1, ord=2) # B,S,N
    if median:
        dists_ = dists.permute(0,2,1).reshape(B*N,S)
        valids_ = valids.permute(0,2,1).reshape(B*N,S)
        median_l2 = reduce_masked_median(dists_, valids_, keep_batch=True).reshape(B, N)
    else:
        median_l2 = dists.mean(dim=1)
    min_idx = median_l2.min(dim=0).indices
    return min_idx


def compute_metrics_iphone3D(data, pred_points):
    keypoints_3d = data["trajs"].permute(1, 0, 2).numpy()
    visibility = ~(data['occluded_trajs'].permute(1, 0).numpy().astype(np.bool_))
    time_pairs = data["time_pairs"]
    index_pairs = data["index_pairs"]

    print(keypoints_3d)
    print(pred_points)
    
    # Compute 3D tracking metrics.
    pair_keypoints_3d = keypoints_3d[index_pairs]
    pair_visibility = visibility[index_pairs]
    pred_points = pred_points[index_pairs]
    is_covisible = (pair_visibility == 1).all(axis=1)
    target_keypoints_3d = pair_keypoints_3d[:, 1, :, :3]
    pred_points = pred_points[:, 1, :, :3]
    epes = []
    for i in range(len(time_pairs)):
        epes.append(
            np.linalg.norm(
                target_keypoints_3d[i][is_covisible[i]]
                - pred_points[i][is_covisible[i]],
                axis=-1,
            )
        )
    epe = np.mean(
        [frame_epes.mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    pck_3d_50cm = np.mean(
        [(frame_epes < 0.5).mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    pck_3d_10cm = np.mean(
        [(frame_epes < 0.1).mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    pck_3d_5cm = np.mean(
        [(frame_epes < 0.05).mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    print(f"3D tracking EPE: {epe:.4f}")
    print(f"3D tracking PCK (50cm): {pck_3d_50cm:.4f}")
    print(f"3D tracking PCK (10cm): {pck_3d_10cm:.4f}")
    print(f"3D tracking PCK (5cm): {pck_3d_5cm:.4f}")
    print("-----------------------------")
    return {'epe': epe, 'pck_3d_50cm': pck_3d_50cm, 'pck_3d_10cm': pck_3d_10cm, 'pck_3d_5cm': pck_3d_5cm}


def compute_metrics_iphone(data, pred_visibilities, pred_points):
    target_points = data["points"].permute(1, 0, 2).numpy()
    visibilities = ~(data['occluded'].permute(1, 0).numpy().astype(np.bool_))
    pred_visibilities = pred_visibilities.astype(np.bool_)
    time_ids = data["time_ids"]
    num_frames = len(time_ids)
    num_pts = target_points.shape[1]

    target_points = target_points[None].repeat(num_frames, axis=0)[..., :2]
    target_visibilities = visibilities[None].repeat(num_frames, axis=0)

    pred_points = pred_points[None].repeat(num_frames, axis=0)
    pred_visibilities = pred_visibilities[None].repeat(num_frames, axis=0)
    pred_visibilities = pred_visibilities.reshape(
        num_frames, -1, num_pts)

    one_hot_eye = np.eye(target_points.shape[0])[..., None].repeat(num_pts, axis=-1)
    evaluation_points = one_hot_eye == 0

    for i in range(num_frames):
        evaluation_points[i, :, ~visibilities[i]] = False

    occ_acc = np.sum(
        np.equal(pred_visibilities, target_visibilities) & evaluation_points
    ) / np.sum(evaluation_points)

    all_frac_within = []
    all_jaccard = []
    for thresh in [4, 8, 16, 32, 64]:
        within_dist = np.sum(
            np.square(pred_points - target_points),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, target_visibilities)

        count_correct = np.sum(is_correct & evaluation_points)
        count_visible_points = np.sum(target_visibilities & evaluation_points)
        frac_correct = count_correct / count_visible_points
        all_frac_within.append(frac_correct)
        true_positives = np.sum(is_correct & pred_visibilities & evaluation_points)
        gt_positives = np.sum(target_visibilities & evaluation_points)
        false_positives = (~target_visibilities) & pred_visibilities
        false_positives = false_positives | ((~within_dist) & pred_visibilities)
        false_positives = np.sum(false_positives & evaluation_points)
        jaccard = true_positives / (gt_positives + false_positives)
        all_jaccard.append(jaccard)
    AJ = np.mean(all_jaccard)
    APCK = np.mean(all_frac_within)

    print(f"2D tracking AJ: {AJ:.4f}")
    print(f"2D tracking avg PCK: {APCK:.4f}")
    print(f"2D tracking occlusion accuracy: {occ_acc:.4f}")
    print("-----------------------------")
    return {'AJ': AJ, 'APCK': APCK, 'occ_acc': occ_acc}


def mask_valid_ids(valids, gs_traj_2D, gt_traj_2D, occluded, pred_visibility, gt_traj_3D, gs_traj_3D, gs_traj_2D_for_vis=None):
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
    pred_visibility = pred_visibility.reshape(
        shape[0]*shape[1], shape[2])[vis_ok].reshape(
            shape[0], -1, shape[2])
    if gt_traj_3D is not None:
        shape = gt_traj_3D.shape
        gt_traj_3D = gt_traj_3D.reshape(
        shape[0]*shape[1], shape[2], shape[3])[vis_ok].reshape(
            shape[0], -1, shape[2], shape[3])
        gs_traj_3D = gs_traj_3D.reshape(
        shape[0]*shape[1], shape[2], shape[3])[vis_ok].reshape(
            shape[0], -1, shape[2], shape[3])
    if gs_traj_2D_for_vis is not None:
        gs_traj_2D_for_vis = gs_traj_2D_for_vis[:, vis_ok, :, :]

    return gs_traj_2D, gt_traj_2D, valids, occluded, pred_visibility, gt_traj_3D, gs_traj_3D, gs_traj_2D_for_vis


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
    if norm_factor is not None:
        sc_pt = torch.tensor(
            [[[W/norm_factor, H/norm_factor]]]).float().to(gs_traj_2D.device)
    else:
        sc_pt = torch.tensor(
            [[[1, 1, 1]]]).float().to(gs_traj_2D.device)

    for thr in thrs:
        # note we exclude timestep0 from this eval
        d_ = (torch.linalg.norm(
            gs_traj_2D[:,1:]/sc_pt - gt_traj_2D[:,1:]/sc_pt, dim=-1, ord=2) < thr).float() # B,S-1,N
        d_ = reduce_masked_mean(d_, valids[:,1:]).item()*100.0
        d_sum += d_
        metrics['d_%d' % thr] = d_
    d_avg = d_sum / len(thrs)
    metrics['d_avg'] = d_avg
    
    dists = torch.linalg.norm(gs_traj_2D/sc_pt - gt_traj_2D/sc_pt, dim=-1, ord=2) # B,S,N
    dist_ok = 1 - (dists > sur_thr).float() * valids # B,S,N
    survival = torch.cumprod(dist_ok, dim=1) # B,S,N
    metrics['survival'] = torch.mean(survival).item()*100.0

    # get the median l2 error for each trajectory
    dists_ = dists.permute(0,2,1).reshape(B*N,S)
    valids_ = valids.permute(0,2,1).reshape(B*N,S)
    median_l2 = reduce_masked_median(dists_, valids_, keep_batch=True)
    metrics['median_l2'] = median_l2.mean().item()

    return metrics


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    ignore_invalid=False):
    """Package a set of frames and tracks for use in TAPNet evaluations.
    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.
    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1]
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1]
    """
    if ignore_invalid:
        target_occluded = np.zeros_like(target_occluded, dtype=bool)
    valid = np.sum(~target_occluded, axis=1) > 0

    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)
    return {
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def get_smallest_AJ(
        query_points: np.ndarray,
        gt_occluded: np.ndarray,
        gt_tracks: np.ndarray,
        pred_occluded: np.ndarray,
        pred_tracks: np.ndarray,
        query_mode: str = 'first',
        norm_factor=256,
        W=256,
        H=256,
        use_gt_occ=False):
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """
    # SCALE to 256  
    sc_pt = np.array(
        [[[[W/norm_factor, H/norm_factor]]]])
    gt_tracks = gt_tracks/sc_pt
    pred_tracks = pred_tracks/sc_pt
    query_points[:, :, 1:] = query_points[:, :, 1:]/sc_pt[0]

    metrics = {}
    if use_gt_occ:
        pred_occluded = gt_occluded

    # Don't evaluate the query point.  Numpy doesn't have one_hot, so we
    # replicate it by indexing into an identity matrix.
    one_hot_eye = np.eye(gt_tracks.shape[2])
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = one_hot_eye[query_frame] == 0

    # If we're using the first point on the track as a query, don't evaluate the
    # other points.
    if query_mode == "first":
        for i in range(gt_occluded.shape[0]):
            index = np.where(gt_occluded[i] == 0)[0][0]
            evaluation_points[i, :index] = False
    elif query_mode != "strided":
        raise ValueError("Unknown query mode " + query_mode)

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = (
            np.sum(
                np.square(pred_tracks - gt_tracks),
                axis=-1,
            )
            < np.square(thresh)
        )
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(2))
        frac_correct = count_correct / count_visible_points
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(2)
        )
        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(2))
        jaccard = true_positives / (gt_positives + false_positives)
        all_jaccard.append(jaccard)

    all_jaccard =  np.mean(np.stack(all_jaccard, axis=1), axis=1)
    all_frac_within = np.mean(np.stack(all_frac_within, axis=1), axis=1)
    min_idx = np.argmax(all_jaccard, axis=0)
    return min_idx


def compute_tapvid_metrics(
        query_points: np.ndarray,
        gt_occluded: np.ndarray,
        gt_tracks: np.ndarray,
        pred_occluded: np.ndarray,
        pred_tracks: np.ndarray,
        query_mode: str = 'first',
        norm_factor=256,
        W=256,
        H=256,
        use_gt_occ=False):
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """
    # SCALE to 256  
    sc_pt = np.array(
        [[[[W/norm_factor, H/norm_factor]]]])
    gt_tracks = gt_tracks/sc_pt
    pred_tracks = pred_tracks/sc_pt
    query_points[:, :, 1:] = query_points[:, :, 1:]/sc_pt[0]

    metrics = {}
    if use_gt_occ:
        pred_occluded = gt_occluded

    # Don't evaluate the query point.  Numpy doesn't have one_hot, so we
    # replicate it by indexing into an identity matrix.
    one_hot_eye = np.eye(gt_tracks.shape[2])
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = one_hot_eye[query_frame] == 0

    # If we're using the first point on the track as a query, don't evaluate the
    # other points.
    if query_mode == "first":
        for i in range(gt_occluded.shape[0]):
            index = np.where(gt_occluded[i] == 0)[0][0]
            evaluation_points[i, :index] = False
    elif query_mode != "strided":
        raise ValueError("Unknown query mode " + query_mode)

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = (
        np.sum(
            np.equal(pred_occluded, gt_occluded) & evaluation_points,
            axis=(1, 2),
        )
        / np.sum(evaluation_points)
    )
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = (
            np.sum(
                np.square(pred_tracks - gt_tracks),
                axis=-1,
            )
            < np.square(thresh)
        )
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
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
        gauss_ids_to_track=None,
        input_k=None,
        input_w2c=None, 
        load_gaussian_tracks=False,
        use_norm_pix=False,
        use_round_pix=False,
        do_scale=False,
        clip=True,
        use_gt_occ=False,
        vis_thresh=0.5,
        vis_thresh_start=0.5,
        best_x=1,
        traj_len=10,
        color_thresh=1000,
        do_transform=False,
        get_gauss_wise3D_track=True,
        get_from3D=False,
        vis_trajs_best_x=False,
        stereo=False,
        novel_view_mode=None):

    # get projectoin matrix
    if cam is None:
        params, _, k, w2c = load_scene_data(config,  os.path.dirname(results_dir))
        if 'visibility' not in params.keys():
            params['visibility'] = torch.ones((params['means3D'].shape[0], params['means3D'].shape[-1]))
        if not stereo and len(params['visibility'].shape) == 3:
            params['visibility'] = params['visibility'][:, 0, :]
        elif len(params['visibility'].shape) == 3:
            params['visibility'] = params['visibility'][:, 1, :]

        if k is None:
            k = input_k
            w2c = input_w2c
        h, w = config["data"]["desired_image_height"], config["data"]["desired_image_width"]
        proj_matrix = get_projection_matrix(w, h, k, w2c).squeeze()
        cam = setup_camera(w, h, k, w2c, device=params['means3D'].device)

        if 'gauss_ids_to_track' in params.keys() and load_gaussian_tracks:
            gauss_ids_to_track = params['gauss_ids_to_track'].long()
            if gauss_ids_to_track.sum() == 0:
                gauss_ids_to_track = None
    else:
        proj_matrix = cam.projmatrix.squeeze()
        h = cam.image_height
        w = cam.image_width
        w2c = None

    visible = params['visibility'][:, 0] > vis_thresh
    for k, v in params.items():
        try:
            params[k] = v[visible]
        except:
            params[k] = v

    # pts = params['means3D'][:, :, 0].float()
    # pts_ones = torch.ones(pts.shape[0], 1).to(params['means3D'].device).float()
    # pts4 = torch.cat((pts, pts_ones), dim=1)
    # transformed_pts = (w2c @ pts4.T).T[:, :3]

    # get gt data
    data = get_gt_traj(config, in_torch=True, stereo=stereo)
    if 'davis' in config['data']["gradslam_data_cfg"].lower():
        dataset = 'davis'
    elif 'iphone' in config['data']["gradslam_data_cfg"].lower():
        dataset = 'iphone'
    elif 'jono' in config['data']["gradslam_data_cfg"].lower():
        dataset = 'jono'

    # get metrics
    metrics = _eval_traj(
        params,
        params['timestep'],
        data,
        proj_matrix=proj_matrix,
        h=h,
        w=w,
        results_dir=results_dir,
        vis_trajs=vis_trajs,
        gauss_ids_to_track=gauss_ids_to_track,
        dataset=dataset,
        use_norm_pix=use_norm_pix,
        use_round_pix=use_round_pix,
        do_scale=do_scale,
        w2c=w2c,
        clip=clip,
        use_gt_occ=use_gt_occ,
        vis_thresh=vis_thresh,
        vis_thresh_start=vis_thresh_start,
        best_x=best_x,
        traj_len=traj_len,
        color_thresh=color_thresh,
        cam=cam,
        config=config,
        do_transform=do_transform,
        get_gauss_wise3D_track=get_gauss_wise3D_track,
        get_from3D=get_from3D,
        vis_trajs_best_x=vis_trajs_best_x,
        novel_view_mode=novel_view_mode)

    return metrics


def meshgrid2d(B, Y, X, stack=False, norm=False, device='cuda:0', on_chans=False):
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


def get_xy_grid(H, W, N=2048, B=1, device='cuda:0', from_mask=False):
    # pick N points to track; we'll use a uniform grid
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = meshgrid2d(B, N_, N_, stack=False, norm=False, device=device)
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
    xy0 = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
    if from_mask:
        xy0 = from_fg_mask(device, H, W, xy0=xy0.long())
    return xy0

def from_fg_mask(device, H, W, xy0=None, mask_path='mask2.jpg'):
    mask = imageio.imread(mask_path)
    mask = cv2.resize(
            mask.astype(float),
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        )
    mask = torch.from_numpy(mask).to(device)
    mask[mask <=125] = 0
    mask[mask>125] = 1
    mask = mask.bool()
    candidates = torch.zeros_like(mask, dtype=bool, device=device)
    candidates[xy0[0, :, 1], xy0[0, :, 0]] = True
    candidates = mask & candidates
    candidates = torch.nonzero(candidates)
    candidates = torch.stack([candidates[:, 1], candidates[:, 0]], dim=1)
    return candidates.unsqueeze(0)



def vis_grid_trajs(
        config,
        params=None,
        cam=None,
        results_dir=None,
        orig_image_size=False,
        no_bg=True,
        clip=True,
        traj_len=10,
        vis_thresh=0.5,
        stereo=False,
        novel_view_mode=None):

    # get projectoin matrix
    if cam is None:
        params, _, k, w2c = load_scene_data(config, os.path.dirname(results_dir))
        if not stereo and len(params['visibility'].shape) == 3:
            params['visibility'] = params['visibility'][:, 0, :]
        elif len(params['visibility'].shape) == 3:
            params['visibility'] = params['visibility'][:, 1, :]
            
        if orig_image_size:
            k, pose, h, w = get_cam_data(config, orig_image_size)
            # w2c = torch.linalg.inv(pose)
            proj_matrix = get_projection_matrix(w, h, k, w2c, device=params['means3D'].device).squeeze()
        else:
            h, w = config["data"]["desired_image_height"], config["data"]["desired_image_width"]
            proj_matrix = get_projection_matrix(w, h, k, w2c, device=params['means3D'].device).squeeze()
    else:
        proj_matrix = cam.projmatrix.squeeze()
        h = cam.image_height
        w = cam.image_width
    # pdb.set_trace()

    if novel_view_mode is not None:
        proj_matrices = torch.load(os.path.join(
            results_dir, f'poses_{novel_view_mode}.pth')).to(params['means3D'].device)
    else:
        proj_matrices = proj_matrix

    visible = params['visibility'][:, 0] > vis_thresh
    for k, v in params.items():
        try:
            params[k] = v[visible]
        except:
            params[k] = v

    # get trajectories to track
    N = 2048 if no_bg else 1024
    start_pixels = get_xy_grid(
        h,
        w,
        N=N,
        device=params['means3D'].device).squeeze().long()
    
    gs_traj_2D, gs_traj_3D, pred_visibility, gs_traj_2D_for_vis = get_gs_traj_pts(
        proj_matrix,
        params,
        params['timestep'],
        w,
        h,
        start_pixels,
        start_pixels_normalized=False,
        no_bg=no_bg,
        do_scale=False,
        get_gauss_wise3D_track=True,
        visuals=True,
        proj_matrices=proj_matrices)
    
    pred_visibility = (pred_visibility > vis_thresh).float()

    # get gt data for visualization (actually only need rgb here)
    data = get_gt_traj(config, in_torch=True, stereo=stereo)
    data['points'] = normalize_points(gs_traj_2D_for_vis, h, w).squeeze()    
    data['occluded'] = torch.zeros(data['points'].shape[:-1]).to(data['points'].device)
    data = {k: v.detach().clone().cpu().numpy() for k, v in data.items()}

    if novel_view_mode is not None:
        data['video'] = np.stack(imageio.mimread(
            os.path.join(results_dir, f'rendered_rgb_{novel_view_mode}/vid_trails.mp4')))
    
    pred_visibility = (pred_visibility > vis_thresh).float()
    print("Visualizing grid...")
    # vis_tracked_points(
    #     os.path.join(results_dir, 'grid_points_vis'),
    #     data,
    #     clip=clip,
    #     pred_visibility=pred_visibility.squeeze(),
    #     traj_len=traj_len)
    vis_trail(
            os.path.join(results_dir, 'grid_points_vis'  + f'_{novel_view_mode}'),
            data,
            pred_visibility=torch.ones_like(pred_visibility.squeeze()).to(pred_visibility.device),
            vis_traj=True if traj_len > 0 else False)
