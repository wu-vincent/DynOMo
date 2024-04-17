"""
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file found here:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# For inquiries contact  george.drettakis@inria.fr

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE #####
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################
"""

import numpy as np
import torch
import torch.nn.functional as func
from torch.autograd import Variable
from math import exp
from torch_scatter import scatter_add


def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]) + 1e-20
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def normalize_quat(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    return q


def calc_mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def calc_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def accumulate_mean2d_gradient(variables, time_idx):
    variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means2D_grad'][variables['seen'], :2], dim=-1)
    # variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
    #     variables['means2D'].grad[variables['seen'], :2], dim=-1)
    variables['denom'][variables['seen']] += 1
    return variables


def update_params_and_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group['params'][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state
        params[k] = group["params"][0]
    return params


def cat_params_to_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            params[k] = group["params"][0]
    return params


def remove_points(
        to_remove,
        params,
        variables,
        optimizer=None,
        offset_0=None,
        support_trajs_trans=None):
    to_keep = ~to_remove
    idxs_to_keep = torch.arange(params['means3D'].shape[0])[to_keep].to(
        variables['self_indices'].device)
    to_keep_idx_mask = torch.isin(variables['self_indices'], idxs_to_keep) 
    to_keep_idx_mask = to_keep_idx_mask & torch.isin(variables['neighbor_indices'], idxs_to_keep)

    # mask parameters
    for k in [k for k in params.keys() if k not in ['cam_unnorm_rots', 'cam_trans']]:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]

    # mask variables
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    variables['seen'] = variables['seen'][to_keep]
    variables['moving'] = variables['moving'][to_keep]
    variables['x_coord_im'] = variables['x_coord_im'][to_keep]
    variables['y_coord_im'] = variables['y_coord_im'][to_keep]
    
    if 'means2D_grad' in variables.keys():
        variables['means2D_grad'] = variables['means2D_grad'][to_keep]
    if 'means2D' in variables.keys():
        variables['means2D'] = variables['means2D'][to_keep]
    if 'instseg' in variables.keys():
        variables['instseg'] = variables['instseg'][to_keep]
    if 'timestep' in variables.keys():
        variables['timestep'] = variables['timestep'][to_keep]
    if 'bg' in variables.keys():
        variables['bg'] = variables['bg'][to_keep]
    
    # mask kNN and map indices to new indices
    variables['neighbor_indices'] = variables['neighbor_indices'][to_keep_idx_mask]
    variables['neighbor_weight'] = variables['neighbor_weight'][to_keep_idx_mask]
    variables['neighbor_dist'] = variables['neighbor_dist'][to_keep_idx_mask]
    variables['self_indices'] = variables['self_indices'][to_keep_idx_mask]

    mapping_tensor = torch.zeros(idxs_to_keep.max().item() + 1).long().to(idxs_to_keep.device)
    mapping_tensor[idxs_to_keep] = torch.arange(idxs_to_keep.shape[0]).long().to(idxs_to_keep.device)

    variables['neighbor_indices'] = mapping_tensor[variables['neighbor_indices']]
    variables['self_indices'] = mapping_tensor[variables['self_indices']]

    # mask offset_0 and support trajs
    if offset_0 is not None:
        offset_0 = offset_0[to_keep_idx_mask]
    if support_trajs_trans is not None:
        support_trajs_trans = support_trajs_trans[to_keep]

    return params, variables, offset_0, support_trajs_trans


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def prune_gaussians(
        params,
        variables,
        optimizer,
        iter,
        prune_dict,
        curr_time_idx,
        offset_0,
        support_trajs_trans):
    variables['means2D_grad'] = variables['means2D'].grad
    if iter <= prune_dict['stop_after']:
        if (iter >= prune_dict['start_after']) and (iter % prune_dict['prune_every'] == 0):
            if iter == prune_dict['stop_after']:
                remove_threshold = prune_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = prune_dict['removal_opacity_threshold']
            # Remove Gaussians with low opacity
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()

            # Remove Gaussians with large kNN dist
            # recompute kNN distance
            with torch.no_grad():
                pdist = torch.nn.PairwiseDistance(p=2)
                offset_t_mag = pdist(
                    params['means3D'][variables["self_indices"], :, curr_time_idx-1],
                    params['means3D'][variables["neighbor_indices"], :, curr_time_idx-1])
                offset_t_mag = scatter_add(offset_t_mag, variables["self_indices"], dim=0)
                
                offset_0_mag = torch.linalg.norm(offset_0, dim=1)
                offset_0_mag = scatter_add(offset_0_mag, variables["self_indices"], dim=0)

                rel_drift = torch.abs(offset_t_mag-offset_0_mag) / offset_0_mag
                drift = rel_drift > prune_dict['kNN_rel_drift']
                to_remove = torch.logical_or(to_remove, drift)

            # Remove Gaussians that are too big
            if iter >= prune_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            if to_remove.sum():
                params, variables, offset_0, support_trajs_trans = remove_points(
                    to_remove,
                    params,
                    variables,
                    optimizer,
                    offset_0,
                    support_trajs_trans)
            torch.cuda.empty_cache()
        
            print(f'Removed {to_remove.sum()} Gaussians during pruning at Iteration {iter}!')
        
        # Reset Opacities for all Gaussians
        if iter > 0 and iter % prune_dict['reset_opacities_every'] == 0 and prune_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
    
    return params, variables, offset_0, support_trajs_trans

def clone_vars(params, variables, to_clone):
    device = variables['self_indices'].device
    idxs_to_clone = torch.arange(params['means3D'].shape[0])[to_clone].to(device)
    to_clone = to_clone.to(device)
    to_clone_self_idx = torch.isin(variables['self_indices'], idxs_to_clone)
    variables['neighbor_indices'] = torch.cat(
        (variables['neighbor_indices'], variables['neighbor_indices'][to_clone_self_idx]), dim=0)
    variables['neighbor_weight'] = torch.cat(
        (variables['neighbor_weight'], variables['neighbor_weight'][to_clone_self_idx]), dim=0)
    variables['neighbor_dist'] = torch.cat(
        (variables['neighbor_dist'], variables['neighbor_dist'][to_clone_self_idx]), dim=0)
    self_idxs = variables['self_indices'][to_clone_self_idx]
    for i, v in enumerate(torch.unique(variables['self_indices'][to_clone_self_idx])):
        self_idxs[self_idxs==v] = i + params['means3D'].shape[0]
    variables['self_indices'] = torch.cat(
        (variables['self_indices'], self_idxs), dim=0)
    variables['timestep'] = torch.cat((variables['timestep'], variables['timestep'][to_clone]), dim=0)
    
    variables['means2D_gradient_accum'] = torch.cat((variables['means2D_gradient_accum'], variables['means2D_gradient_accum'][to_clone]), dim=0)
    variables['denom'] = torch.cat((variables['denom'], variables['denom'][to_clone]), dim=0)
    variables['max_2D_radius'] = torch.cat((variables['max_2D_radius'], variables['max_2D_radius'][to_clone]), dim=0)
    variables['seen'] = torch.cat((variables['seen'], variables['seen'][to_clone]), dim=0)
    variables['means2D_grad'] = torch.cat((variables['means2D_grad'], variables['means2D_grad'][to_clone]), dim=0)
    variables['means2D'] = torch.cat((variables['means2D'], variables['means2D'][to_clone]), dim=0)
    variables['moving'] = torch.cat((variables['moving'], variables['moving'][to_clone]), dim=0)

    if 'normals' in variables.keys():
        variables['normals'] = torch.cat((variables['normals'], variables['normals'][to_clone]), dim=0)
    return variables


def split_vars(params, variables, to_split, n):
    device = variables['self_indices'].device
    idxs_to_split = torch.arange(params['means3D'].shape[0])[to_split].to(device)
    to_split_self_idx = torch.isin(variables['self_indices'], idxs_to_split)
    to_split = to_split.to(device)
    variables['neighbor_indices'] = torch.cat(
        (variables['neighbor_indices'], variables['neighbor_indices'][to_split_self_idx].repeat(n)), dim=0)
    variables['neighbor_weight'] = torch.cat(
        (variables['neighbor_weight'], variables['neighbor_weight'][to_split_self_idx].repeat(n)), dim=0)
    variables['neighbor_dist'] = torch.cat(
        (variables['neighbor_dist'], variables['neighbor_dist'][to_split_self_idx].repeat(n)), dim=0)
    self_idxs = variables['self_indices'][to_split_self_idx]
    for i, v in enumerate(torch.unique(variables['self_indices'][to_split_self_idx])):
        self_idxs[self_idxs==v] = i + params['means3D'].shape[0]
    variables['self_indices'] = torch.cat(
        (variables['self_indices'], self_idxs), dim=0)
    variables['timestep'] = torch.cat((variables['timestep'], variables['timestep'][to_split].repeat(n)), dim=0)
    
    variables['means2D_gradient_accum'] = torch.cat((variables['means2D_gradient_accum'], variables['means2D_gradient_accum'][to_split].repeat(n)), dim=0)
    variables['denom'] = torch.cat((variables['denom'], variables['denom'][to_split].repeat(n)), dim=0)
    variables['max_2D_radius'] = torch.cat((variables['max_2D_radius'], variables['max_2D_radius'][to_split].repeat(n)), dim=0)
    variables['seen'] = torch.cat((variables['seen'], variables['seen'][to_split].repeat(n)), dim=0)
    variables['means2D_grad'] = torch.cat((variables['means2D_grad'], variables['means2D_grad'][to_split].repeat(n, 1)), dim=0)
    variables['means2D'] = torch.cat((variables['means2D'], variables['means2D'][to_split].repeat(n, 1)), dim=0)
    variables['moving'] = torch.cat((variables['moving'], variables['moving'][to_split].repeat(n)), dim=0)
    
    if 'normals' in variables.keys():
        variables['normals'] = torch.cat((variables['normals'], variables['normals'][to_split].repeat(n, 1)), dim=0)
    
    return variables


def densify(params, variables, optimizer, iter, densify_dict, time_idx):
    device = params['means3D'].device
    if iter <= densify_dict['stop_after']:
        variables = accumulate_mean2d_gradient(variables, time_idx)
        grad_thresh = densify_dict['grad_thresh']
        if (iter >= densify_dict['start_after']) and (iter % densify_dict['densify_every'] == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0

            # Remove Gaussians with large kNN dist
            # recompute kNN distance
            with torch.no_grad():
                pdist = torch.nn.PairwiseDistance(p=2)
                dist = pdist(
                    params['means3D'][variables["self_indices"], :, time_idx-1],
                    params['means3D'][variables["neighbor_indices"], :, time_idx-1])
                dist = scatter_add(dist, variables["self_indices"], dim=0) 
                far_away = torch.logical_and(dist < densify_dict['kNN_dist_thresh_max'] * variables['scene_radius'],
                        dist > densify_dict['kNN_dist_thresh_min'] * variables['scene_radius'])

            # clone
            if densify_dict['scale_split_thresh'] == 'scene_radius':
                thresh = 0.01 * variables['scene_radius']
            else:
                thresh = torch.median(torch.max(torch.exp(params['log_scales']), dim=1).values) * 0.001
            to_clone = torch.logical_or(torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['log_scales']), dim=1).values <= thresh)), far_away).to(device)
            
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            variables = clone_vars(params, variables, to_clone)
            params = cat_params_to_optimizer(new_params, params, optimizer)

            # split
            num_pts = params['means3D'].shape[0]
            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads
            if densify_dict['scale_clone_thresh'] == 'scene_radius':
                thresh = 0.01 * variables['scene_radius']
            else:
                thresh = torch.median(torch.max(torch.exp(params['log_scales']), dim=1).values)
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                         torch.max(torch.exp(params['log_scales']), dim=1).values > thresh).to(device)
            if to_split.sum():
                n = densify_dict['num_to_split_into'] - 1
                new_params = dict()
                for k, v in params.items():
                    if k not in ['cam_unnorm_rots', 'cam_trans']:
                        if len(v.shape) == 3:
                            new_params[k] = v[to_split].repeat(n, 1, 1)
                        elif len(v.shape) == 2:
                            new_params[k] = v[to_split].repeat(n, 1)
                        else:
                            new_params[k] = v[to_split].repeat(n)

                # split variables
                variables = split_vars(params, variables, to_split, n)
                # update means and scales of new
                stds = torch.exp(params['log_scales'])[to_split].repeat(n, 3)
                means = torch.zeros((stds.size(0), 3), device="cuda")
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(params['unnorm_rotations'][to_split][:, :, time_idx]).repeat(n, 1, 1)
                new_params['means3D'][:, :, time_idx] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
                new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n+1))
                # update means and scales of prev
                stds = torch.exp(params['log_scales'])[to_split].repeat(1, 3)
                means = torch.zeros((stds.size(0), 3), device="cuda")
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(params['unnorm_rotations'][to_split][:, :, time_idx])
                params['means3D'][to_split][:, :, time_idx] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
                params['log_scales'][to_split] = torch.log(torch.exp(params['log_scales'][to_split]) / (0.8 * n+1))
                # cat new and prev
                params = cat_params_to_optimizer(new_params, params, optimizer)

            num_pts = params['means3D'].shape[0]
            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            
            # if to_split.sum():
            #     to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            #     params, variables = remove_points(to_remove, params, variables, optimizer)

            if iter == densify_dict['stop_after']:
                remove_threshold = densify_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = densify_dict['removal_opacity_threshold']

            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if iter >= densify_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            if to_remove.sum():
                params, variables = remove_points(to_remove, params, variables, optimizer)
            torch.cuda.empty_cache()

            print(f'Added {to_clone.sum()} Gaussians by cloning and {to_split.sum()} Gaussians by splitting during densification at Iteration {iter}!')
            print(f'Removed {to_remove.sum()} big Gaussians during densification at Iteration {iter}!')

        # Reset Opacities for all Gaussians (This is not desired for mapping on only current frame)
        if iter > 0 and iter % densify_dict['reset_opacities_every'] == 0 and densify_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
        
    return params, variables


def update_learning_rate(optimizer, means3D_scheduler, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in optimizer.param_groups:
            if param_group["name"] == "means3D":
                lr = means3D_scheduler(iteration)
                param_group['lr'] = lr
                return lr


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper