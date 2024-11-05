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
from torch_scatter import scatter_add
import torch.nn.functional as F


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


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


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
        time_idx=0,
        means2d=None):
    to_keep = ~to_remove
    idxs_to_keep = torch.arange(params['means3D'].shape[0])[to_keep].to(
        variables['self_indices'].device)
    to_keep_idx_mask = torch.isin(variables['self_indices'], idxs_to_keep) 
    to_keep_idx_mask = to_keep_idx_mask & torch.isin(variables['neighbor_indices'], idxs_to_keep)
    prev_time = variables['timestep'] < time_idx

    num_gaussians = params['means3D'].shape[0]
    num_gaussians_prev = params['means3D'][prev_time].shape[0]

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

    for k, v in variables.items():
        if k in ['scene_radius', 'last_time_idx'] or v is None:
            continue
        # mask variables
        if k == 'offset_0' and variables['offset_0'] is not None:
            variables['offset_0'] = variables['offset_0'][to_keep_idx_mask]
            variables[k] = v[to_keep_idx_mask].contiguous()
        elif v.shape[0] == num_gaussians:
            variables[k] = v[to_keep].contiguous()
        elif v.shape[0] == num_gaussians_prev:
            variables[k] = v[to_keep[prev_time]].contiguous()
        elif v.shape[0] == to_keep_idx_mask.shape[0]:
            variables[k] = v[to_keep_idx_mask].contiguous()

    mapping_tensor = torch.zeros(idxs_to_keep.max().item() + 1).long().to(idxs_to_keep.device)
    mapping_tensor[idxs_to_keep] = torch.arange(idxs_to_keep.shape[0]).long().to(idxs_to_keep.device)

    variables['neighbor_indices'] = mapping_tensor[variables['neighbor_indices']]
    variables['self_indices'] = mapping_tensor[variables['self_indices']]

    means2d = maybe_remove(means2d, to_keep)

    return params, variables, means2d


def maybe_remove(tensor, to_keep_mask):
    if tensor is not None:
        return tensor[to_keep_mask]
    else:
        return tensor


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def prune_gaussians(
        params,
        variables,
        optimizer,
        iter,
        prune_dict,
        curr_time_idx,
        means2d):
    variables['means2D_grad'] = variables['means2D'].grad
    if iter <= prune_dict['stop_after']:
        if (iter >= prune_dict['start_after']) and (iter % prune_dict['prune_every'] == 0):
            if iter == prune_dict['stop_after']:
                remove_threshold = prune_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = prune_dict['removal_opacity_threshold']
            # Remove Gaussians with low opacity
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            opa_remove_sum = to_remove.sum()

            # Remove Gaussians with large kNN dist
            # recompute kNN distance
            with torch.no_grad():
                pdist = torch.nn.PairwiseDistance(p=2)
                offset_t_mag = pdist(
                    params['means3D'][variables["self_indices"], :, curr_time_idx-1],
                    params['means3D'][variables["neighbor_indices"], :, curr_time_idx-1])
                offset_t_mag = scatter_add(offset_t_mag, variables["self_indices"], dim=0)
                
                offset_0_mag = torch.linalg.norm(variables['offset_0'], dim=1)
                offset_0_mag = scatter_add(offset_0_mag, variables["self_indices"], dim=0)

                rel_drift = torch.abs(offset_t_mag-offset_0_mag) / offset_0_mag
                drift = rel_drift > prune_dict['kNN_rel_drift']
                to_remove[variables['timestep']<curr_time_idx] = torch.logical_or(
                    to_remove[variables['timestep']<curr_time_idx],
                    drift)

            # Remove Gaussians that are too big
            if iter >= prune_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            if to_remove.sum():
                params, variables, means2d = remove_points(
                    to_remove,
                    params,
                    variables,
                    optimizer,
                    time_idx=curr_time_idx,
                    means2d=means2d
                    )
            torch.cuda.empty_cache()
            print(f'Removed {to_remove.sum()} Gaussians during pruning at Iteration {iter} - {opa_remove_sum} by opacity, {drift.sum()} because of drift, {big_points_ws.sum()} because of scale!')
        
        # Reset Opacities for all Gaussians
        if iter > 0 and iter % prune_dict['reset_opacities_every'] == 0 and prune_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
    
    return params, variables, means2d

def clone_vars(params, variables, to_clone):
    device = variables['self_indices'].device
    idxs_to_clone = torch.arange(params['means3D'].shape[0])[to_clone].to(device)
    to_clone = to_clone.to(device)
    to_clone_self_idx = torch.isin(variables['self_indices'], idxs_to_clone)
    variables['neighbor_indices'] = torch.cat(
        (variables['neighbor_indices'], variables['neighbor_indices'][to_clone_self_idx]), dim=0)
    variables['neighbor_weight'] = torch.cat(
        (variables['neighbor_weight'], variables['neighbor_weight'][to_clone_self_idx]), dim=0)
    variables['neighbor_weight_sm'] = torch.cat(
        (variables['neighbor_weight_sm'], variables['neighbor_weight_sm'][to_clone_self_idx]), dim=0)
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
    variables['neighbor_weight_sm'] = torch.cat(
        (variables['neighbor_weight_sm'], variables['neighbor_weight_sm'][to_split_self_idx].repeat(n)), dim=0)
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


def densify(params, variables, optimizer, iter, densify_dict, time_idx, means2d):
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
                params, variables, means2d = remove_points(to_remove, params, variables, optimizer, time_idx=time_idx, means2d=means2d)
            torch.cuda.empty_cache()

            print(f'Added {to_clone.sum()} Gaussians by cloning and {to_split.sum()} Gaussians by splitting during densification at Iteration {iter}!')
            print(f'Removed {to_remove.sum()} big Gaussians during densification at Iteration {iter}!')

        # Reset Opacities for all Gaussians (This is not desired for mapping on only current frame)
        if iter > 0 and iter % densify_dict['reset_opacities_every'] == 0 and densify_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
        
    return params, variables, means2d


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def three2two(proj_matrix, means3D, w, h, do_round=False, do_normalize=False):
    points_xy = proj_matrix.T.matmul(torch.cat(
            [means3D, torch.ones(means3D.shape[0], 1).to(means3D.device)], dim=1).T)
    points_xy = points_xy / points_xy[3, :]
    points_xy = points_xy[:2].T
    # points_xy[:, 0] = ((points_xy[:, 0] + 1) * w - 1) * 0.5
    # points_xy[:, 1] = ((points_xy[:, 1] + 1) * h - 1) * 0.5
    points_xy[:, 0] = ((points_xy[:, 0] + 1) * w) * 0.5
    points_xy[:, 1] = ((points_xy[:, 1] + 1) * h) * 0.5

    if do_normalize:
        points_xy = normalize_points(points_xy, h, w)
    elif do_round:
        points_xy = torch.round(points_xy).long()

    return points_xy


def normalize_points(points_xy, h, w):

    if len(points_xy.shape) == 2:
        points_xy[:, 0] = (points_xy[:, 0])/w
        points_xy[:, 1] = (points_xy[:, 1])/h
    elif len(points_xy.shape) == 3:
        points_xy[:, :, 0] = (points_xy[:, :, 0])/w
        points_xy[:, :, 1] = (points_xy[:, :, 1])/h
    elif len(points_xy.shape) == 4:
        points_xy[:, :, :, 0] = (points_xy[:, :, :, 0])/w
        points_xy[:, :, :, 1] = (points_xy[:, :, :, 1])/h
    elif len(points_xy.shape) == 5:
        points_xy[:, :, :, :, 0] = (points_xy[:, :, :, :, 0])/w
        points_xy[:, :, :, :, 1] = (points_xy[:, :, :, :, 1])/h
    else:
        print('Not implemented to normalize this')
        quit()
    return points_xy


def unnormalize_points(points_xy, h, w, do_round=False):
    """
    points_xy: NxTimesx2 or Nx2 or Timesx2
    """
    
    if len(points_xy.shape) == 2:
        points_xy[:, 0] =  (points_xy[:, 0] * w)
        points_xy[:, 1] =  (points_xy[:, 1] * h)
    elif len(points_xy.shape) == 3:
        points_xy[:, :, 0] =  (points_xy[:, :, 0] * w)
        points_xy[:, :, 1] =  (points_xy[:, :, 1] * h)
    elif len(points_xy.shape) == 4:
        points_xy[:, :, :, 0] =  (points_xy[:, :, :, 0] * w)
        points_xy[:, :, :, 1] =  (points_xy[:, :, :, 1] * h)
    else:
        print('Not implemented to unnormalize this')
        quit()

    if do_round:
        points_xy = torch.round(points_xy).long()

    return points_xy