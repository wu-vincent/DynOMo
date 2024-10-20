import torch
import torch.nn.functional as F
from utils.slam_external import build_rotation
from src.utils.slam_external import quat_mult, build_rotation
from src.utils.slam_external import calc_ssim


def l1_loss_v1(x, y, mask=None, reduction='mean', weight=None):
    l1 = torch.abs((x - y))
    if weight is not None:
        l1 = l1 * weight
    if mask is not None:
        l1 = l1[mask]
    if reduction == 'mean':
        return l1.mean()
    elif reduction == 'none':
        return l1
    else:
        return l1.sum()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def l2_loss_v2(x, y, mask=None, weight=None, reduction='mean'):
    l2 = torch.sqrt(((x - y) ** 2) + 1e-20)
    if weight is not None:
        l2 = l2 * weight
    if mask is not None:
        l2 = l2[mask]
    if reduction == 'mean':
        return l2.mean()
    else:
        return l2.sum()
    

def get_hook(should_be_disabled, grad_weight=0):
    def hook(grad):
        # grad = grad.clone() # NEVER change the given grad inplace
        # Assumes 1D but can be generalized
        if grad_weight == 0:
            grad[should_be_disabled, :] = 0
        else:
            grad[should_be_disabled, :] *= grad_weight
        return grad
    return hook


def physics_based_losses(
        params,
        iter_time_idx,
        transformed_gaussians,
        variables,
        offset_0,
        iter,
        use_iso,
        update_iso=False,
        post_init=True,
        device="cuda:0",
        losses=None):

    weight = variables["neighbor_weight"].unsqueeze(1)
    
    all_times = len(params["unnorm_rotations"].shape) == 3
    if all_times:
        prev_params = params
        curr_params = params[:, :, iter_time_idx]
    else:
        prev_params = variables
        curr_params = params

    # get relative rotation
    other_rot = prev_params["unnorm_rotations"][:, :, iter_time_idx - 1].detach().clone().to(device)
    other_rot[:, 1:] = -1 * other_rot[:, 1:]
    other_means = prev_params["means3D"][:, :, iter_time_idx - 1].detach().clone().to(device)
    curr_rot = curr_params["unnorm_rotations"]
    rel_rot = quat_mult(curr_rot, other_rot)
    rel_rot_mat = build_rotation(rel_rot)

    # rigid body
    curr_means = curr_params["means3D"]
    offset = curr_means[variables["self_indices"]] - curr_means[variables["neighbor_indices"]]
    offset_other_coord = (rel_rot_mat[variables["self_indices"]].transpose(2, 1) @ offset.unsqueeze(-1)).squeeze(-1)
    other_offset = other_means[variables["self_indices"]] - other_means[variables["neighbor_indices"]]
    loss_rigid = l2_loss_v2(
        offset_other_coord,
        other_offset,
        weight=weight)

    losses['rigid'] = loss_rigid
    losses['rot'] = l2_loss_v2(
        rel_rot[variables["neighbor_indices"]],
        rel_rot[variables["self_indices"]],
        weight=weight)

    # store offset_0 and compute isometry
    if use_iso:
        offset = curr_means[variables["self_indices"]] - curr_means[variables["neighbor_indices"]]
        if iter == 0 and update_iso:
            if iter_time_idx == 1:
                offset_0 = offset.detach().clone()
            else:
                if not post_init:
                    offset_0 = torch.cat(
                      [offset_0,
                      offset[variables['timestep'][
                           variables["self_indices"]] == iter_time_idx].detach().clone()])
                else:
                    offset_0 = torch.cat(
                      [offset_0,
                      offset[variables['timestep'][
                           variables["self_indices"]] == iter_time_idx+1].detach().clone()])
        losses['iso'] = l2_loss_v2(
            torch.sqrt((offset ** 2).sum(-1) + 1e-20),
            torch.sqrt((offset_0 ** 2).sum(-1) + 1e-20),
            weight=weight.squeeze())

    return losses, offset_0


def get_rendered_losses(config, losses, curr_data, im, depth, mask, embeddings, bg=None, load_embeddings=False, iter_time_idx=0, scene=None, device="cuda:0"):
    # RGB Loss
    if not config['calc_ssmi']:
        losses['im'] = l1_loss_v1(
            curr_data['im'].permute(1, 2, 0),
            im.permute(1, 2, 0),
            mask,
            reduction='mean')
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) \
            + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    # DEPTH LOSS
    if config['use_l1']:
        losses['depth'] = l1_loss_v1(
            curr_data['depth'],
            depth,
            mask,
            reduction='mean')

    # EMBEDDING LOSS
    if load_embeddings and config['loss_weights']['embeddings'] != 0:
        embeddings_gt = curr_data['embeddings']
        if config['norm_embeddings']:
            embeddings_gt = torch.nn.functional.normalize(
                curr_data['embeddings'], p=2, dim=0)
            embeddings = torch.nn.functional.normalize(
                embeddings, p=2, dim=0)

        losses['embeddings'] = l2_loss_v2(
            embeddings_gt.permute(1, 2, 0),
            embeddings.permute(1, 2, 0),
            mask.squeeze(),
            reduction='mean')
          
    # BG REG LOSS
    if config['bg_reg']:
        # hard foce bg
        if iter_time_idx > 0:
            is_bg = scene.params['bg'].detach().clone().squeeze() > 0.5
            losses['bg_reg'] = l1_loss_v1(
                scene.params['means3D'][:, :][is_bg],
                scene.variables['means3D'][:, :, iter_time_idx-1][is_bg].to(device))

        # bg loss with mask    
        losses['bg_loss'] = l1_loss_v1(
            bg.squeeze(),
            curr_data['bg'].float().squeeze(),
            mask=mask.squeeze())
    
    return losses


def get_l1_losses(losses, config, iter_time_idx, scene, load_embeddings):
    l1_mask = scene.variables['timestep'] < iter_time_idx
    if config['loss_weights']['l1_bg'] and iter_time_idx > 0:
        losses['l1_bg'] = l1_loss_v1(scene.params['bg'][l1_mask], scene.variables['prev_bg'])
    
    if config['loss_weights']['l1_rgb'] and iter_time_idx > 0:
        losses['l1_rgb'] = l1_loss_v1(scene.params['rgb_colors'][l1_mask], scene.variables['prev_rgb_colors'])
    
    if config['loss_weights']['l1_embeddings'] and iter_time_idx > 0 and load_embeddings:
        losses['l1_embeddings'] = l1_loss_v1(scene.params['embeddings'][l1_mask], scene.variables['prev_embeddings'])

    if config['loss_weights']['l1_scale'] != 0 and iter_time_idx > 0:
        losses['l1_scale'] = l1_loss_v1(scene.params['log_scales'][l1_mask], scene.variables['prev_log_scales'])

    if config['loss_weights']['l1_opacity'] != 0 and iter_time_idx > 0:
        losses['l1_opacity'] = l1_loss_v1(scene.params['logit_opacities'][l1_mask], scene.variables['prev_logit_opacities'])

    return losses
