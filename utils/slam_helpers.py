import torch
import torch.nn.functional as F
from utils.slam_external import build_rotation, normalize_quat, matrix_to_quaternion
from utils.two2threeD_helpers import three2two
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization_embeddings import GaussianRasterizerEmb as EmbeddingRenderer
from torch_scatter import scatter_mean, scatter_add



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


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


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


def params2rendervar(params):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=params['means3D'].device) + 0
    }
    return rendervar


def project_points(points_3d, intrinsics):
    """
    Function to project 3D points to image plane.
    params:
    points_3d: [num_gaussians, 3]
    intrinsics: [3, 3]
    out: [num_gaussians, 2]
    """
    points_2d = torch.matmul(intrinsics, points_3d.transpose(0, 1))
    points_2d = points_2d.transpose(0, 1)
    points_2d = points_2d / points_2d[:, 2:]
    points_2d = points_2d[:, :2]
    return points_2d


def params2silhouette(params):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    sil_color = torch.zeros_like(params['rgb_colors'])
    sil_color[:, 0] = 1.0
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': sil_color,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=params['means3D'].device) + 0
    }
    return rendervar


def get_depth_and_silhouette(pts_3D, w2c):
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


def get_depth_and_silhouette_and_instseg(pts_3D, w2c, instseg):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3), device=pts_3D.device).float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = instseg.squeeze()
    
    return depth_silhouette


def get_instsegbg(tensor_shape, instseg, bg):
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


def params2depthplussilhouette(params, w2c):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': get_depth_and_silhouette(params['means3D'], w2c),
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=params['means3D'].device) + 0
    }
    return rendervar


def transformed_params2rendervar(params, transformed_gaussians, time_idx, first_occurance, time_window=1, active_gaussians_mask=None, depth=None):
    # Check if Gaussians are Isotropic
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
    rendervar, time_mask =  mask_timestamp(rendervar, time_idx+time_window-1, first_occurance, active_gaussians_mask=active_gaussians_mask, depth=depth)
    return rendervar, time_mask


def mask_timestamp(rendervar, timestamp, first_occurance, moving_mask=None, strictly_less=False, active_gaussians_mask=None, depth=None):
    if strictly_less:
        time_mask = first_occurance < timestamp
    else:
        time_mask = first_occurance <= timestamp

    if moving_mask is not None:
        time_mask = time_mask & moving_mask

    if active_gaussians_mask is not None:
        time_mask = time_mask & active_gaussians_mask

    if depth is not None:
        # print('num_mask gauss', time_mask.sum() - (time_mask & (rendervar['means3D'][:, 2] > 0.9 * depth.min())).sum())
        time_mask = time_mask & (rendervar['means3D'][:, 2] > min(0.2, 0.2 * depth.min().item()))

    masked_rendervar = dict()
    for k, v in rendervar.items():
        masked_rendervar[k] = v[time_mask]
    return masked_rendervar, time_mask


def transformed_params2depthplussilhouette(params, w2c, transformed_gaussians, time_idx, first_occurance, time_window=1, active_gaussians_mask=None, depth=None):
    # Check if Gaussians are Isotropic
    if len(params['log_scales'].squeeze().shape) == 1:
        log_scales = params['log_scales'] 
    elif len(params['log_scales'].squeeze().shape) == 2 and params['log_scales'].squeeze().shape[1] == 3:
        log_scales = params['log_scales']
    elif len(params['log_scales'].squeeze().shape) == 2 : 
        log_scales = params['log_scales'][:, time_idx].unsqueeze(1)
    else:   
        log_scales = params['log_scales'][:, :, time_idx]

    if log_scales.shape[1] == 1:
        log_scales = torch.tile(log_scales, (1, 3))
    else:
        log_scales = log_scales
    
    # Initialize Render Variables
    rendervar = {
        'means3D': transformed_gaussians['means3D'],
        'colors_precomp': get_depth_and_silhouette(transformed_gaussians['means3D'], w2c),
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(transformed_gaussians['means3D'], requires_grad=True, device=params['means3D'].device) + 0
    }
    rendervar, time_mask = mask_timestamp(rendervar, time_idx+time_window-1, first_occurance, active_gaussians_mask, depth=depth)
    return rendervar, time_mask


def transformed_params2instsegbg(
        params,
        transformed_gaussians,
        time_idx,
        variables,
        time_window=1,
        active_gaussians_mask=None,
        depth=None):
    # Check if Gaussians are Isotropic
    if len(params['log_scales'].squeeze().shape) == 1:
        log_scales = params['log_scales'] 
    elif len(params['log_scales'].squeeze().shape) == 2 and params['log_scales'].squeeze().shape[1] == 3:
        log_scales = params['log_scales']
    elif len(params['log_scales'].squeeze().shape) == 2 : 
        log_scales = params['log_scales'][:, time_idx].unsqueeze(1)
    else:
        log_scales = params['log_scales'][:, :, time_idx]

    if log_scales.shape[1] == 1:
        log_scales = torch.tile(log_scales, (1, 3))
    else:
        log_scales = log_scales

    # Initialize Render Variables
    rendervar = {
        'means3D': transformed_gaussians['means3D'],
        'colors_precomp': get_instsegbg(transformed_gaussians['means3D'].shape[0], params['instseg'], params['bg']),
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(transformed_gaussians['means3D'], requires_grad=True, device=params['means3D'].device) + 0,
    }
    rendervar, time_mask = mask_timestamp(rendervar, time_idx+time_window-1, variables['timestep'], active_gaussians_mask=active_gaussians_mask, depth=depth)
    return rendervar, time_mask


def transformed_params2emb(
        params,
        transformed_gaussians,
        time_idx,
        variables,
        emb_idx,
        max_idx,
        time_window=1,
        active_gaussians_mask=None,
        depth=None):
    # Check if Gaussians are Isotropic
    if len(params['log_scales'].squeeze().shape) == 1:
        log_scales = params['log_scales'] 
    elif len(params['log_scales'].squeeze().shape) == 2 and params['log_scales'].squeeze().shape[1] == 3:
        log_scales = params['log_scales']
    elif len(params['log_scales'].squeeze().shape) == 2 : 
        log_scales = params['log_scales'][:, time_idx].unsqueeze(1)
    else:
        log_scales = params['log_scales'][:, :, time_idx]

    if log_scales.shape[1] == 1:
        log_scales = torch.tile(log_scales, (1, 3))
    else:
        log_scales = log_scales

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

    rendervar, time_mask = mask_timestamp(rendervar, time_idx+time_window-1, variables['timestep'], active_gaussians_mask=active_gaussians_mask, depth=depth)

    return rendervar, time_mask


def transform_to_frame(
        params,
        time_idx,
        gaussians_grad,
        camera_grad=False,
        gauss_time_idx=None,
        delta=0,
        motion_mlp=None,
        base_transformations=None,
        variables=None,
        gt_w2c=None):
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

    coefficients = None
    if gauss_time_idx is None:
        gauss_time_idx = time_idx

    # Get Frame Camera Pose
    if gt_w2c is None:
        if camera_grad:
            cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx])
            cam_tran = params['cam_trans'][..., time_idx]
        else:
            cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
            cam_tran = params['cam_trans'][..., time_idx].detach()
        rel_w2c = torch.eye(4, device=params['means3D'].device).float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
    else:
        with torch.no_grad():
            # Get the ground truth pose relative to frame 0
            rel_w2c = gt_w2c
            rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
            rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
            rel_w2c_tran = rel_w2c[:3, 3].detach()
            # Update the camera parameters
            params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
            params['cam_trans'][..., time_idx] = rel_w2c_tran

    # Check if Gaussians need to be rotated (Isotropic or Anisotropic)
    if params['log_scales'].shape[1] == 1:
        transform_rots = False # Isotropic Gaussians
    else:
        transform_rots = True # Anisotropic Gaussians
    
    # Get Centers and Unnorm Rots of Gaussians in World Frame
    if motion_mlp is None and base_transformations is None:
        if all_times:
            pts = params['means3D'][:, :, gauss_time_idx]
            unnorm_rots = params['unnorm_rotations'][:, :, gauss_time_idx]
        else:
            pts = params['means3D']
            unnorm_rots = params['unnorm_rotations']
        if not gaussians_grad:
            pts = pts.detach()
            unnorm_rots = unnorm_rots.detach()
    else:
        pts = variables['means3D'][:, :, gauss_time_idx-1].detach()
        unnorm_rots = variables['unnorm_rotations'][:, :, gauss_time_idx-1].detach()
        if motion_mlp is not None:
            pts = pts + motion_mlp(pts, gauss_time_idx-1).squeeze()
        if base_transformations is not None:
            pts, unnorm_rots, coefficients  = base_transformations(pts, unnorm_rots, time_idx=time_idx)

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

    return transformed_gaussians, coefficients


def get_smallest_axis(params, iter_time_idx, return_idx=False):
    """Returns the smallest axis of the Gaussians.

    Args:
        return_idx (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    rotation_matrices = build_rotation(params['unnorm_rotations'][:, :, iter_time_idx])
    smallest_axis_idx = params['log_scales'].min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
    smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
    if return_idx:
        return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
    return smallest_axis.squeeze(dim=2)
    

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

def get_hook_bg(should_be_shrinked, grad_weight):
    def hook(grad):
        # grad = grad.clone() # NEVER change the given grad inplace
        # Assumes 1D but can be generalized
        grad[should_be_shrinked, :] = 0
        return grad
    return hook


def dyno_losses(
        params,
        iter_time_idx,
        transformed_gaussians,
        variables,
        offset_0,
        iter,
        use_iso,
        update_iso=False,
        time_window=1,
        weight='bg',
        post_init=True,
        mag_iso=False,
        weight_rot=True,
        weight_rigid=True,
        weight_iso=True,
        last_x=1):
    losses = dict()
    if weight == 'bg':
        bg_weight = 1 - torch.abs(params['bg'][variables["self_indices"]].detach().clone() - params['bg'][variables["neighbor_indices"]].detach().clone())
    elif weight == 'none':
        bg_weight = params['bg'][variables["self_indices"]].detach().clone()*0+1
    else:
        bg_weight = variables["neighbor_weight"].unsqueeze(1)
    
    all_times = len(params["unnorm_rotations"].shape) == 3
    if all_times:
        prev_params = params
        curr_params = params[:, :, iter_time_idx]
    else:
        prev_params = variables
        curr_params = params

    # compute loss for last x timesteps
    loss_rigid = 0
    for i in range(1, min(iter_time_idx, last_x)+1):
        # get relative rotation
        other_rot = prev_params["unnorm_rotations"][:, :, iter_time_idx - i].detach().clone()
        other_rot[:, 1:] = -1 * other_rot[:, 1:]
        other_means = prev_params["means3D"][:, :, iter_time_idx - i].detach().clone()
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
            weight=bg_weight if weight_rigid else None)

    losses['rigid'] = loss_rigid
    losses['rot'] = l2_loss_v2(
        rel_rot[variables["neighbor_indices"]],
        rel_rot[variables["self_indices"]],
        weight=bg_weight if weight_rot else None)

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
                           variables["self_indices"]] == iter_time_idx+time_window-1].detach().clone()])
                else:
                    offset_0 = torch.cat(
                      [offset_0,
                      offset[variables['timestep'][
                           variables["self_indices"]] == iter_time_idx+time_window-2].detach().clone()])
        if mag_iso:
            losses['iso'] = l2_loss_v2(
                torch.sqrt((offset ** 2).sum(-1) + 1e-20),
                torch.sqrt((offset_0 ** 2).sum(-1) + 1e-20),
                weight=bg_weight.squeeze() if weight_iso else None)
        else:
            losses['iso'] = l2_loss_v2(
                offset,
                offset_0,
                weight=bg_weight.squeeze() if weight_iso else None)

    return losses, offset_0


def compute_visibility(visible, weight, visibility_modus='thresh', get_norm_pix_pos=False, thresh=0.5, num_gauss=0):
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

    weighted_norm_x, weighted_norm_y = get_weighted_pix_pos(
        params,
        w,
        h,
        pix_id,
        vis_pix_flat,
        weight_pix_flat,
        weight_sum_per_gauss)
    
    return visibility, weighted_norm_x, weighted_norm_y
    
def get_weighted_pix_pos(w, h, pix_id, vis_pix_flat, weight_pix_flat, weight_sum_per_gauss):
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


def get_renderings(
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
        time_window=1,
        delta=False,
        motion_mlp=None,
        base_transformations=None,
        do_compute_visibility=False,
        remove_close=False):

    transformed_gaussians, coefficients = transform_to_frame(params, iter_time_idx,
                                        gaussians_grad=True if not disable_grads and not track_cam and motion_mlp is None and base_transformations is None else False,
                                        camera_grad=track_cam,
                                        delta=delta,
                                        motion_mlp=motion_mlp,
                                        base_transformations=base_transformations,
                                        gt_w2c=variables['gt_w2c_all_frames'][iter_time_idx] if config['gt_w2c'] else None)

    if "to_deactivate" in variables.keys():
        active_gaussians_mask = variables["to_deactivate"] == 100000
    else:
        active_gaussians_mask = None

    if get_rgb:
        # RGB Rendering
        rendervar, time_mask = transformed_params2rendervar(
            params,
            transformed_gaussians,
            iter_time_idx,
            first_occurance=variables['timestep'],
            time_window=time_window,
            active_gaussians_mask=active_gaussians_mask,
            depth=data['depth'] if remove_close else None)

        if not disable_grads:
            rendervar['means2D'].retain_grad()
        im, radius, _, weight, visible = Renderer(raster_settings=data['cam'])(**rendervar) 
        variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
        if do_compute_visibility:
            visibility = compute_visibility(visible, weight, num_gauss=params['means3D'].shape[0])
        else:
            visibility = None
    else:
        im, radius, weight, visible, visibility = None, None, None, None, None

    if get_depth:
        # Depth & Silhouette Rendering
        depth_sil_rendervar, _ = transformed_params2depthplussilhouette(
            params,
            data['w2c'],
            transformed_gaussians,
            iter_time_idx,
            first_occurance=variables['timestep'],
            time_window=time_window,
            active_gaussians_mask=active_gaussians_mask,
            depth=data['depth'] if remove_close else None)
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
        seg_rendervar, _ = transformed_params2instsegbg(
            params,
            transformed_gaussians,
            iter_time_idx,
            variables,
            time_window=time_window,
            active_gaussians_mask=active_gaussians_mask,
            depth=data['depth'] if remove_close else None)
        instseg, _, _, _, _ = Renderer(raster_settings=data['cam'])(**seg_rendervar)
        # instseg 
        bg = instseg[0, :, :].unsqueeze(0)
        instseg = instseg[2, :, :]
    else:
        instseg, bg = None, None
    
    if get_embeddings:
        rendered_embeddings = list()
        for emb_idx in range(0, params['embeddings'].shape[1], 3):
            max_idx = min(params['embeddings'].shape[1]-emb_idx, 3)
            emb_rendervar, _ = transformed_params2emb(
                params,
                transformed_gaussians,
                iter_time_idx,
                variables,
                emb_idx,
                max_idx,
                time_window=time_window,
                active_gaussians_mask=active_gaussians_mask,
                depth=data['depth'] if remove_close else None)
            _embeddings, _, _, _, _ = Renderer(raster_settings=data['cam'])(**emb_rendervar)
            rendered_embeddings.append(_embeddings[:max_idx])
        rendered_embeddings = torch.cat(rendered_embeddings, dim=0)
    else:
        rendered_embeddings = None
    return variables, im, radius, depth, instseg, mask, transformed_gaussians, visible, weight, time_mask, None, silhouette, rendered_embeddings, bg, visibility, coefficients


def get_renderings_batched(
        params,
        variables,
        iter_time_idx,
        data,
        config, 
        disable_grads=False,
        track_cam=False,
        prev_means2d_list=None,
        get_rgb=True,
        get_depth=True,
        get_motion=False,
        get_seg=False,
        get_embeddings=True,
        time_window=1,
        delta=False,
        remove_close=False):
        
        batch_renderings = list()
        for i in range(len(data)):
            if prev_means2d_list is not None:
                prev_means2d = prev_means2d_list[i]
            else:
                prev_means2d = None
            # get renderings for current time
            renderings = \
                get_renderings(
                    params,
                    variables,
                    iter_time_idx+i,
                    data[i],
                    config,
                    disable_grads=disable_grads,
                    get_motion=get_motion,
                    prev_means2d=prev_means2d,
                    get_embeddings=get_embeddings,
                    time_window=time_window,
                    delta=i if delta else 0,
                    remove_close=remove_close)
            batch_renderings.append(renderings)
        return batch_renderings