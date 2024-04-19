import torch
import torch.nn.functional as F
from utils.slam_external import build_rotation, normalize_quat
from utils.two2threeD_helpers import three2two
from diff_gaussian_rasterization import GaussianRasterizer as Renderer


def l1_loss_v1(x, y, mask=None, reduction='mean'):
    l1 = torch.abs((x - y))
    if mask is not None:
        l1 = l1[mask]
    if reduction == 'mean':
        return l1.mean()
    else:
        return l1.sum()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()

def l2_loss_v2(x, y, mask=None, reduction='mean'):
    if mask is not None:
        loss = torch.sqrt(((x - y) ** 2) + 1e-20)[mask]
    else:
        loss = torch.sqrt(((x - y) ** 2) + 1e-20)
    if reduction == 'mean':
        return loss.mean()
    else:
        return loss.sum()


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
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
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
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
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
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
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
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = instseg.squeeze()
    
    return depth_silhouette


def get_instsegmoving(tensor_shape, instseg, moving):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth and Silhouette
    depth_silhouette = torch.zeros((tensor_shape, 3)).cuda().float()
    depth_silhouette[:, 0] = 1.0
    depth_silhouette[:, 1] = moving.float().squeeze()
    depth_silhouette[:, 2] = instseg.squeeze()
    return depth_silhouette


def get_2D_motion(tensor_shape, gauss_flow):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth and Silhouette
    motion2d = torch.zeros((tensor_shape, 3)).cuda().float()
    motion2d[:, 0] = gauss_flow[:, 0]
    motion2d[:, 1] = gauss_flow[:, 1]
    motion2d[:, 2] = 1.0
    return motion2d


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
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def transformed_params2rendervar(params, transformed_gaussians, time_idx, first_occurance):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    rendervar = {
        'means3D': transformed_gaussians['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(transformed_gaussians['means3D'], requires_grad=True, device="cuda") + 0
    }
    rendervar, time_mask =  mask_timestamp(rendervar, time_idx, first_occurance, moving_mask=None)
    return rendervar, time_mask


def mask_timestamp(rendervar, timestamp, first_occurance, moving_mask=None, strictly_less=False):
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


def transformed_params2silhouette(params, transformed_gaussians):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    sil_color = torch.zeros_like(params['rgb_colors'])
    sil_color[:, 0] = 1.0
    rendervar = {
        'means3D': transformed_gaussians['means3D'],
        'colors_precomp': sil_color,
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def transformed_params2depthplussilhouette(params, w2c, transformed_gaussians, time_idx, first_occurance):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    rendervar = {
        'means3D': transformed_gaussians['means3D'],
        'colors_precomp': get_depth_and_silhouette(transformed_gaussians['means3D'], w2c),
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(transformed_gaussians['means3D'], requires_grad=True, device="cuda") + 0
    }
    rendervar, time_mask = mask_timestamp(rendervar, time_idx, first_occurance)
    return rendervar, time_mask


def transformed_params2instsegmov(
        params,
        transformed_gaussians,
        time_idx,
        variables,
        moving):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    rendervar = {
        'means3D': transformed_gaussians['means3D'],
        'colors_precomp': get_instsegmoving(transformed_gaussians['means3D'].shape[0], params['instseg'], moving),
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(transformed_gaussians['means3D'], requires_grad=True, device="cuda") + 0,
    }
    rendervar, time_mask = mask_timestamp(rendervar, time_idx, variables['timestep'])
    return rendervar, time_mask

def transformed_params2dmotion(
        params,
        transformed_gaussians,
        time_idx,
        variables,
        means2d,
        prev_means2d):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']

    gauss_flow = means2d - prev_means2d

    # Initialize Render Variables
    rendervar = {
        'means3D': transformed_gaussians['means3D'],
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(transformed_gaussians['means3D'], requires_grad=True, device="cuda") + 0,
        'colors_precomp': get_2D_motion(gauss_flow.shape[0], gauss_flow)
    }
    rendervar, time_mask = mask_timestamp(rendervar, time_idx, variables['timestep'], strictly_less=True)

    return rendervar, time_mask

def transformed_params2depthsilinstseg(
        params,
        w2c,
        transformed_gaussians,
        time_idx,
        first_occurance):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    rendervar = {
        'means3D': transformed_gaussians['means3D'],
        'colors_precomp': get_depth_and_silhouette_and_instseg(transformed_gaussians['means3D'], w2c, params['instseg']),
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(transformed_gaussians['means3D'], requires_grad=True, device="cuda") + 0,
    }
    rendervar, time_mask = mask_timestamp(rendervar, time_idx, first_occurance)
    return rendervar, time_mask


def transform_to_frame(
        params,
        time_idx,
        gaussians_grad,
        camera_grad=False,
        delta=0):
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
    # Get Frame Camera Pose
    if camera_grad:
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]
    else:
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        cam_tran = params['cam_trans'][..., time_idx].detach()
    rel_w2c = torch.eye(4).cuda().float()
    rel_w2c[:3, :3] = build_rotation(cam_rot)
    rel_w2c[:3, 3] = cam_tran

    # Check if Gaussians need to be rotated (Isotropic or Anisotropic)
    if params['log_scales'].shape[1] == 1:
        transform_rots = False # Isotropic Gaussians
    else:
        transform_rots = True # Anisotropic Gaussians
    
    # Get Centers and Unnorm Rots of Gaussians in World Frame
    if gaussians_grad:
        pts = params['means3D'][:, :, time_idx]
        unnorm_rots = params['unnorm_rotations'][:, :, time_idx]
    else:
        pts = params['means3D'][:, :, time_idx].detach()
        unnorm_rots = params['unnorm_rotations'][:, :, time_idx].detach()

    transformed_gaussians = {}
    # Transform Centers of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
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
    

def get_hook(should_be_disabled):
    def hook(grad):
        grad = grad.clone() # NEVER change the given grad inplace
        # Assumes 1D but can be generalized
        grad[should_be_disabled, :] = 0
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
        update_iso=False):
    losses = dict()

    # get relative rotation
    other_rot = params["unnorm_rotations"][:, :, iter_time_idx - 1].clone().detach()
    other_rot[:, 1:] = -1 * other_rot[:, 1:]
    other_means = params["means3D"][:, :, iter_time_idx - 1].clone().detach()
    curr_rot = transformed_gaussians["unnorm_rotations"]
    rel_rot = quat_mult(curr_rot, other_rot)
    rel_rot_mat = build_rotation(rel_rot)

    # Force same segment to have similar rotation and translation
    # mean_rel_rot_seg = scatter_mean(rel_rot, instseg_mask.squeeze(), dim=0)
    losses['rot'] = weighted_l2_loss_v2(
        rel_rot[variables["neighbor_indices"]],
        rel_rot[variables["self_indices"]],
        variables["neighbor_weight"])

    # rigid body
    curr_means = transformed_gaussians["means3D"] # params["means3D"][:, :, iter_time_idx]
    offset = curr_means[variables["self_indices"]] - curr_means[variables["neighbor_indices"]]
    offset_other_coord = (rel_rot_mat[variables["self_indices"]].transpose(2, 1) @ offset.unsqueeze(-1)).squeeze(-1)
    other_offset = other_means[variables["self_indices"]] - other_means[variables["neighbor_indices"]]
    losses['rigid'] = l2_loss_v2(offset_other_coord, other_offset)
    
    # store offset_0 and compute isometry
    if use_iso:
        offset = curr_means[variables["self_indices"]] - curr_means[variables["neighbor_indices"]]
        if iter == 0 and update_iso:
            if iter_time_idx == 1:
                offset_0 = offset.clone().detach()
            else:
                offset_0 = torch.cat(
                    [offset_0,
                    offset[variables['timestep'][
                        variables["self_indices"]] == iter_time_idx].clone().detach()])
        losses['iso'] = l2_loss_v2(offset, offset_0)

    return losses, offset_0


def get_renderings(
        params,
        variables,
        iter_time_idx,
        data,
        config, 
        disable_grads=False,
        track_cam=False,
        mov_thresh=0.01,
        prev_means2d=None,
        get_rgb=True,
        get_depth=True,
        get_motion=False,
        get_seg=False):

    transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                        gaussians_grad=True if not disable_grads and not track_cam else False,
                                        camera_grad=track_cam)
    
    # project means to 2d for flow
    means2d = three2two(
            data['cam'].projmatrix.squeeze(),
            transformed_gaussians['means3D'],
            data['cam'].image_width,
            data['cam'].image_height)
    
    if get_rgb:
        # RGB Rendering
        rendervar, time_mask = transformed_params2rendervar(
            params,
            transformed_gaussians,
            iter_time_idx,
            first_occurance=variables['timestep'])
        if not disable_grads:
            rendervar['means2D'].retain_grad()
        im, radius, _, weight, visible = Renderer(raster_settings=data['cam'])(**rendervar) 
        variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
    else:
        im, radius, weight, visible = None, None, None, None
    
    if get_depth:
        # Depth & Silhouette Rendering
        depth_sil_rendervar, _ = transformed_params2depthplussilhouette(
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
        seg_rendervar, _ = transformed_params2instsegmov(
            params,
            transformed_gaussians,
            iter_time_idx,
            variables,
            variables['moving'] > mov_thresh)
        instseg, _, _, _, _ = Renderer(raster_settings=data['cam'])(**seg_rendervar)
        # instseg 
        moving = instseg[1, :, :].unsqueeze(0)
        instseg = instseg[2, :, :]
    else:
        instseg, moving = None, None

    if get_motion and prev_means2d is not None:
        # render motion
        prev_transformed_gaussians = transform_to_frame(params, iter_time_idx-1,
                                            gaussians_grad=False)
        mot_rendervar, _ = transformed_params2dmotion(
            params,
            prev_transformed_gaussians,
            iter_time_idx-1,
            variables,
            means2d,
            prev_means2d)
        motion2d, _, _, _, _ = Renderer(raster_settings=data['cam'])(**mot_rendervar)
    else:
        motion2d = None

    return variables, im, radius, depth, instseg, mask, transformed_gaussians, means2d, visible, weight, motion2d, time_mask, moving, silhouette
