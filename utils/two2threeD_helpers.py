import torch


def three2two(proj_matrix, means3D, w, h, do_round=False, do_normalize=False):
    points_xy = proj_matrix.T.matmul(torch.cat(
            [means3D, torch.ones(means3D.shape[0], 1).cuda()], dim=1).T)
    points_xy = points_xy / points_xy[3, :]
    points_xy = points_xy[:2].T

    points_xy[:, 0] = ((points_xy[:, 0] + 1) * w - 1) * 0.5
    points_xy[:, 1] = ((points_xy[:, 1] + 1) * h - 1) * 0.5

    if do_normalize:
        points_xy = normalize_points(points_xy, h, w)
    if do_round:
        points_xy = points_xy.long()

    return points_xy


def two2three(proj_matrix, means2D, depth, w, h, do_round=False, do_unnormalize=False):
    if do_unnormalize:
        points_xy = unnormalize_points(points_xy, h, w, do_round=do_round)
    points_xy = proj_matrix.T.matmul(torch.cat(
            [means3D, torch.ones(means3D.shape[0], 1).cuda()], dim=1).T)
    points_xy[:, 0] = (points_xy[:, 0] * 2 + 1) / w - 1
    points_xy[:, 1] = (points_xy[:, 1] * 2 + 1) / h - 1
    points_xy *= depth
    means3D = torch.cat([points_xy, depth])

    points_xyz = torch.inverse(proj_matrix).T.matmul(torch.cat(
            [means3D, torch.ones(means3D.shape[0], 1).cuda()], dim=1).T)
    print("NOOOOOT SURE!!!!")
    return points_xy


def normalize_points(points_xy, h, w):
    if len(points_xy.shape) == 2:
        points_xy[:, 0] = (points_xy[:, 0] + 1)/w
        points_xy[:, 1] = (points_xy[:, 1] + 1)/h
    else:
        points_xy[:, :, 0] = (points_xy[:, 0] + 1)/w
        points_xy[:, :, 1] = (points_xy[:, 1] + 1)/h
    return points_xy

def unnormalize_points(points_xy, h, w, do_round=False):
    """
    points_xy: NxTimesx2 or Nx2 or Timesx2
    """
    if len(points_xy.shape) == 2:
        points_xy[:, 0] =  (points_xy[:, 0]*w) - 1
        points_xy[:, 1] =  (points_xy[:, 1]*h) - 1
    else:
        points_xy[:, :, 0] =  (points_xy[:, :, 0]*w) - 1
        points_xy[:, :, 1] =  (points_xy[:, :, 1]*h) - 1
    if do_round:
        points_xy = points_xy.long()

    return points_xy
