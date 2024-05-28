import torch


def three2two(proj_matrix, means3D, w, h, do_round=False, do_normalize=False, do_scale=False):
    points_xy = proj_matrix.T.matmul(torch.cat(
            [means3D, torch.ones(means3D.shape[0], 1).to(means3D.device)], dim=1).T)
    points_xy = points_xy / points_xy[3, :]
    points_xy = points_xy[:2].T
    # points_xy[:, 0] = ((points_xy[:, 0] + 1) * w - 1) * 0.5
    # points_xy[:, 1] = ((points_xy[:, 1] + 1) * h - 1) * 0.5
    points_xy[:, 0] = ((points_xy[:, 0] + 1) * w) * 0.5
    points_xy[:, 1] = ((points_xy[:, 1] + 1) * h) * 0.5

    if do_scale:
        sc_pt = torch.tensor(
            [[w/256, h/256]]).float().to(means3D.device)
        points_xy = points_xy/sc_pt
        w = 256
        h = 256

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
    else:
        print('Not implemented to normalize this')
        quit()
    return points_xy


def unnormalize_points(points_xy, h, w, do_round=False, do_scale=False):
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
    
    if do_scale:
        sc_pt = torch.tensor(
            [[w/256, h/256]]).float().to(points_xy.device)
        points_xy = points_xy/sc_pt

    if do_round:
        points_xy = torch.round(points_xy).long()

    return points_xy
