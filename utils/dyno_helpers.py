from lapsolver import solve_dense
import torch


def intersect_and_union(pred1: torch.tensor, pred2: torch.tensor):
    """Calculate Intersection and Union.

    Args:
        pred1 (torch.tensor): Prediction segmentation map
            or predict result filename. The shape is (H, W).
        pred2 (torch.tensor): Ground truth segmentation map
            or pred2 filename. The shape is (H, W).
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.

    Returns:
        torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
        torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
        torch.Tensor: The prediction histogram on all classes.
        torch.Tensor: The ground truth histogram on all classes.
    """

    num_p1 = torch.unique(pred1).shape[0]
    num_p2 = torch.unique(pred2).shape[0]
    intersect = torch.zeros(num_p1, num_p2)
    union = torch.zeros(num_p1, num_p2)
    area1 = torch.zeros(num_p1)
    area2 = torch.zeros(num_p2)
    
    for i, p1 in enumerate(torch.unique(pred1)):
        for j, p2 in enumerate(torch.unique(pred2)):
            intersect[i, j] = ((pred1 == p1) & (pred2 == p2)).sum()
            area1[i] = (pred1 == p1).sum()
            area2[j] = (pred2 == p2).sum()
            union[i, j] = ((pred1 == p1) | (pred2 == p2)).sum()

    return intersect, union, area1, area2

def get_greedy_assignment(iou_dist, seg_ids):
    # greedy
    col = list()
    assignments = torch.zeros(iou_dist.shape[1])
    count = 1
    for c in range(iou_dist.shape[1]):
        r = torch.argmin(iou_dist[:, c])
        if iou_dist[r, c] < 0.8:
            col.append(c)
            assignments[c] = seg_ids[r]
        else:
            assignments[c] = torch.max(seg_ids) + count
            count += 1
        
    return assignments

def get_hungarian_assignment(iou_dist, seg_ids):
    assignments = torch.zeros(iou_dist.shape[1])
    iou_dist[iou_dist > 0.8] = torch.nan
    row, col = solve_dense(iou_dist)
    row, col = row.tolist(), col.tolist()
    count = 1
    for c in range(iou_dist.shape[1]):
        r = torch.argmin(iou_dist[:, c])
        if c in col:
            assignments[c] = seg_ids[row[col.index(c)]]
        else:
            assignments[c] = torch.max(seg_ids) + count
            count += 1

def get_assignments(pred1, pred2, method='greedy'):
    seg_ids = torch.unique(pred1)
    intersect, union, _, _ = intersect_and_union(pred1, pred2)
    iou_dist = intersect/union
    if method == 'greedy':
        assignments = get_greedy_assignment(iou_dist, seg_ids)
    else:
        assignments = get_hungarian_assignment(iou_dist, seg_ids)
    pred_new = torch.zeros_like(pred2)
    seg_ids_pred2 = torch.unique(pred2)
    for a, s in zip(assignments, seg_ids_pred2):
        pred_new[pred2 == s] = a

    return pred_new