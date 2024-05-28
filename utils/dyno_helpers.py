from lapsolver import solve_dense
import torch
import pytorch3d.loss
import sklearn.cluster
import numpy as np


def intersect_and_union2D(pred1: torch.tensor, pred2: torch.tensor):
    num_p1 = torch.unique(pred1).shape[0]
    num_p2 = torch.unique(pred2).shape[0]
    intersect = torch.zeros(num_p1, num_p2).to(pred1.device)
    union = torch.zeros(num_p1, num_p2).to(pred1.device)
    area1 = torch.zeros(num_p1).to(pred1.device)
    area2 = torch.zeros(num_p2).to(pred1.device)
    
    for i, p1 in enumerate(torch.unique(pred1)):
        for j, p2 in enumerate(torch.unique(pred2)):
            intersect[i, j] = ((pred1 == p1) & (pred2 == p2)).sum()
            area1[i] = (pred1 == p1).sum()
            area2[j] = (pred2 == p2).sum()
            union[i, j] = ((pred1 == p1) | (pred2 == p2)).sum()

    return intersect, union, area1, area2


def intersect_and_union3D(pred1: torch.tensor, pred2: torch.tensor):
    intersection = torch.sum(pred1 & pred2)
    union = torch.sum(pred1 | pred2)
    iou = intersection / union if union != 0 else 0

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

def chamfer_distance(pc1: torch.tensor, pc2: torch.tensor, instseg1, instseg2, filtered_instseg2):
    num_p1 = torch.unique(instseg1).shape[0]
    num_p2 = torch.unique(instseg2).shape[0]
    cd_dist = torch.ones(num_p1, num_p2) * 1000
    for i, p1 in enumerate(torch.unique(instseg1)):
        for j, p2 in enumerate(torch.unique(instseg2)):
            if p2 not in filtered_instseg2:
                continue
            cd_dist[i, j] = pytorch3d.loss.chamfer_distance(pc1[:, instseg1 == p1], pc2[:, instseg2 == p2], single_directional=False)[0]
    return cd_dist


def get_masks(pred):
    mask_dets = list()
    for p in torch.unique(pred):
        mask_dets = pred == p
    return torch.stack(mask_dets)


def get_greedy_assignment(iou_dist, seg_ids, seg_ids2, filtered_instseg2, thresh=0.8):
    # greedy
    assignments = torch.ones(iou_dist.shape[1]) * - 1
    count = 0
    combined = torch.cat((torch.arange(1, 255+1), seg_ids.cpu()))
    uniques, counts = combined.unique(return_counts=True)
    unused = uniques[counts == 1]
    unused = unused[torch.randperm(unused.shape[0])]
    for c in range(iou_dist.shape[1]):
        if filtered_instseg2 is not None:
            if seg_ids2[c] not in filtered_instseg2:
                continue
        r = torch.argmin(iou_dist[:, c])
        if iou_dist[r, c] < thresh:
            assignments[c] = seg_ids[r]
        else:
            assignments[c] = unused[count]
            count += 1
        
    return assignments


def get_hungarian_assignment(iou_dist, seg_ids, seg_ids2=None, filtered_instseg2=None, thresh=0.8):
    assignments = torch.ones(iou_dist.shape[1]) * - 1
    iou_dist[iou_dist > thresh] = torch.nan
    row, col = solve_dense(iou_dist)
    row, col = row.tolist(), col.tolist()
    count = 0
    combined = torch.cat((torch.arange(1, 255+1), seg_ids.cpu()))
    uniques, counts = combined.unique(return_counts=True)
    unused = uniques[counts == 1]
    unused = unused[torch.randperm(unused.shape[0])]
    for c in range(iou_dist.shape[1]):
        if filtered_instseg2 is not None:
            if seg_ids2[c] not in filtered_instseg2:
                continue
        if c in col:
            assignments[c] = seg_ids[row[col.index(c)]]
        else:
            assignments[c] = unused[count]
            count += 1
    
    return assignments


def get_assignments2D(pred1, pred2, filtered_instseg2, method='hungarian'):
    seg_ids = torch.unique(pred1)
    seg_ids_2 = torch.unique(pred2)
    intersect, union, _, _ = intersect_and_union2D(pred1.clone().detach(), pred2)
    dist = 1 - intersect/union
    if method == 'greedy':
        assignments = get_greedy_assignment(dist.cpu().numpy(), seg_ids)
    else:
        assignments = get_hungarian_assignment(dist.cpu().numpy(), seg_ids, seg_ids_2, filtered_instseg2)
    pred_new = torch.zeros_like(pred2)
    seg_ids_pred2 = torch.unique(pred2)
    for a, s in zip(assignments, seg_ids_pred2):
        pred_new[pred2 == s] = a

    return pred_new


def get_assignments3D(pc1, pc2, instseg1, instseg2, filtered_instseg2, method='hungarian', distance_measure_3D='chamfer'):
    instseg1 = instseg1.clone().detach()
    pc1 = pc1.clone().detach()
    seg_ids = torch.unique(instseg1)
    seg_ids_2 = torch.unique(instseg2)
    if distance_measure_3D == 'chamfer':
        dist = chamfer_distance(pc1.unsqueeze(0), pc2.unsqueeze(0), instseg1, instseg2, filtered_instseg2)
    if method == 'greedy':
        assignments = get_greedy_assignment(dist, seg_ids, seg_ids_2, filtered_instseg2, thresh=40)
    else:
        assignments = get_hungarian_assignment(dist, seg_ids, seg_ids_2, filtered_instseg2, thresh=40)
    instseg_new = torch.zeros_like(instseg2)
    for a, s in zip(assignments, seg_ids_2):
        if a == -1:
            continue
        instseg_new[instseg2 == s] = a
    return instseg_new


def dbscan_filter(pt_cloud, eps=0.5, min_samples=25):
    instseg = pt_cloud[:, 6]
    pts = pt_cloud[:, :3]
    mask = torch.ones(instseg.shape[0], dtype=bool)
    clustering_algo = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)
    for s in torch.unique(instseg):
        seg_pts = pts[instseg == s]
        clustering = clustering_algo.fit(seg_pts.clone().detach().cpu().numpy())
        labels = torch.from_numpy(clustering.labels_)
        uniques, counts = labels.unique(return_counts=True)
