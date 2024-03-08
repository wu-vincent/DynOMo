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