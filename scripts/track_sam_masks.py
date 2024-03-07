import torch
import numpy as np
import glob
from natsort import natsorted
import argparse
import os
import shutil
import sys
from importlib.machinery import SourceFileLoader
from lapsolver import solve_dense
import cv2


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

from utils.common_utils import seed_everything


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


def main(config, basedir, sequence, threshold=0.5):
    input_folder = os.path.join(basedir, sequence)
    instseg_paths = natsorted(glob.glob(f"{input_folder}/results/*_0sam.npy"))
    instance_id = 0
    for index in range(len(instseg_paths)-1):
        print(index, len(instseg_paths))
        instseg_path = instseg_paths[index]
        instseg = np.load(instseg_path, mmap_mode="r").astype(dtype=np.int64)
        instseg = torch.from_numpy(instseg)
        if index == 0:
            mapping = dict()
            instseg_new = torch.zeros(instseg.shape)
            for i, c in enumerate(torch.unique(instseg)):
                mapping[c] = i+1
                instseg_new[instseg==c] = i+1
            instance_id += 1
            instseg_paths_new = instseg_path[:-4] + 'new.npy'
            np.save(instseg_paths_new, instseg_new.numpy())

            # store image
            viz_gt_instseg = instseg_new.numpy()
            smax, smin = viz_gt_instseg.max(), viz_gt_instseg.min()
            normalized_instseg = np.clip((viz_gt_instseg - smin) / (smax - smin), 0, 1)
            instseg_colormap = cv2.applyColorMap((normalized_instseg * 255).astype(np.uint8), cv2.COLORMAP_JET)
            instseg_paths_new_png = instseg_path[:-4] + 'new.png'
            cv2.imwrite(instseg_paths_new_png, instseg_colormap)

        instseg_2_path = instseg_paths[index+1]
        instseg_2 = np.load(instseg_2_path, mmap_mode="r").astype(dtype=np.int64)
        instseg_2 = torch.from_numpy(instseg_2)

        instseg_classes = torch.unique(instseg_new)
        instseg_2_classes = torch.unique(instseg_2)
        instseg_2_new = torch.zeros(instseg_2.shape)

        # get iou
        area_intersect, area_union, area1, area2 = intersect_and_union(
            instseg_new,
            instseg_2)
        iou_dist = 1 - area_intersect/area_union
        ioa1_dist = 1 - area_intersect/area1.unsqueeze(1).repeat((1, area2.shape[0]))
        ioa2_dist = 1 - area_intersect/area2.unsqueeze(1).repeat((1, area1.shape[0])).T
        ioa_dist = torch.cat([ioa1_dist.unsqueeze(2), ioa2_dist.unsqueeze(2)], dim=2)
        ioa_dist = torch.min(ioa_dist, dim=2).values

        # greedy
        row = list()
        col = list()
        for c in range(instseg_2_classes.shape[0]):
            r = torch.argmin(iou_dist[:, c])
            r2 = torch.argmin(ioa_dist[:, c])
            if iou_dist[r, c] < 0.8: #ioa_dist[r2, c] < 0.9: # iou_dist[r, c] < 0.8:
                row.append(r)
                col.append(c)
        
        # linear sum assignment
        # iou_dist[iou_dist > 0.8] = torch.nan
        # row, col = solve_dense(iou_dist)
        # row, col = row.tolist(), col.tolist()
        
        for r, c in zip(row, col):
            instseg_2_new[instseg_2 == instseg_2_classes[c]] = instseg_classes[r]
        
        for c, _id in enumerate(instseg_2_classes):
            if c in col:
                r = row[col.index(c)]
                instseg_2_new[instseg_2 == _id] = instseg_classes[r]
            else:
                instance_id += 1
                instseg_2_new[instseg_2 == _id] = instance_id
        
        # store instseg npy
        instseg_paths_2_new = instseg_2_path[:-4] + 'new.npy'
        np.save(instseg_paths_2_new, instseg_2_new.numpy())

        # store image
        viz_gt_instseg = instseg_2_new.numpy()
        smax, smin = viz_gt_instseg.max(), viz_gt_instseg.min()
        normalized_instseg = np.clip((viz_gt_instseg - smin) / (smax - smin), 0, 1)
        instseg_colormap = cv2.applyColorMap((normalized_instseg * 255).astype(np.uint8), cv2.COLORMAP_JET)
        instseg_paths_2_new_png = instseg_2_path[:-4] + 'new.png'
        cv2.imwrite(instseg_paths_2_new_png, instseg_colormap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    main(
        experiment.config,
        basedir=experiment.config["data"]["basedir"],
        sequence=os.path.basename(experiment.config["data"]["sequence"])
        )