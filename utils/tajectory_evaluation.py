import json
import os


def eval_traj(params, basedir, sequence):
    # 3D gt
    with open(os.path.join(basedir, 'annotations', '3dgt.json'), 'r') as f:
        traj_gt_3d = json.load(f)
    traj_gt_3d = traj_gt_3d[sequence]

    # 2D gt
    with open(os.path.join(basedir, 'annotations', '2dgt.json'), 'r') as f:
        traj_gt_3d = json.load(f)
    traj_gt_3d = traj_gt_3d[sequence]
    
    