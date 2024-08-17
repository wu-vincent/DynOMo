
import argparse
import os
import shutil
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

from utils.common_utils import seed_everything
from utils.eval_traj import eval_traj, vis_grid_trajs
import json

cam_seq_match = {
    'softball': 27,
    'basketball': 21,
    'football': 18,
    'juggle': 14,
    'boxes': 27,
    'tennis': 8
}

def get_cams(t, md, seq):
    cams = dict()
    for c in range(len(md['fn'][t])):
        w, h, k, w2c, cam_id = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c], md['cam_id'][t][c]
        cams[cam_id] = {'id': c, 'w2c': w2c, 'k': k, 'h': h, 'w': w}
    return cams

if __name__ == "__main__":
    vis_trajs = False
    get_gauss_wise3D_track = True
    use_round_pix = False
    clip = False
    use_gt_occ = False
    vis_thresh = 0.5
    vis_thresh_start = 0.5
    color_thresh = 1000
    best_x = 1
    traj_len = 0

    results_dirs = list()
    # results_dirs += ["../Dynamic3DGaussians/output_orig_jono/exp1"]
    # results_dirs += ["../Dynamic3DGaussians/output_one_cam_jono/exp1"]
    # results_dirs += ["../Dynamic3DGaussians/output_one_cam_jono_fewer_epochs/stereo"]
    results_dirs += ["../Dynamic3DGaussians/experiments/output_stereo/stereo"]

    

    for _results_dir in results_dirs:
        print(f"Evaluating experimetn {_results_dir}")
        for seq in os.listdir(_results_dir):
            print(f"Evaluating Sequence {seq}!")
            md = json.load(open(f"/scratch/jseidens/data/data/{seq}/train_meta.json", 'r'))  # metadata
            cams = get_cams(0, md, seq)
            config = dict(
                data=dict(
                    basedir="/scratch/jseidens/data/data",
                    gradslam_data_cfg="./configs/data/jono_data.yaml",
                    sequence=f'{seq}/ims/{cam_seq_match[seq]}',
                    desired_image_height=360, #180, #360,
                    desired_image_width=640, #320, #640,
                    start=0,
                    end=-1,
                    stride=1,
                    num_frames=-1,
                    load_embeddings=True,
                    embedding_dim=32,
                    get_pc_jono=False,
                    jono_depth=False
                ))
            results_dir = os.path.join(_results_dir, seq, 'eval')
            metrics = eval_traj(
                config,
                results_dir=results_dir,
                vis_trajs=vis_trajs,
                input_k=cams[cam_seq_match[seq]]['k'],
                input_w2c=cams[cam_seq_match[seq]]['w2c'],
                clip=clip,
                use_gt_occ=use_gt_occ,
                vis_thresh=vis_thresh,
                vis_thresh_start=vis_thresh_start,
                best_x=best_x,
                traj_len=traj_len,
                color_thresh=color_thresh,
                do_transform=False,
                use_round_pix=use_round_pix,
                get_gauss_wise3D_track=get_gauss_wise3D_track)
            print(metrics)
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, 'eval', 'traj_metrics.txt'), 'w') as f:
                f.write(f"Trajectory metrics: {metrics}")

            if False: #experiment.config['viz']['vis_grid']:
                vis_grid_trajs(
                    experiment.config,
                    params=None,
                    cam=None,
                    results_dir=results_dir,
                    orig_image_size=True)
                print(f"Stored visualizations to {results_dir}...")
