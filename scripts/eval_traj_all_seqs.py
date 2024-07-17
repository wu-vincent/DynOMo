import os
import sys 

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import argparse
from importlib.machinery import SourceFileLoader
from utils.common_utils import seed_everything
import os
import shutil
import copy
from utils.eval_traj import eval_traj, vis_grid_trajs, eval_traj_window
import glob
from scripts.splatam import RBDG_SLAMMER


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--eval_renderings", type=int, default=0, help="Path to experiment file")
    parser.add_argument("--eval_traj", type=int, default=1, help="Path to experiment file")


    args = parser.parse_args()
    
    dataset = 'jono'
    # experiment = '0_kNN_100_500_0_-1_32_True_True_False_True_False_True' # ssmi and mag iso
    # experiment = '0_kNN_100_500_0_-1_32_True_False_False_True_False_True' # mag iso
    # experiment = '0_kNN_100_500_0_-1_32_True_False_False_True_False' # orig
    ### ABOVE BUG IN LOADING HYPERPARAMS
    experiments = list()
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True"]  # mag iso longer
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True_False_0.5"] # mag iso longer no depth rem, less add 
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True_True_0.5_0_0_0"] # mag iso longer add less
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True_True_0.95_0_0_0.001"] # mag iso longer embeddings
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True_False_0.95_0_0_0"] # mag iso no add
    # experiments += ['0_kNN_500_1000_0_-1_32_False_True_False_True_True_0.5_0_0_0.001'] # mag iso longer add less embeddings
    # experiments += ["0_kNN_100_100_0_-1_32_False_True_True_True_False_0.5_0_0_0_debug_jono"] # debug depth
    # experiments += ["0_kNN_100_100_0_-1_32_False_True_False_True_True_0.5_0_0_0"] # mag iso add less opa/means/rot/cam/embeddings higher lr
    # experiments += ["0_kNN_100_100_0_-1_32_False_True_False_True_True_0.5_0_0_0.001"] # mag iso add less embeddings opa/means/rot/cam/embeddings higher lr
    # experiments += ["0_kNN_100_500_0_-1_32_False_True_False_True_True_0.5_0_0_0.001"] # 500 init mag iso add less embeddings opa/means/rot/cam/embeddings higher lr
    # experiments += ["0_kNN_100_500_0_-1_32_False_True_False_True_True_0.5_0_0_0"] # 500 init mag iso add less opa/means/rot/cam
    # experiments += ["0_kNN_500_1000_500_-1_32_False_True_False_True_False_0.5_0_0_0.001_False"] # mag iso longer add less no rem embeddings l1 on rgb and embeddings
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True_False_0.5_0_0_0.001_False"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_True_True_False_0.5_0_0_0.001_False"] # jono depth latest setting
    # experiments += ["0_kNN_500_1000_500_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_False_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_False_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_1_1_5_1"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_0_1"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_0_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_0_1"]

    ### BEST SO FAR
    # experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_5_5_0.001_False_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    experiments += ["0_kNN_500_1000_0_-1_32_True_True_False_0.5_5_5_5_0.001_False_False_False_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    
    dataset = 'davis'
    ### ABOVE BUG IN LOADING HYPERPARAMS 
    experiments = list()
    # experiments += ["0_kNN_100_500_100_-1_32_True_True_True_0.95"] # mag iso
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_True_0.95"] # mag iso longer
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_False_0.5"] # mag iso longer, add less, no rem
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_False_0.5_0_0_0_initcam"] # mag iso longer, add less, no rem, no cam forward
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_True_0.5_0_0_0_initcam"] # mag iso longer, add less, rem, no cam forward
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_True_0.5_0_0_0.001_initcam"] # mag iso longer, add less, rem, no cam forward, embeddings
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_True_0.5_0_0_0_True_initcam"] # mag iso add less opa/means/rot/cam higher lr
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0_True_initcam"] # mag iso add less no rem opa/means/rot/cam higher lr
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_True_0.5_0_0_0.001_True_initcam"] # mag iso add less embeddings opa/means/rot/cam/embeddings higher lr
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_False_initcam"] # mag iso add less no rem embeddings opa/means/rot/cam/embeddings higher lr red
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_False_True"] # l1 loss embeddings 5
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_True_False"] # use embeddings but with cosine distance and exp weight and no seg masks for kNN
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_True_True"] # use embeddings but with cosine distance and exp weight
    # experiments += ["0_kNN_100_100_0_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_False_True"] # no cam forwars
    # experiments += ["0_kNN_0_0_100_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_debug_just_fg"]
    # experiments += ["0_kNN_0_0_300_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_debug_just_fg"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_debug_just_fg"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_True_debug_just_fg"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_False_debug_just_fg"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_True_True_True_2000"] # cosine distance as distance
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_True_True_True_2000_16_0.1_True"]  # cosine distance as distance, cosine similarity as weight, iso weight 16, use weight for iso, cam prop, 0.1 depth weight cam
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_False"]
    # experiments += ["0_kNN_100_100_0_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_False"]
    # experiments += ["0_kNN_100_100_0_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_True"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_True"]
    # experiments += ["0_kNN_100_100_0_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_True_no_sc"]

    # experiments += ["0_kNN_100_100_0_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_False_False_True_False"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True"]
    # experiments += ["0_kNN_100_100_0_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_False_True"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_no_sc"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_no_sc_rgb"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_5_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_no_sc_rgb"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_5_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_all"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_5_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_all_updaterestart"]
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_False_0.5_5_5_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_all_updaterestart"]
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_False_0.5_5_5_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_all_updaterestart_othermask"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_1_1_5"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_0.01"]
    # experiments += ["0_kNN_10_10_10_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_100000_5 "]

    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_8_5"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_100000_5"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_8"]
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_False_0.5_0_5_0.001_True_0_100_True_False_False_16_1_2_8"]
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_False_0.5_0_5_0.001_True_0_100_True_False_False_16_1_2_8_False_False"]
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_False_0.5_0_5_0.001_True_0_100_True_False_False_16_1_2_8_True_True"]
    # experiments += ["../SplaTAM/experiments/dynosplatam_davis"]
    
    # experiments += ["0_kNN_500_1000_0_-1_32_False_0.5_5_0_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_False_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_20_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_20_1"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_10_1"]
    
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1_aniso"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_0_4_4_0.1_True_False_False_True_True_2_1_5_1_20_240_455_aniso"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_0_0_4_0.1_True_False_False_True_True_2_1_5_1_aniso"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_0_0_0_0.1_True_False_False_True_True_2_1_5_1_aniso"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_4_4_0.1_True_False_False_True_True_2_1_5_1_60_240_455_aniso"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_4_4_0.1_True_False_False_True_True_2_1_5_1_60_480_910_aniso"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_4_4_0.1_True_False_False_True_True_2_1_5_1_20_120_227_aniso"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_4_0.1_True_False_False_True_True_2_1_5_1_20_240_455_aniso"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_0.1_True_False_False_True_True_2_1_5_1_20_240_455_aniso"]
    experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_4_0.1_True_False_False_True_True_2_1_5_1_20_240_455_0.0_0.0_0.0_aniso_round3"]

    # dataset = 'jono_baseline'
    # experiments = list()
    # experiments += ["output_one_cam_jono"]
    # experiments += ["output_one_cam_fewer_epochs"]
    # experiments += ["output_orig"]

    load_gaussian_tracks = False
    vis_trajs = False
    clip = False
    use_gt_occ = False
    vis_thresh = 0.01
    vis_thresh_start = 0.25
    best_x = 1
    traj_len = 5
    vis_all = True
    primary_device = "cuda:7"

    for experiment in experiments:
        if dataset == 'davis':
            if not "SplaTAM" in experiment:
                exp_dirs = f"experiments/dynosplatam_davis/*/*_{experiment}"
            else:
                exp_dirs = f"../SplaTAM/experiments/dynosplatam_davis/*"
        elif dataset == "jono":
            exp_dirs = f"experiments/dynosplatam_jono/*/ims/*_{experiment}"
        else:
            exp_dirs = f"../Dynamic3DGaussians/{experiment}/exp1/*"

        print(f"\nEvaluating experiment {experiment}")
        paths = glob.glob(exp_dirs)
        for i, p in enumerate(paths):
            val_dict = dict()
            if dataset == 'davis':
                if not "SplaTAM" in experiment:
                    seq = p.split('/')[-2][8:]
                    run_name = '/'.join(p.split('/')[-2:])
                else:
                    seq = p.split('/')[-1][:-2]
                    run_name = p.split('/')[-1]

            elif dataset == "jono":
                seq = '/'.join(p.split('/')[-3:])
                seq = seq[8:].split('_')[0]
                run_name = '/'.join(p.split('/')[-3:])
            else:
                seq = p.split('/')[-1]
                run_name = p.split('/')[-1]

            if 'annotations' in seq:
                continue

            if 'motocr' not in seq and 'libby' not in seq:
                continue

            print(f"SEQUENCE {seq}...")
            
            # copy config and get create runname
            seq_experiment = SourceFileLoader(
                os.path.basename(args.experiment), args.experiment
                # 'config.py', os.path.join(p, 'config.py')
                ).load_module()

            seq_experiment.config['run_name'] = run_name
            seq_experiment.config['data']['sequence'] = seq
            seq_experiment.config['primary_device'] = primary_device

            print(experiment)
            if "120_227" in experiment:
                seq_experiment.config['data']['desired_image_height'] = 120
                seq_experiment.config['data']['desired_image_width'] = 227
            elif "480_910" in experiment:
                seq_experiment.config['data']['desired_image_height'] = 480
                seq_experiment.config['data']['desired_image_width'] = 910

            if args.eval_traj:
                if not os.path.isfile(os.path.join(p, 'eval', 'traj_metrics.txt')) and not "SplaTAM" in experiment:
                    print(f"Experiment not done {run_name} yet.")
                    continue
                elif not os.path.isfile(f"../SplaTAM/experiments/dynosplatam_davis/{run_name}/params.npz") and "SplaTAM" in experiment:
                    print(f"Experiment not done {run_name} yet.")
                    continue

                if seq_experiment.config['time_window'] == 1:
                    metrics = eval_traj(
                        seq_experiment.config,
                        results_dir=p,
                        load_gaussian_tracks=load_gaussian_tracks,
                        vis_trajs=vis_trajs, # seq_experiment.config['viz']['vis_tracked'])
                        clip=clip,
                        use_gt_occ=use_gt_occ,
                        vis_thresh=vis_thresh,
                        vis_thresh_start=vis_thresh_start,
                        best_x=best_x,
                        traj_len=traj_len)
                    print(metrics)
                    with open(os.path.join(p, 'eval', 'traj_metrics.txt'), 'w') as f:
                        f.write(f"Trajectory metrics: {metrics}")
                    if False: # seq_experiment.config['viz']['vis_grid']:
                        vis_grid_trajs(
                            seq_experiment.config,
                            params=None,
                            cam=None,
                            results_dir=p,
                            orig_image_size=True,
                            no_bg=False,
                            clip=True,
                            traj_len=0,
                            vis_thresh=vis_thresh)
                        print(f"Stored visualizations to {p}...")
                else:
                    metrics = eval_traj_window(
                        seq_experiment.config,
                        results_dir=p,
                        load_gaussian_tracks=load_gaussian_tracks,
                        vis_trajs=vis_trajs, # seq_experiment.config['viz']['vis_tracked'])
                        clip=clip,
                        use_gt_occ=use_gt_occ,
                        vis_thresh=vis_thresh,
                        vis_thresh_start=vis_thresh_start,
                        best_x=best_x,
                        traj_len=traj_len)
                    print(metrics)
                    with open(os.path.join(p, 'eval', 'traj_metrics.txt'), 'w') as f:
                        f.write(f"Trajectory metrics: {metrics}")

            if args.eval_renderings:
                if not os.path.isfile(os.path.join(p, 'eval', 'traj_metrics.txt')) and not "SplaTAM" in experiment:
                    print(f"Experiment not done {run_name} yet.")
                    continue
                seq_experiment.config['use_wandb'] = False
                seq_experiment.config['checkpoint'] = True
                seq_experiment.config['just_eval'] = True
                seq_experiment.config['viz']['vis_all'] = vis_all
                # seq_experiment.config['data']['jono_depth'] = True
                seq_experiment.config['run_name'] = run_name
                seq_experiment.config['data']['sequence'] = seq
                seq_experiment.config['wandb']['name'] = run_name
 
                rgbd_slammer = RBDG_SLAMMER(seq_experiment.config)
                rgbd_slammer.eval()
