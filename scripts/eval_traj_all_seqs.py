import os
import sys 

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import argparse
from importlib.machinery import SourceFileLoader
import os
import glob
from src.model.dynomo import DynOMo
from src.evaluate.trajectory_evaluator import TrajEvaluator
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="which dataset to summarize", choices=['davis', 'panoptic_sports', 'iphone'])
    parser.add_argument("--eval_renderings", default=0, type=int, help="if eval renderings")
    parser.add_argument("--eval_traj", default=1, type=int, help="if eval traj")
    parser.add_argument("--vis_trajs", default=0, type=int, help="if eval traj")
    parser.add_argument("--vis_grid", default=0, type=int, help="if eval traj")
    parser.add_argument("--compute_metrics", default=1, type=int, help="if eval traj")
    parser.add_argument("--compute_flow", default=0, type=int, help="if compute flow")
    parser.add_argument("--novel_view_mode", default=None, help="if eval novel view")
    parser.add_argument("--get_gauss_wise3D_track", action='store_false', help="if using one gaussian only or using alpha composition")
    parser.add_argument("--get_from3D", action='store_true', help="if searching in 3D instead of 2D")
    parser.add_argument("--queries_first_t", action='store_true', help="if queries only from first frae")
    parser.add_argument("--best_x", defualt=1, type=int, help="take best of x Gaussians")
    parser.add_argument("--vis_trajs_best_x", action='store_true', help="if visualizing best x")
    parser.add_argument("--traj_len", defautl=0, type=int, help="if 0 only visualize points and not trajs")
    parser.add_argument("--vis_all",action='store_false', help="if visualize all renders")
    parser.add_argument("--vis_gt",action='store_false', help="if visualize GT")
    parser.add_argument("--novel_view_mode", default=None, choices=[None, 'circle', 'test_cam', 'zoom_out'], help="if novel view should be rendered")
    args = parser.parse_args()
    
    if args.dataset == 'iphone':
        base_exp_dirs = f"experiments/dynosplatam_iphone/*/*/eval"
    elif args.dataset == 'davis':
        base_exp_dirs = glob.glob(f"experiments/dynosplatam_davis/*/*/eval")
    else:
        base_exp_dirs = f"experiments/dynosplatam_jono/*/ims/*/eval"        

    vis_trajs = args.vis_trajs
    vis_gird = args.vis_grid
    compute_flow = args.compute_flow

    get_gauss_wise3D_track = args.get_gauss_wise3D_track
    get_from3D = args.get_from3D
    vis_trajs_best_x = args.vis_trajs_best_x
    queries_first_t = True if args.dataset != 'iphone' else args.queries_first_t
    
    best_x = args.best_x
    traj_len = args.traj_len
    vis_all = args.vis_all
    vis_gt = args.vis_gt
    primary_device = "cuda:1"
    novel_view_mode = args.novel_view_mode

    print(f"Evaluating gauss-wise-track {get_gauss_wise3D_track} and get from 3D {get_from3D}.")

    for exp_dirs in base_exp_dirs:
        print(f"\nEvaluating experiment: {exp_dirs}")
        paths = glob.glob(exp_dirs)
        for i, p in enumerate(paths):
            val_dict = dict()
            seq = p.split('/')[-2][8:]
            run_name = '/'.join(p.split('/')[-2:])

            if args.dataset == 'davis':
                seq = p.split('/')[-3]
            elif args.dataset == "panoptic_sports":
                seq = p.split('/')[-4]
            else:
                seq = p.split('/')[-2]

            if 'annotations' in seq:
                continue

            print(f"\mSEQUENCE {seq}...")
            
            # copy config and get create runname
            seq_experiment = SourceFileLoader(
                os.path.basename(args.experiment), args.experiment
                # 'config.py', os.path.join(p, 'config.py')
                ).load_module()

            seq_experiment.config['run_name'] = run_name
            seq_experiment.config['data']['sequence'] = seq
            seq_experiment.config['primary_device'] = primary_device

            if args.eval_traj:
                if not os.path.isfile(os.path.join(p, 'params.npz')):
                    print(f"Experiment not done {run_name} yet.")
                    continue

                if not get_gauss_wise3D_track and (not get_from3D or 'panoptic' not in args.dataset):
                    add_on = '_alpha'
                elif get_from3D:
                    add_on = '_from_3D'
                else:
                    add_on = ''
                if best_x != 1:
                    add_on = add_on + f'_{best_x}'
                if not queries_first_t:
                    add_on = add_on + '_not_only_first'
                
                # if os.path.isfile(os.path.join(p, 'eval', f'traj_metrics{add_on}.json')):
                #     continue
                
                evaluator = TrajEvaluator(
                    seq_experiment.config,
                    results_dir=p + '/eval',
                    vis_trajs=vis_trajs, # seq_experiment.config['viz']['vis_tracked'])
                    best_x=best_x,
                    traj_len=traj_len,
                    get_gauss_wise3D_track=get_gauss_wise3D_track,
                    get_from3D=get_from3D,
                    vis_trajs_best_x=vis_trajs_best_x,
                    queries_first_t=queries_first_t,
                    primary_device=primary_device)

                if args.compute_metrics:
                    metrics = evaluator.eval_traj()
                    print("Trajectory metrics:", metrics)
                    with open(os.path.join(p, 'eval', f'traj_metrics{add_on}.json'), 'w') as f:
                        json.dump(metrics, f)

                if vis_gird:
                    evaluator.vis_grid_trajs()
                    print(f"Stored visualizations to {p}...")
                
                if compute_flow:
                    evaluator.vis_flow()

            if args.eval_renderings or args.novel_view_mode is not None:
                if not os.path.isfile(os.path.join(p, 'eval', f'traj_metrics.json')):
                    print(f"Experiment not done {run_name} yet.")
                    continue
                seq_experiment.config['use_wandb'] = False
                seq_experiment.config['checkpoint'] = True
                seq_experiment.config['just_eval'] = True
                seq_experiment.config['viz']['vis_all'] = vis_all if novel_view_mode != '' else False
                seq_experiment.config['viz']['vis_gt'] = vis_gt if novel_view_mode != '' else False
                seq_experiment.config['run_name'] = run_name
                seq_experiment.config['data']['sequence'] = seq

                dynomo = DynOMo(seq_experiment.config)
                dynomo.eval(
                    novel_view_mode=args.novel_view_mode,
                    eval_traj=0,
                    vis_trajs=0)
