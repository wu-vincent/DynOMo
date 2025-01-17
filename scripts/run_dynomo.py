# Parallelize a function over GPUs.
# Author: Jeff Tan (jefftan@andrew.cmu.edu)
# Usage: gpu_map(func, [arg1, ..., argn]) or gpu_map(func, [(a1, b1), ..., (an, bn)])
# Use the CUDA_VISIBLE_DEVICES environment variable to specify which GPUs to parallelize over:
# E.g. if `your_code.py` calls gpu_map, invoke with `CUDA_VISIBLE_DEVICES=0,1,2,3 python your_code.py`

import multiprocessing
import os
import tqdm
import sys
sys.path.append(os.getcwd())
import argparse
from importlib.machinery import SourceFileLoader
from src.utils.common_utils import seed_everything
import os 
import shutil
from src.model.dynomo import DynOMo
from src.datasets.sequence_dicts import SEQEUNCE_DICT
import json
import glob
import pandas as pd
import numpy as np


def gpu_map(func, args, n_ranks=None, gpus=None, method="static", progress_msg=None):
    """Map a function over GPUs

    Args:
        func: Function to parallelize
        args: List of argument tuples, to split evenly over GPUs
        gpus (List(int) or None): Optional list of GPU device IDs to use
        method (str): Either "static" or "dynamic" (default "static").
            Static assignment is the fastest if workload per task is balanced;
            dynamic assignment better handles tasks with uneven workload.
        progress_msg (str or None): If provided, display a progress bar with
            this description
    Returns:
        outs: List of outputs
    """
    mp = multiprocessing.get_context("spawn")  # spawn allows CUDA usage
    devices = os.getenv("CUDA_VISIBLE_DEVICES")
    outputs = None

    # Compute list of GPUs
    if gpus is None:
        if devices is None:
            num_gpus = int(os.popen("nvidia-smi -L | wc -l").read())
            gpus = list(range(num_gpus))
        else:
            gpus = [int(n) for n in devices.split(",")]

    if n_ranks is None:
        n_ranks = len(gpus)

    # Map arguments over GPUs using static or dynamic assignment
    try:
        # Static assignment: Spawn `ngpu` processes, each with `nargs / ngpu`
        # argument tuples interleaved across GPUs
        if method == "static":
            # Interleave arguments across GPUs
            args_by_rank = [args[rank::n_ranks] for rank in range(n_ranks)]
            args_by_rank = [[a+[gpus[i]] for a in args_by_rank[i]] for i in range(n_ranks)]

            # Spawn processes
            spawned_procs = []
            result_queue = mp.Queue()
            for rank in range(n_ranks):
                gpu_id = gpus[rank % len(gpus)]
                # Environment variables get copied on process creation
                # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                proc_args = (func, args_by_rank[rank], rank, result_queue, progress_msg)
                proc = mp.Process(target=gpu_map_static_helper, args=proc_args)
                proc.start()
                spawned_procs.append(proc)

            # Wait to finish
            for proc in spawned_procs:
                proc.join()

            # Construct output list
            outputs_by_rank = {}
            while True:
                try:
                    rank, out = result_queue.get(block=False)
                    outputs_by_rank[rank] = out
                except multiprocessing.queues.Empty:
                    break

            outputs = []
            for it in range(len(args)):
                rank = it % n_ranks
                idx = it // n_ranks
                outputs.append(outputs_by_rank[rank][idx])

        # Dynamic assignment: Spawn `nargs` processes as GPUs become available,
        # one process for each argument tuple.
        elif method == "dynamic":
            gpu_queue = mp.Queue()
            for rank in range(n_ranks):
                gpu_id = gpus[rank % len(gpus)]
                gpu_queue.put(gpu_id)

            # Spawn processes as GPUs become available
            spawned_procs = []
            result_queue = mp.Queue()
            if progress_msg is not None:
                args = tqdm.tqdm(args, desc=progress_msg)
            for it, arg in enumerate(args):
                # Take latest available gpu_id (blocking)
                gpu_id = gpu_queue.get()
                
                # Environment variables get copied on process creation
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                proc_args = (func, arg + [f"cuda:{gpu_id}"], it, gpu_id, result_queue, gpu_queue)
                proc = mp.Process(target=gpu_map_dynamic_helper, args=proc_args)
                proc.start()
                spawned_procs.append(proc)

            # Wait to finish
            for proc in spawned_procs:
                proc.join()

            # Construct output list
            outputs_by_it = {}
            while True:
                try:
                    it, out = result_queue.get(block=False)
                    outputs_by_it[it] = out
                except multiprocessing.queues.Empty:
                    break

            outputs = []
            for it in range(len(args)):
                outputs.append(outputs_by_it[it])

        # Don't spawn any new processes
        elif method is None:
            return [func(*arg) for arg in args]

        else:
            raise NotImplementedError

    except Exception as e:
        import traceback
        print("".join(traceback.format_exception(None, e, e.__traceback__)))

    # Restore env vars
    finally:
        if devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
        else:
            pass
            # del os.environ["CUDA_VISIBLE_DEVICES"]
        return outputs


def gpu_map_static_helper(func, args, rank, result_queue, progress_msg):
    if progress_msg is not None:
        args = tqdm.tqdm(args, desc=progress_msg)
    out = [func(*arg) if isinstance(arg, tuple) else func(arg) for arg in args]
    result_queue.put((rank, out))


def gpu_map_dynamic_helper(func, arg, it, gpu_id, result_queue, gpu_queue):
    out = func(*arg) if isinstance(arg, tuple) else func(arg)
    gpu_queue.put(gpu_id)
    result_queue.put((it, out))


def run_splatam(args):
    config_file, seq, experiment_args, gpu_id = args
    seq_experiment = SourceFileLoader(
            os.path.basename(config_file), config_file
        ).load_module()

    tracking_iters = seq_experiment.config['tracking_obj']['num_iters']
    tracking_iters_init = seq_experiment.config['tracking_obj']['num_iters_init']
    tracking_iters_cam = seq_experiment.config['tracking_cam']['num_iters']
    
    online_depth = '' if experiment_args['online_depth'] is None else '_' + experiment_args['online_depth']
    online_emb = '' if experiment_args['online_emb'] is None else '_' + experiment_args['online_emb']

    run_name = f"deinsify_stride_1_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}{online_depth}{online_emb}/{seq}"
    # run_name = f"stride_1_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}{online_depth}{online_emb}/{seq}"
    run_name = f"transformed_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}{online_depth}{online_emb}/{seq}"
    run_name = f"s1_0.5wh_v2_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}{online_depth}{online_emb}/{seq}"
    # run_name = f"{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}{online_depth}{online_emb}/{seq}"
    
    # seq_experiment.config['stride'] = 1
    # seq_experiment.config['prune_densify']['use_gaussian_splatting_densification'] = True

    # Set Experiment Seed
    seed_everything(seed=seq_experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        seq_experiment.config["workdir"], run_name
    )

    if experiment_args['just_eval']:
        if not os.path.isfile(os.path.join(results_dir, 'params.npz')):
            print(f"Experiment not there {run_name}")
            return
        else:
            print(f"Evaluating experiment {run_name}")
        with open(os.path.join(results_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        config['run_name'] = run_name
        config['wandb']['name'] = run_name
        config['data']['sequence'] = seq

        with open(os.path.join(results_dir, 'config.json'), 'w') as f:
            json.dump(seq_experiment.config, f)

        config['primary_device'] = f"cuda:{gpu_id}"
        config['just_eval'] = experiment_args['just_eval']
        config['checkpoint'] = True
        config['viz']['vis_all'] = experiment_args['vis_all']
        config['viz']['vis_gt'] = experiment_args['vis_gt']

        dynomo = DynOMo(config)
        dynomo.eval(
            experiment_args['novel_view_mode'],
            experiment_args['eval_renderings'],
            experiment_args['eval_trajs'],
            experiment_args['vis_trajs'],
            experiment_args['vis_grid'],
            experiment_args['vis_fg_only'],
            experiment_args['best_x'],
            experiment_args['alpha_traj'],
            experiment_args['traj_len'],
            )

    else:
        if os.path.isfile(os.path.join(results_dir, 'params.npz')): 
            print(f"Experiment already done {run_name}")
            return
        else:
            print(f"Doing experiment {run_name}")
        # update config with args
        seq_experiment.config['run_name'] = run_name
        seq_experiment.config['data']['sequence'] = seq
        seq_experiment.config['wandb']['name'] = run_name
        seq_experiment.config['just_eval'] = experiment_args['just_eval']
        seq_experiment.config['data']['online_depth'] = experiment_args['online_depth']
        seq_experiment.config['data']['online_emb'] = experiment_args['online_emb']
        seq_experiment.config['primary_device'] = f"cuda:{gpu_id}"

        seq_experiment.config['viz']['vis_trajs'] = experiment_args['vis_trajs']
        seq_experiment.config['viz']['vis_grid'] = experiment_args['vis_grid']
        seq_experiment.config['viz']['vis_all'] = experiment_args['vis_all']
        seq_experiment.config['viz']['vis_gt'] = experiment_args['vis_gt']

        dynomo = DynOMo(seq_experiment.config)
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'config.json'), 'w') as f:
            json.dump(seq_experiment.config, f)
        dynomo.track()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    # optimization flags
    parser.add_argument("--gpus", nargs='+', type=list, help="gpus to use")
    parser.add_argument("--sequence", default=None, help="gpus to use")
    parser.add_argument("--online_depth", default=None, choices=[None, 'DepthAnything', 'DepthAnythingV2-vitl'], help="if computing depth online")
    parser.add_argument("--online_emb", default=None, choices=[None, 'dinov2_vits14', 'dinov2_vits14_reg'], help="if computing embeddings online")
    # evaluation flags
    parser.add_argument("--just_eval", action="store_true", help="if only eval")
    parser.add_argument("--not_eval_renderings", action="store_false", help="if eval renderings")
    parser.add_argument("--not_eval_trajs", action="store_false", help="if eval traj")
    parser.add_argument("--not_vis_trajs", action="store_false", help="if vis evaluation grids")
    parser.add_argument("--not_vis_grid", action="store_false", help="if vis grid")
    parser.add_argument("--vis_bg_and_fg", action="store_false", help="if only vis fg")
    parser.add_argument("--vis_gt", action="store_true", help="ground truth also")
    parser.add_argument("--vis_rendered", action="store_true", help="if visualizing all renderings")
    parser.add_argument("--novel_view_mode", default=None, choices=[None, 'zoom_out', 'circle'], help="if eval novel view")
    parser.add_argument("--best_x", default=1, type=int, help="oracle result, get best Gaussian out of x")
    parser.add_argument("--alpha_traj", action="store_true", help="if using alpha blending for trajectory")
    parser.add_argument("--traj_len", default=10, type=int, help="if using alpha blending for trajectory")
    # parse args
    args = parser.parse_args()

    experiment = SourceFileLoader(
            os.path.basename(args.experiment), args.experiment
        ).load_module()

    args.online_depth = args.online_depth if args.online_depth is not None else \
        experiment.config['data']['online_depth']
    args.online_emb = args.online_emb if args.online_emb is not None else \
        experiment.config['data']['online_emb']

    if args.just_eval or args.novel_view_mode is not None:
        experiment.config['just_eval'] = True

    experiment_args = dict(
        just_eval = experiment.config['just_eval'],
        novel_view_mode=args.novel_view_mode,
        eval_renderings=args.not_eval_renderings,
        vis_trajs=args.not_vis_trajs,
        eval_trajs=args.not_eval_trajs,
        vis_grid=args.not_vis_grid,
        vis_fg_only=args.vis_bg_and_fg,
        vis_gt=args.vis_gt,
        vis_all=args.vis_rendered,
        best_x=args.best_x,
        alpha_traj=args.alpha_traj,
        online_depth=args.online_depth,
        online_emb=args.online_emb,
        traj_len=args.traj_len
        )
        
    configs_to_paralellize = list()
    sequences = list([args.sequence]) if args.sequence is not None else SEQEUNCE_DICT[experiment.config['data']['name']]
    for seq in sequences:
        # copy config and get create runname
        configs_to_paralellize.append([args.experiment, seq, experiment_args])
    gpus = [int(g[0]) for g in args.gpus[0] if g != ',']
    n_ranks = len(gpus)

    gpu_map(
        run_splatam,
        configs_to_paralellize,
        n_ranks=n_ranks,
        gpus=gpus,
        method='static')

    tracking_iters = experiment.config['tracking_obj']['num_iters']
    tracking_iters_init = experiment.config['tracking_obj']['num_iters_init']
    tracking_iters_cam = experiment.config['tracking_cam']['num_iters']
    online_depth = '' if experiment_args['online_depth'] is None else '_' + experiment_args['online_depth']
    online_emb = '' if experiment_args['online_emb'] is None else '_' + experiment_args['online_emb']

    run_name = f"deinsify_stride_1_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}{online_depth}{online_emb}"
    # run_name = f"stride_1_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}{online_depth}{online_emb}"
    run_name = f"transformed_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}{online_depth}{online_emb}"
    run_name = f"s1_0.5wh_v2_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}{online_depth}{online_emb}"
    # run_name = f"{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}{online_depth}{online_emb}"
    
    alpha_add = '' if not args.alpha_traj else '_alpha_traj'
    best_add = '' if args.best_x == 1 else f'_{args.best_x}'
    if 'panoptic_sport' not in args.experiment:
        result_files = os.path.join(
            experiment.config["workdir"], run_name, "*", f"eval/traj_metrics{best_add}{alpha_add}.json"
        )
    else:
        result_files = os.path.join(
            experiment.config["workdir"], run_name, "*/*/*", f"eval/traj_metrics{best_add}{alpha_add}.json"
        )

    summary_short = None
    summary_long = None
    if 'davis' in args.experiment:
        columns = ['d_avg', 'survival', 'median_l2', 'occlusion_accuracy', 'average_jaccard', 'average_pts_within_thresh', 'FPS', 'duration [min]']
    elif 'panoptic_sport' in args.experiment:
        columns = ['d_avg', 'survival', 'median_l2', 'occlusion_accuracy', 'd_avg_3D', 'survival_3D', 'median_l2_3D', 'average_jaccard', 'average_pts_within_thresh', 'FPS', 'duration [min]']
    else:
        columns = ["AJ", "APCK", "occ_acc", "epe", "pck_3d_50cm", "pck_3d_10cm", "pck_3d_5cm", 'FPS', 'duration [min]']

    for f in glob.glob(result_files):
        with open(f, 'r') as jf:
            metrics = json.load(jf)
        if 'iphone' not in args.experiment:
            metrics = {k1: v1 for k, v in metrics.items() for k1, v1 in metrics[k].items()}
        seq = f.replace(
            f"/eval/traj_metrics{best_add}{alpha_add}.json", '').replace(
                os.path.join(experiment.config["workdir"], run_name) + '/', '')
        params = np.load(os.path.join(os.path.dirname(os.path.dirname(f)), 'params.npz'))
        metrics['FPS']  = (params['duration'].item()) # +2.3)
        metrics['duration [min]']  = params['overall_duration'].item() / 60

        if summary_short is None:
            summary_short = pd.DataFrame(columns=columns)
            summary_long = pd.DataFrame(columns=list(metrics.keys()))
            
        summary_short.loc[seq] = {k: v for k, v in metrics.items() if k in columns}
        summary_long.loc[seq] = metrics
    
    summary_short = summary_short.sort_index()
    summary_long = summary_short.sort_index()

    summary_name = f"{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}"
    summary_short.loc['mean'] = summary_short.mean()
    summary_short.to_json(os.path.join(experiment.config["workdir"], run_name, f"{summary_name}{best_add}{alpha_add}.json"))
    print(f"METRICS: \n {summary_short}")

    summary_long.loc['mean'] = summary_long.mean()
    summary_long.to_json(os.path.join(experiment.config["workdir"], run_name, f"{summary_name}{best_add}{alpha_add}_long.json"))


