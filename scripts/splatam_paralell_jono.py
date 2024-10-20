# Parallelize a function over GPUs.
# Author: Jeff Tan (jefftan@andrew.cmu.edu)
# Usage: gpu_map(func, [arg1, ..., argn]) or gpu_map(func, [(a1, b1), ..., (an, bn)])
# Use the CUDA_VISIBLE_DEVICES environment variable to specify which GPUs to parallelize over:
# E.g. if `your_code.py` calls gpu_map, invoke with `CUDA_VISIBLE_DEVICES=0,1,2,3 python your_code.py`

import multiprocessing
import os
import tqdm
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
from src.model.dynomo import DynOMo


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
    print(f"Processing sequence {seq}")
    seq_experiment = SourceFileLoader(
            os.path.basename(config_file), config_file
        ).load_module()

    if experiment_args['stereo']:
        stereo_mode = f"_{experiment_args['stereo_mode']}"
    else:
        stereo_mode = ''

    run_name = f"splatam_{seq}_{experiment_args['tracking_iters']}_{experiment_args['tracking_iters_init']}_{experiment_args['tracking_iters_cam']}"
    print(run_name)

    seq_experiment.config['run_name'] = run_name
    seq_experiment.config['data']['sequence'] = seq
    seq_experiment.config['wandb']['name'] = run_name
    seq_experiment.config['just_eval'] = experiment_args['just_eval']
    seq_experiment.config['primary_device'] = f"cuda:{gpu_id}"

    # Set Experiment Seed
    seed_everything(seed=seq_experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        seq_experiment.config["workdir"], seq_experiment.config["run_name"]
    )
    if seq_experiment.config['just_eval']:
            seq_experiment.config['checkpoint'] = True
    seq_experiment.config['checkpoint'] = True
    
    dynomo = DynOMo(seq_experiment.config)

    if seq_experiment.config['just_eval']:
        if not os.path.isfile(os.path.join(results_dir, 'params.npz')):
            print(f"Experiment not there {run_name}")
            return

        dynomo.eval(
            experiment_args['novel_view_mode'],
            experiment_args['eval_renderings'],
            experiment_args['eval_traj'],
            experiment_args['vis_trajs'],
            )

    else:
        if os.path.isfile(os.path.join(results_dir, 'params.npz')):
            print(f"Experiment already done {run_name}\n\n")
            return
        
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(config_file, os.path.join(results_dir, "config.py"))

        dynomo.track()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--just_eval", default=0, type=int, help="if only eval")
    parser.add_argument("--eval_renderings", default=1, type=int, help="if eval renderings")
    parser.add_argument("--eval_traj", default=1, type=int, help="if eval traj")
    parser.add_argument("--vis_trajs", default=0, type=int, help="if eval traj")
    parser.add_argument("--vis_grid", default=0, type=int, help="if eval traj")
    parser.add_argument("--novel_view_mode", default=None, help="if eval novel view")
    args = parser.parse_args()

    experiment = SourceFileLoader(
            os.path.basename(args.experiment), args.experiment
        ).load_module()

    if args.just_eval or args.novel_view_mode is not None:
        experiment.config['just_eval'] = True

    experiment_args = dict(
        just_eval = experiment.config['just_eval'],
        novel_view_mode=args.novel_view_mode,
        eval_renderings=args.eval_renderings,
        vis_trajs=args.vis_trajs,
        eval_traj=args.eval_traj,
        vis_grid=args.vis_grid,
        )
    
    davis_seqs = [
        'motocross-jump',
        'goat',
        'car-roundabout',
        'breakdance',
        'drift-chicane',
        'drift-straight',
        'judo',
        'soapbox',
        'dogs-jump',
        'parkour',
        'india',
        'pigs',
        'cows',
        'gold-fish',
        'paragliding-launch',
        'camel',
        'blackswan',
        'dog',
        'bike-packing',
        'shooting',
        'lab-coat',
        'kite-surf',
        'bmx-trees',
        'dance-twirl',
        'car-shadow',
        'libby',
        'scooter-black',
        'mbike-trick',
        'loading',
        'horsejump-high']
    
    davis_seqs = ['motocross-jump']
    
    jono_seqs = ["boxes/ims/27", "softball/ims/27", "basketball/ims/21", "football/ims/18", "juggle/ims/14", "tennis/ims/8"]

    configs_to_paralellize = list()
    for seq in jono_seqs:
        # copy config and get create runname
        configs_to_paralellize.append([args.experiment, seq, experiment_args])
    
    # n_ranks = min(torch.cuda.device_count(), len(configs_to_paralellize))
    # gpus = ','.join([str(i) for i in range(n_ranks)])
    
    n_ranks = 3
    gpus = [0,1,6]

    gpu_map(
        run_splatam,
        configs_to_paralellize,
        n_ranks=n_ranks,
        gpus=gpus,
        method='static')
