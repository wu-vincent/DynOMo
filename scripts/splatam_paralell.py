# Parallelize a function over GPUs.
# Author: Jeff Tan (jefftan@andrew.cmu.edu)
# Usage: gpu_map(func, [arg1, ..., argn]) or gpu_map(func, [(a1, b1), ..., (an, bn)])
# Use the CUDA_VISIBLE_DEVICES environment variable to specify which GPUs to parallelize over:
# E.g. if `your_code.py` calls gpu_map, invoke with `CUDA_VISIBLE_DEVICES=0,1,2,3 python your_code.py`

import multiprocessing
import os
import tqdm
import torch
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
from scripts.splatam import RBDG_SLAMMER
import copy


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
    print(f"Processing seq {seq}")
    seq_experiment = SourceFileLoader(
            os.path.basename(config_file), config_file
        ).load_module()

    aniso = 'aniso' if experiment_args['aniso'] else 'iso'
    if experiment_args['aniso']:
        seq_experiment.config['gaussian_distribution'] = "anisotropic"
    else:
        seq_experiment.config['gaussian_distribution'] = "isotropic"

    if 'jono' in seq_experiment.config['data']['gradslam_data_cfg']:
        # seq = os.path.join(seq, 'ims/27')
        tracking_iters_cam = 0
        run_name = f"splatam_{seq}_{experiment_args['seed']}_{experiment_args['mov_init_by']}_{experiment_args['tracking_iters']}_{experiment_args['tracking_iters_init']}_{experiment_args['tracking_iters_cam']}_{experiment_args['num_frames']}_{experiment_args['feature_dim']}_{experiment_args['init_jono']}_{experiment_args['jono_depth']}_{experiment_args['remove_gaussians']}_{experiment_args['sil_thres_gaussians']}_{experiment_args['l1_losses_embedding']}_{experiment_args['l1_losses_color']}_{experiment_args['bg_reg']}_{experiment_args['embeddings_lr']}_{experiment_args['red_lr']}_{experiment_args['red_lr_cam']}_{experiment_args['embedding_weight']}_{experiment_args['use_seg_for_nn']}_{experiment_args['weight_iso']}_{experiment_args['exp_weight']}_{experiment_args['loss_weight_emb']}_{experiment_args['loss_weight_iso']}_{experiment_args['loss_weight_rigid']}_{experiment_args['loss_weight_rot']}_{experiment_args['loss_weight_depth_cam']}_{experiment_args['forward_propagate_camera']}_{experiment_args['trafo_mat']}_{experiment_args['feats_224']}_{experiment_args['restart_if_fail']}_{experiment_args['early_stop']}_{experiment_args['stride']}_{experiment_args['time_window']}_{experiment_args['l1_losses_scale']}_{experiment_args['last_x']}_{experiment_args['kNN']}_{experiment_args['desired_image_height']}_{experiment_args['desired_image_width']}_{experiment_args['instseg_obj']}_{experiment_args['instseg_cam']}_{experiment_args['remove_close']}_transformed"
    else:
        run_name = f"splatam_{seq}/splatam_{seq}_{experiment_args['seed']}_{experiment_args['mov_init_by']}_{experiment_args['tracking_iters']}_{experiment_args['tracking_iters_init']}_{experiment_args['tracking_iters_cam']}_{experiment_args['num_frames']}_{experiment_args['feature_dim']}_{experiment_args['remove_gaussians']}_{experiment_args['sil_thres_gaussians']}_{experiment_args['l1_losses_embedding']}_{experiment_args['l1_losses_color']}_{experiment_args['bg_reg']}_{experiment_args['embeddings_lr']}_{experiment_args['red_lr']}_{experiment_args['red_lr_cam']}_{experiment_args['embedding_weight']}_{experiment_args['use_seg_for_nn']}_{experiment_args['weight_iso']}_{experiment_args['exp_weight']}_{experiment_args['loss_weight_emb']}_{experiment_args['loss_weight_iso']}_{experiment_args['loss_weight_rigid']}_{experiment_args['loss_weight_rot']}_{experiment_args['loss_weight_depth_cam']}_{experiment_args['forward_propagate_camera']}_{experiment_args['trafo_mat']}_{experiment_args['feats_224']}_{experiment_args['restart_if_fail']}_{experiment_args['early_stop']}_{experiment_args['stride']}_{experiment_args['time_window']}_{experiment_args['l1_losses_scale']}_{experiment_args['last_x']}_{experiment_args['kNN']}_{experiment_args['desired_image_height']}_{experiment_args['desired_image_width']}_{experiment_args['instseg_obj']}_{experiment_args['instseg_cam']}_{experiment_args['smoothness']}_{experiment_args['prune_gaussians']}_{experiment_args['use_depth_error_for_adding_gaussians']}_{experiment_args['norm_embeddings']}_{experiment_args['add_only_bg']}_{experiment_args['remove_close']}_{experiment_args['loss_weight_depth_obj']}_{experiment_args['bg_loss']}_{aniso}_deb_l2_emb_r3_l1_all" #norm_embeddings
    print(run_name)

    seq_experiment.config['run_name'] = run_name
    seq_experiment.config['data']['sequence'] = seq
    seq_experiment.config['data']['do_transform'] = True
    seq_experiment.config['wandb']['name'] = run_name

    seq_experiment.config['early_stop'] = experiment_args['early_stop']
    seq_experiment.config['norm_embeddings'] = experiment_args['norm_embeddings']
    seq_experiment.config['stride'] = experiment_args['stride']
    seq_experiment.config['time_window'] = experiment_args['time_window']
    seq_experiment.config['use_wandb'] = experiment_args['use_wandb']
    seq_experiment.config['eval_during'] = experiment_args['eval_during']
    seq_experiment.config['base_transformations'] = experiment_args['base_transformations']
    seq_experiment.config['base_transformations_mlp'] = experiment_args['base_transformations_mlp']

    seq_experiment.config['seed'] = experiment_args['seed']
    seq_experiment.config['tracking_obj']['num_iters'] = experiment_args['tracking_iters']
    seq_experiment.config['tracking_obj']['num_iters_init'] = experiment_args['tracking_iters_init']
    seq_experiment.config['tracking_cam']['num_iters'] = experiment_args['tracking_iters_cam']
    seq_experiment.config['refine']['num_iters'] = experiment_args['refine_iters']
    seq_experiment.config['tracking_obj']['make_grad_bg_smaller'] = experiment_args['make_grad_bg_smaller']
    seq_experiment.config['tracking_obj']['mag_iso'] = experiment_args['mag_iso']
    seq_experiment.config['data']['jono_depth'] = experiment_args['jono_depth']
    seq_experiment.config['data']['get_pc_jono'] = experiment_args['init_jono']
    seq_experiment.config['data']['num_frames'] = experiment_args['num_frames']
    seq_experiment.config['data']['desired_image_height'] = experiment_args['desired_image_height']
    seq_experiment.config['data']['desired_image_width'] = experiment_args['desired_image_width']
    seq_experiment.config['data']['end'] = experiment_args['num_frames']
    seq_experiment.config['remove_gaussians']['remove'] = experiment_args['remove_gaussians']
    seq_experiment.config['add_gaussians']['sil_thres_gaussians'] = experiment_args['sil_thres_gaussians']
    seq_experiment.config['viz']['vis_all'] = experiment_args['vis_all']
    seq_experiment.config['viz']['vis_gt'] = experiment_args['vis_gt']
    seq_experiment.config['just_eval'] = experiment_args['just_eval']

    if experiment_args['l1_losses_embedding'] != 0:
        seq_experiment.config['tracking_obj']['loss_weights']['l1_embeddings'] = experiment_args['l1_losses_embedding']
    if experiment_args['l1_losses_color'] != 0:
        seq_experiment.config['tracking_obj']['loss_weights']['l1_rgb'] = experiment_args['l1_losses_color']
    if experiment_args['l1_losses_scale'] != 0:
        seq_experiment.config['tracking_obj']['loss_weights']['l1_scale'] = experiment_args['l1_losses_scale']
    if experiment_args['l1_losses_opacity'] != 0:
        seq_experiment.config['tracking_obj']['loss_weights']['l1_opacity'] = experiment_args['l1_losses_opacity']
    if experiment_args['l1_losses_bg'] != 0:
        seq_experiment.config['tracking_obj']['loss_weights']['l1_bg'] = experiment_args['l1_losses_bg']

    if experiment_args['bg_reg'] != 0:
        seq_experiment.config['tracking_obj']['loss_weights']['bg_reg'] = experiment_args['bg_reg']
    seq_experiment.config['tracking_obj']['loss_weights']['bg_loss'] = experiment_args['bg_loss']

    if experiment_args['embeddings_lr'] != 0:
        seq_experiment.config['tracking_obj']['lrs']['embeddings'] = experiment_args['embeddings_lr']
    if experiment_args['instseg_obj']:
        seq_experiment.config['tracking_obj']['loss_weights']['instseg'] = experiment_args['instseg_obj']
    if experiment_args['smoothness']:
        seq_experiment.config['tracking_obj']['loss_weights']['smoothness'] = experiment_args['smoothness']
        seq_experiment.config['tracking_cam']['loss_weights']['smoothness'] = experiment_args['smoothness']
    if experiment_args['instseg_cam']:
        seq_experiment.config['tracking_obj']['loss_weights']['instseg'] = experiment_args['instseg_cam']
    
    if experiment_args['red_lr'] == True:
        seq_experiment.config['tracking_obj']['lrs']['means3D'] *= 10
        seq_experiment.config['tracking_obj']['lrs']['unnorm_rotations'] *= 10
        seq_experiment.config['tracking_obj']['lrs']['logit_opacities'] *= 10
        # seq_experiment.config['tracking_obj']['lrs']['log_scales'] *= 10
        seq_experiment.config['tracking_cam']['lrs']['embeddings'] *= 10
    if experiment_args['red_lr_cam'] == True:
        seq_experiment.config['tracking_cam']['lrs']['cam_unnorm_rots'] *= 10
        seq_experiment.config['tracking_cam']['lrs']['cam_trans'] *= 10
    
    if experiment_args['embedding_weight'] == True:
        seq_experiment.config['dist_to_use'] = 'embeddings'
        seq_experiment.config['tracking_obj']['dyno_weight'] = 'embeddings'
    else:
        seq_experiment.config['dist_to_use'] = 'l2'
        seq_experiment.config['tracking_obj']['dyno_weight'] = 'embeddings'
    seq_experiment.config['tracking_obj']['weight_iso'] = experiment_args['weight_iso']
    seq_experiment.config['tracking_obj']['loss_weights']['iso'] = experiment_args['loss_weight_iso']
    seq_experiment.config['tracking_obj']['loss_weights']['embeddings'] = experiment_args['loss_weight_emb']
    seq_experiment.config['tracking_obj']['loss_weights']['rigid'] = experiment_args['loss_weight_rigid']
    seq_experiment.config['tracking_obj']['loss_weights']['rot'] = experiment_args['loss_weight_rot']
    seq_experiment.config['tracking_obj']['loss_weights']['depth'] = experiment_args['loss_weight_depth_obj']
    seq_experiment.config['tracking_cam']['forward_prop'] = experiment_args['forward_propagate_camera']
    seq_experiment.config['tracking_cam']['loss_weights']['depth'] = experiment_args['loss_weight_depth_cam']
    seq_experiment.config['tracking_cam']['loss_weights']['embeddings'] = experiment_args['loss_weight_emb']
    seq_experiment.config['tracking_cam']['loss_weights']['im'] = experiment_args['loss_weight_img']
    seq_experiment.config['tracking_cam']['restart_if_fail'] = experiment_args['restart_if_fail']
    seq_experiment.config['exp_weight'] = experiment_args['exp_weight']

    seq_experiment.config['tracking_obj']['last_x'] = experiment_args['last_x']

    if experiment_args['use_seg_for_nn'] == False:
        seq_experiment.config['use_seg_for_nn'] = False

    seq_experiment.config['remove_outliers_l2'] = experiment_args['remove_outliers_l2']
    seq_experiment.config['trafo_mat'] = experiment_args['trafo_mat']
    seq_experiment.config['data']['feats_224'] = experiment_args['feats_224']
    seq_experiment.config['prune_densify']['prune_gaussians'] = experiment_args['prune_gaussians']
    seq_experiment.config['prune_densify']['pruning_dict']['start_after'] = int(experiment_args['tracking_iters']/2)
    seq_experiment.config['prune_densify']['pruning_dict']['prune_every'] = int(experiment_args['tracking_iters']/2) + 1

    seq_experiment.config['add_gaussians']['use_depth_error_for_adding_gaussians'] = experiment_args['use_depth_error_for_adding_gaussians']
    seq_experiment.config['add_gaussians']['only_bg'] = experiment_args['add_only_bg']
    seq_experiment.config['remove_close'] = experiment_args['remove_close']
    
    seq_experiment.config['primary_device'] = f"cuda:{gpu_id}"

    # Set Experiment Seed
    seed_everything(seed=seq_experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        seq_experiment.config["workdir"], seq_experiment.config["run_name"]
    )
    if seq_experiment.config['just_eval']:
            seq_experiment.config['checkpoint'] = True
    
    if seq_experiment.config['time_window'] == 1:
        rgbd_slammer = RBDG_SLAMMER(seq_experiment.config)
    else:
        rgbd_slammer = RBDG_SLAMMER_WINDOW(seq_experiment.config)

    if seq_experiment.config['just_eval'] and experiment_args['novel_view_mode'] is None:
        if not os.path.isfile(os.path.join(results_dir, 'params.npz')):
            print(f"Experiment not there {run_name}")
            return
        rgbd_slammer.eval()

    elif seq_experiment.config['just_eval']:
        if not os.path.isfile(os.path.join(results_dir, 'params.npz')):
            print(f"Experiment not there {run_name}")
            return
        rgbd_slammer.eval(experiment_args['novel_view_mode'])
        
    else:
        if os.path.isfile(os.path.join(results_dir, 'params.npz')): 
            print(f"Experiment already done {run_name}\n\n")
            return
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(config_file, os.path.join(results_dir, "config.py"))

        rgbd_slammer.rgbd_slam()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--just_eval", default=0, type=int, help="if only eval")
    parser.add_argument("--novel_view_mode", default=None, help="if only eval")
    args = parser.parse_args()

    experiment = SourceFileLoader(
            os.path.basename(args.experiment), args.experiment
        ).load_module()

    if args.just_eval or args.novel_view_mode is not None:
        experiment.config['just_eval'] = True

    experiment_args = dict(
        mov_init_by = experiment.config['mov_init_by'],
        seed = experiment.config['seed'],
        feature_dim = experiment.config['data']['embedding_dim'],
        ssmi_all_mods = experiment.config['tracking_obj']['ssmi_all_mods'],
        load_embeddings = experiment.config['data']['load_embeddings'],
        num_frames = experiment.config['data']['num_frames'],
        dyno_losses = experiment.config['tracking_obj']['dyno_losses'],
        just_eval = experiment.config['just_eval'],
        vis_all = False,
        vis_gt = False,
        tracking_iters = 200,
        tracking_iters_init = 200,
        tracking_iters_cam = 200,
        refine_iters = 0,
        mag_iso = True,
        init_jono = False,
        jono_depth = False,
        l1_losses_embedding = 20,
        l1_losses_color =20, # 0.01,
        bg_reg = 5,
        embeddings_lr = 0.001,
        red_lr = True,
        red_lr_cam = True,
        remove_gaussians = False,
        sil_thres_gaussians = 0.5,
        make_grad_bg_smaller = False,
        remove_outliers_l2 = 100,
        embedding_weight = True,
        use_seg_for_nn = True,
        weight_iso = True,
        exp_weight = 2000,
        loss_weight_iso = 16,
        loss_weight_emb = 16,
        loss_weight_img = 1,
        loss_weight_rigid = 128,
        loss_weight_rot = 16,
        loss_weight_depth_cam=0.1,
        forward_propagate_camera=True,
        trafo_mat=False,
        feats_224=False,
        restart_if_fail=True,
        early_stop=True,
        stride=2,
        l1_losses_scale=20, #0,
        l1_losses_opacity=20, #0,
        l1_losses_bg=20, # 0
        time_window=1,
        last_x=1,
        use_wandb=False,
        eval_during=False,
        base_transformations=False,
        base_transformations_mlp=False,
        kNN=20,
        desired_image_height=240, # 240, # 120, #240, #480, 180, 360
        desired_image_width=455, # 455, # 227, # 455, #910, 320, 640
        instseg_obj=0.0,
        instseg_cam=0.0,
        smoothness=0,
        prune_gaussians=False,
        use_depth_error_for_adding_gaussians=False,
        norm_embeddings=True,
        add_only_bg=False,
        remove_close=True,
        loss_weight_depth_obj=0.1,
        bg_loss=3,
        aniso=True,
        novel_view_mode=args.novel_view_mode
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
    
    # davis_seqs = ['splatam_libby', 'splatam_india', 'splatam_gold-fish', 'splatam_bmx-trees']
    
    jono_seqs = ["boxes/ims/27", "softball/ims/27", "basketball/ims/21", "football/ims/18", "juggle/ims/14", "tennis/ims/8"]

    rgb = [0] # list(range(50))

    configs_to_paralellize = list()
    for seq in davis_seqs:
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
