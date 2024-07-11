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
from scripts.splatam import RBDG_SLAMMER
import copy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--just_eval", default=0, type=int, help="if only eval")

    args = parser.parse_args()

    experiment = SourceFileLoader(
            os.path.basename(args.experiment), args.experiment
        ).load_module()
    experiment.config['just_eval'] = args.just_eval

    mov_init_by = experiment.config['mov_init_by']
    seed = experiment.config['seed']
    feature_dim = experiment.config['data']['embedding_dim']
    ssmi_all_mods = experiment.config['tracking_obj']['ssmi_all_mods']
    load_embeddings = experiment.config['data']['load_embeddings']
    num_frames = experiment.config['data']['num_frames']
    dyno_losses = experiment.config['tracking_obj']['dyno_losses']
    just_eval = experiment.config['just_eval']
    vis_all = False
    vis_gt = False

    tracking_iters = 500
    tracking_iters_init = 1000
    tracking_iters_cam = 100
    mag_iso = True
    init_jono = False
    jono_depth = False
    l1_losses = 0
    bg_reg = 0
    embeddings_lr = 0.001
    red_lr = False
    make_grad_bg_smaller = False

    remove_gaussians = False
    sil_thres_gaussians = 0.5

    primary_device = "cuda:0"

    davis_seqs_1 = [
        'motocross-jump',
        'goat',
        'car-roundabout',
        'breakdance',
        'drift-chicane',
        'drift-straight',
        'judo',
        'soapbox',
        'dogs-jump',
        'parkour']
    davis_seqs_2 = [
        'india',
        'pigs',
        'cows',
        'gold-fish',
        'paragliding-launch',
        'camel',
        'blackswan',
        'dog',
        'bike-packing',
        'shooting']
    davis_seqs_3 = [
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
    
    # jono_seqs = os.listdir(experiment.config['data']['basedir'])
    jono_seqs = ["softball/ims/27", "basketball/ims/21", "football/ims/18", "juggle/ims/14", "tennis/ims/8", "boxes/ims/27"]

    for seq in davis_seqs_1 + davis_seqs_2 + davis_seqs_3:
        print(f"SEQUENCE {seq}...")
        if 'annotations' in seq:
            continue
        # copy config and get create runname
        seq_experiment = SourceFileLoader(
            os.path.basename(args.experiment), args.experiment
            ).load_module()

        if 'jono' in experiment.config['data']['gradslam_data_cfg']:
            # seq = os.path.join(seq, 'ims/27')
            tracking_iters_cam = 0
            run_name = f"splatam_{seq}_{seed}_{mov_init_by}_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}_{num_frames}_{feature_dim}_{init_jono}_{dyno_losses}_{jono_depth}_{mag_iso}_{remove_gaussians}_{sil_thres_gaussians}_{l1_losses}_{bg_reg}_{embeddings_lr}"
        else:
            run_name = f"splatam_{seq}/splatam_{seq}_{seed}_{mov_init_by}_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}_{num_frames}_{feature_dim}_{dyno_losses}_{mag_iso}_{remove_gaussians}_{sil_thres_gaussians}_{l1_losses}_{bg_reg}_{embeddings_lr}_{red_lr}_initcam"

        seq_experiment.config['run_name'] = run_name
        seq_experiment.config['data']['sequence'] = seq
        seq_experiment.config['wandb']['name'] = run_name

        seq_experiment.config['tracking_obj']['num_iters'] = tracking_iters
        seq_experiment.config['tracking_obj']['num_iters_init'] = tracking_iters_init
        seq_experiment.config['tracking_cam']['num_iters'] = tracking_iters_cam
        seq_experiment.config['tracking_obj']['make_grad_bg_smaller'] = make_grad_bg_smaller
        if make_grad_bg_smaller:
            seq_experiment.config['tracking_obj']['disable_rgb_grads_old'] = False
        seq_experiment.config['tracking_obj']['mag_iso'] = mag_iso
        seq_experiment.config['data']['get_pc_jono'] = init_jono
        seq_experiment.config['data']['jono_depth'] = jono_depth
        seq_experiment.config['remove_gaussians']['remove'] = remove_gaussians
        seq_experiment.config['add_gaussians']['sil_thres_gaussians'] = sil_thres_gaussians
        seq_experiment.config['primary_device'] = primary_device
        seq_experiment.config['viz']['vis_all'] = vis_all
        seq_experiment.config['viz']['vis_gt'] = vis_gt
        seq_experiment.config['just_eval'] = just_eval

        if l1_losses != 0:
            seq_experiment.config['tracking_obj']['loss_weights']['l1_embeddings'] = l1_losses
            seq_experiment.config['tracking_obj']['loss_weights']['l1_rgb'] = l1_losses
            # seq_experiment.config['tracking_obj']['loss_weights']['l1_scale'] = l1_losses
        
        if bg_reg != 0:
            seq_experiment.config['tracking_obj']['loss_weights']['bg_reg'] = bg_reg
        
        if embeddings_lr != 0:
            seq_experiment.config['tracking_obj']['lrs']['embeddings'] = embeddings_lr
        
        if red_lr == True:
            seq_experiment.config['tracking_obj']['lrs']['means3D'] *= 10
            seq_experiment.config['tracking_obj']['lrs']['unnorm_rotations'] *= 10
            seq_experiment.config['tracking_obj']['lrs']['logit_opacities'] *= 10
            seq_experiment.config['tracking_cam']['lrs']['cam_unnorm_rots'] *= 10
            seq_experiment.config['tracking_cam']['lrs']['cam_trans'] *= 10
            seq_experiment.config['tracking_cam']['lrs']['embeddings'] *= 10

        # Set Experiment Seed
        seed_everything(seed=experiment.config['seed'])
        
        # Create Results Directory and Copy Config
        results_dir = os.path.join(
            seq_experiment.config["workdir"], seq_experiment.config["run_name"]
        )
        if seq_experiment.config['just_eval']:
            seq_experiment.config['checkpoint'] = True

        rgbd_slammer = RBDG_SLAMMER(seq_experiment.config)

        if seq_experiment.config['just_eval']:
            if not os.path.isfile(os.path.join(results_dir, 'eval', 'traj_metrics.txt')):
                print(f"Experiment not there {run_name}")
                continue
            rgbd_slammer.eval()
        else:
            if os.path.isfile(os.path.join(results_dir, 'eval', 'traj_metrics.txt')):
                print(f"Experiment already done {run_name}")
                continue
            
            os.makedirs(results_dir, exist_ok=True)
            shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

            rgbd_slammer.rgbd_slam()