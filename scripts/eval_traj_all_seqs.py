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


def get_jono_base_exps():
    experiments = list()
    names = list()
    experiments += ["output_orig_jono"]
    names.append("original_jono")
    experiments += ["output_one_cam_jono"]
    names.append("jono_one_cam")
    experiments += ["output_one_cam_jono_fewer_epochs"]
    names.append("jono_one_cam_fewer_epochs")

    return experiments, names

def get_davis_exps():
    experiments = list()
    names = list()
    
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # names.append('iso')
    # iso experiment 
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # names.append('iso')
    # aniso experiment
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1_aniso"]
    # names.append('aniso')
    # 16-16 4 experiment
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_4_0.1_True_False_False_True_True_2_1_5_1_20_240_455_aniso"]
    # names.append('16-16-4')
    # stringer bg scale
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_4_0.1_True_False_False_True_True_2_1_5_1_20_240_455_0.0_0.0_0.0_bgsclaestrong_aniso"]
    # names.append('strong_bg_sclae')
    # 16-16-16 experiment
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_0.1_True_False_False_True_True_2_1_5_1_20_240_455_aniso"]
    # names.append('16-16-16')
    # setting scale l1 to 20
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_16_0.1_True_False_False_True_True_2_1_20_1_20_240_455_0.0_0.0_0.0_False_aniso"]
    # names.append('sclae_l1_20')
    # increasing weight of rigid and rot
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_32_32_0.1_True_False_False_True_True_2_1_5_1_20_240_455_0.0_0.0_0.0_False_aniso"]
    # names.append('16-32-32')
    
    # fixing scale over time    
    experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_aniso"]
    names.append('sclae_fix')

    ### Ablations
    # no embeddings plus instseg
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_0_16_16_4_0.1_True_False_False_True_True_2_1_5_1_20_240_455_16.0_16.0_0.0_aniso"]
    # ames.append('no_emb_plus_inst')
    # no embeddings
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_0_16_4_4_0.1_True_False_False_True_True_2_1_5_1_20_240_455_0.0_0.0_0.0_aniso"]
    # ames.append('no_emb')
    # no iso
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_0_4_4_0.1_True_False_False_True_True_2_1_5_1_20_240_455_aniso"]
    # ames.append('0_4_4')
    # no rigid
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_0_0_4_0.1_True_False_False_True_True_2_1_5_1_aniso"]
    # ames.append('0_0_4')
    # no rot
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_0_0_0_0.1_True_False_False_True_True_2_1_5_1_aniso"]
    # ames.append('0_0_0')

    # less epochs
    # 16-16-16
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_aniso"]
    # names.append('less_ep_16_16_16')
    # 4-16-4
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_4_16_4_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_aniso"]
    # names.append('less_ep_4_16_4')
    # 16-32-16
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_32_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_aniso"]
    names.append('less_ep_16_32_16')
    # 16-32-32
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_32_32_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_aniso"]
    # names.append('less_ep_16_32_32')
    # 160_160_160_160
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_160_160_160_160_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_aniso"]
    # names.append('less_ep_160_160_160_160')
    # 160_160_160
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_160_160_160_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_aniso"]
    # names.append('less_ep_160_160_160')
    # 20_20
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_16_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_aniso"]
    # names.append('less_ep_20_20')
    # 16_0_0
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_0_0_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_aniso"]
    # names.append('less_ep_16_0_0')
    # not_restart
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_16_0.1_True_False_False_False_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_aniso"]
    # names.append('less_ep_16_16_16_not_re')
    # 16_32_16 20 20
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_32_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_aniso_r2"]
    names.append('less_ep_16_32_16_20_20')
    # 16_32_32 20 20
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_32_32_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_aniso_r2"]
    # names.append('less_ep_16_32_32_20_20')
    #  16_16_16 no cam optim
    # experiments += ["0_kNN_200_200_0_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_False_aniso"]
    # names.append('less_ep_16_16_16_no_cam')
    # 16_16_16 simple trans cam
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_False_aniso_simpletrnascam"]
    # names.append('less_ep_16_16_16_simple_trans_cam')
    # 16_16_16 smoothness 1
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_1.0_False_False_True_False_False_aniso"]
    # names.append('less_ep_16_16_16_smooth1')
    # 16_16_16 smoothness 5
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_5.0_False_False_True_False_False_0.1_aniso"]
    # names.append('less_ep_16_16_16_smooth5')
    # 16_16_16 no depth cam
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_16_0_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_False_0.1_aniso"]
    # names.append('less_ep_16_16_16_nodepthcam')
    # 16_16_16 no depth obj
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_False_0_aniso"]
    # names.append('less_ep_16_16_16_nodepthobj')
    # 16_128_16
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_False_0.1_aniso"]
    # names.append('less_ep_16_128_16')
    # 126_16_16
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_128_16_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_False_0.1_aniso"]
    # names.append('less_ep_128_16_16')
    # 16_128_16_20_20_20
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_20_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_False_0.1_aniso"]
    # names.append("less_16_128_16_20_20_20")

    # 16_128_16_20_20_5
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_False_0.1_aniso"]
    names.append("less_16_128_16_20_20_5")
    
    # 16_128_16_20_20_20 0.1 smooth
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_20_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.1_False_False_True_False_False_0.1_aniso"]
    names.append("less_16_128_16_20_20_20_0.1")

    # 16_128_16_20_20_5 repeat
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_False_0.1_aniso"]
    names.append("less_16_128_16_20_20_20_5_repeat")

    # long training 16_128_16_20_20_5
    experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_False_0.1_aniso"]
    names.append("16_128_16_20_20_5")

    # stride 1 again
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_1_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_False_0.1_aniso_repeat"]
    names.append("less_16_128_16_20_20_5_1")

    # no embeddings 0_16_128_16_20_20_5
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_0_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_False_0.1_aniso_repeat"]
    # names.append("0_16_128_16_20_20_5")

    # no embeddings, higher im 8_0_16_128_16_20_20_5
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_0_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_False_0.1_8.0_aniso_repeat"]
    # names.append("8_0_16_128_16_20_20_5")

    # 16_128_16_20_20_5 full res
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_480_910_0.0_0.0_0_False_False_True_False_False_0.1_aniso"]
    # names.append('16_128_16_20_20_5_full_res')

    # 16_128_16_20_20_5 high depth
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_1.0_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_False_1.0_aniso"]
    # names.append('16_128_16_20_20_5_high_depth')

    return experiments, names


def get_jono_exps():
    experiments = list()
    names = list()
    ### BEST SO FAR
    # experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # names.append('ours')
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_5_5_0.001_False_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # names.append('jono_depth')
    # experiments += ["0_kNN_500_1000_0_-1_32_True_True_False_0.5_5_5_5_0.001_False_False_False_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # names.append('jono_depth_jono_pc_inig')
    
    #### higher rigidity ####
    # davis size
    experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0"]
    names.append('not_higher_lr')

    # davis size cam
    experiments += ["0_kNN_500_1000_500_-1_32_False_False_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0"]
    names.append('not_higher_lr_plus_cam')
    experiments += ["0_kNN_200_200_200_-1_32_False_False_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0"]
    names.append('higher_lr_plus_cam_plus_short')
    
    # jono depth
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_180_320_0.0_0.0"]
    names.append('not_higher_lr_jono_depth')
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_180_320_0.0_0.0_transformed"]
    names.append('not_higher_lr_normal_size_transformed_jono_depth')
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_transformed"]
    names.append('not_higher_lr_orig_size_transformed_jono_depth')
    
    # orig size
    experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0"]
    names.append('not_higher_lr_orig_size')
    experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_transformed"]
    names.append('not_higher_lr_original_size_transformed')

    # normal size
    experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_180_320_0.0_0.0"]
    names.append('not_higher_lr_normal_size')
    experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_180_320_0.0_0.0_transformed"]
    names.append('not_higher_lr_normal_size_transformed')

    experiments = list()
    names = list()
    # jono pc
    experiments += ["0_kNN_500_1000_0_-1_32_True_True_False_0.5_20_20_5_0.001_False_False_False_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_180_320_0.0_0.0_transformed"]
    names.append('not_higher_lr_normal_size_jono_pc')


    return experiments, names

def get_splatam_exps():
    experiments = list()
    names = list()
    experiments += ["../SplaTAM/experiments/dynosplatam_davis_init_params"]
    names += ["initial_splatam"]
    experiments += ["../SplaTAM/experiments/dynosplatam_davis_longer"]
    names += ["longer_splatam"]

    return experiments, names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--eval_renderings", type=int, default=0, help="Path to experiment file")
    parser.add_argument("--eval_novel_view", type=int, default=0, help="Path to experiment file")
    parser.add_argument("--eval_traj", type=int, default=1, help="Path to experiment file")


    args = parser.parse_args()
    
    dataset = 'davis'
    # dataset = 'davis_splatam'
    dataset = 'jono'
    # dataset = 'jono_baseline'

    if dataset == 'jono_baseline':
        experiments, names = get_jono_base_exps()
    elif dataset == 'jono':
        experiments, names = get_jono_exps()
    elif dataset == 'davis':
        experiments, names = get_davis_exps()
    elif dataset == 'davis_splatam':
        experiments, names = get_splatam_exps()

    load_gaussian_tracks = False
    vis_trajs = True
    vis_gird = True

    get_gauss_wise3D_track = True
    use_round_pix = False
    get_from3D = True
    
    clip = False
    use_gt_occ = True
    vis_thresh = -0.01
    vis_thresh_start = 0.5
    color_thresh = 1000
    best_x = 1
    traj_len = 10
    vis_all = True
    primary_device = "cuda:4"
    novel_view_mode = 'circle' # 'circle', 'test_cam'

    print(f"Evaluating gauss-wise-track {get_gauss_wise3D_track} and round pixel {use_round_pix}.")

    for name, experiment in zip(names, experiments):
        if dataset == 'davis':
            exp_dirs = f"experiments/dynosplatam_davis/*/*_{experiment}"
        elif dataset == 'davis_splatam':
            exp_dirs = f"{experiment}/*"
        elif dataset == "jono":
            exp_dirs = f"experiments/dynosplatam_jono/*/ims/*_{experiment}"
        else:
            exp_dirs = f"../Dynamic3DGaussians/{experiment}/exp1/*"

        print(f"\nEvaluating experiment {name}: {experiment}")
        paths = glob.glob(exp_dirs)
        for i, p in enumerate(paths):
            val_dict = dict()
            if dataset == 'davis':
                seq = p.split('/')[-2][8:]
                run_name = '/'.join(p.split('/')[-2:])
            elif dataset == 'davis_splatam':
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

            # if 'horse' not in seq:
            #     continue

            print(f"\mSEQUENCE {seq}...")
            
            # copy config and get create runname
            seq_experiment = SourceFileLoader(
                os.path.basename(args.experiment), args.experiment
                # 'config.py', os.path.join(p, 'config.py')
                ).load_module()

            seq_experiment.config['run_name'] = run_name
            seq_experiment.config['data']['sequence'] = seq
            seq_experiment.config['primary_device'] = primary_device

            if "120_227" in experiment:
                seq_experiment.config['data']['desired_image_height'] = 120
                seq_experiment.config['data']['desired_image_width'] = 227
            elif "480_910" in experiment:
                seq_experiment.config['data']['desired_image_height'] = 480
                seq_experiment.config['data']['desired_image_width'] = 910
            elif "240_455" in experiment:
                seq_experiment.config['data']['desired_image_height'] = 240
                seq_experiment.config['data']['desired_image_width'] = 455
            elif "360_640" in experiment:
                seq_experiment.config['data']['desired_image_height'] = 360
                seq_experiment.config['data']['desired_image_width'] = 640

            if 'transformed' not in experiment:
                do_transform = True
            else:
                do_transform = False

            if args.eval_traj:
                if not os.path.isfile(os.path.join(p, 'eval', 'traj_metrics.txt')) and not "SplaTAM" in experiment:
                    print(f"Experiment not done {run_name} yet.")
                    continue
                elif not os.path.isfile(f"{experiment}/{run_name}/params.npz") and "SplaTAM" in experiment:
                    print(f"Experiment not done {run_name} yet.")
                    continue
                
                if seq_experiment.config['time_window'] == 1:
                    if not get_gauss_wise3D_track:
                        add_on = '_alpha'
                    elif use_round_pix:
                        add_on = '_round'
                    else:
                        add_on = ''
                    
                    if best_x != 1:
                        add_on = add_on + f'_{best_x}'

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
                        traj_len=traj_len,
                        color_thresh=color_thresh,
                        do_transform=do_transform,
                        use_round_pix=use_round_pix,
                        get_gauss_wise3D_track=get_gauss_wise3D_track,
                        get_from3D=get_from3D
                        )
                    print(metrics)
                    quit()
                    with open(os.path.join(p, 'eval', f'traj_metrics{add_on}.txt'), 'w') as f:
                        f.write(f"Trajectory metrics: {metrics}")

                    if vis_gird:
                        vis_grid_trajs(
                            seq_experiment.config,
                            params=None,
                            cam=None,
                            results_dir=p,
                            orig_image_size=True,
                            no_bg=True,
                            clip=True,
                            traj_len=traj_len,
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
                        traj_len=traj_len,
                        color_thresh=color_thresh)
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

                if 'transformed' in experiment:
                    seq_experiment.config['data']['do_transform'] = True
 
                rgbd_slammer = RBDG_SLAMMER(seq_experiment.config)
                rgbd_slammer.eval()
            
            if args.eval_novel_view:
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

                if 'transformed' in experiment:
                    seq_experiment.config['data']['do_transform'] = True
 
                rgbd_slammer = RBDG_SLAMMER(seq_experiment.config)
                rgbd_slammer.render_novel_view(novel_view_mode=novel_view_mode)
