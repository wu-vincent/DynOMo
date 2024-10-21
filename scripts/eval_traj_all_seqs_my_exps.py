import os
import sys 

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import argparse
from importlib.machinery import SourceFileLoader
from utils.common_utils import seed_everything
import os
import shutil
import copy
from utils.eval_traj import eval_traj, vis_grid_trajs
import glob
from scripts.splatam import RBDG_SLAMMER
from utils.eval_traj_restructured import TrajEvaluator
import json


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
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_False_0.1_aniso"]
    # names.append("16_128_16_20_20_5")

    # stride 1 again
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_1_1_0_1_20_240_455_0.0_0.0_0.0_False_False_True_False_False_0.1_aniso_repeat"]
    # names.append("less_16_128_16_20_20_5_1")

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

    #### transformed Gaussians in dyno losses bug removal #####
    # less ep 16_128_16_20_20_5 full res l1
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_480_910_0.0_0.0_0_False_False_True_False_True_0.1_aniso_deb"]
    # names.append('less_16_128_16_20_20_5_full_res_l1')

    # less ep 16_128_16_20_20_5 l1
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_aniso_deb"]
    # names.append('less_16_128_16_20_20_5_bug_rem_l1')

    #less ep 16_128_16_20_20_5 full res l2
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_aniso_deb_l2_emb"]
    # names.append('less_16_128_16_20_20_5_bug_rem')

    # no seg for nn less ep 16_128_16_20_20_5 l2
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_False_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_aniso_deb_l2_emb"]
    # names.append('less_16_128_16_20_20_5_bug_rem_no_seg')

    # no seg no bg loss less ep 16_128_16_20_20_5 l2
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_False_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_0_aniso_deb_l2_emb"]
    # names.append('less_16_128_16_20_20_5_bug_rem_no_seg_rem_no_bbg')

    # no seg no depth loss 
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_False_True_2000_16_16_128_16_0_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0_3_aniso_deb_l2_emb"]
    # names.append('less_16_128_16_20_20_5_bug_rem_no_seg_no_depth')

    # no seg no depth loss after opt forwatd
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_False_True_2000_16_16_128_16_0_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0_3_aniso_deb_l2_emb_forward_after_opt"]
    # names.append('less_16_128_16_20_20_5_bug_rem_no_seg_no_depth_after_opt')

    # no seg loss after opt forwatd
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_False_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_forward_after_opt"]
    # names.append('less_16_128_16_20_20_5_bug_rem_no_seg_after_opt')

    #less ep 16_128_16_20_20_5 l2 repeat
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r2"]
    # names.append('less_16_128_16_20_20_5_bug_rem_repeat')

    # after opt forwatd
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_forward_after_opt_r2"]
    # names.append('less_16_128_16_20_20_5_bug_rem_after_opt')

    # no seg no bg loss less ep 16_128_16_20_20_5 l2 repeat
    # experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_False_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_0_aniso_deb_l2_emb_r2"]
    # names.append('less_16_128_16_20_20_5_bug_no_seg_rem_no_bbg_repeat')

    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3_l2_knn"]
    names.append('kNN_l2')
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_False_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append("r3_no_early_stop")
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_480_910_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append("r3_orig_size")
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_1_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append('r3_stride1')
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append('r3')
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_480_910_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3_opa"]
    names.append("r3_opa_lr_higher")

    
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_480_910_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3_opa_scale"]
    names.append("r3_opa_scale_lr_higher")
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_1_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append('r3_stride1')
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3_scale"]
    names.append('r3_scale')
    experiments = list()
    names = list()
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append('r3')

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

    # jono pc
    experiments += ["0_kNN_500_1000_0_-1_32_True_True_False_0.5_20_20_5_0.001_False_False_False_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_180_320_0.0_0.0_transformed"]
    names.append('not_higher_lr_normal_size_jono_pc')
    experiments += ["0_kNN_500_1000_0_-1_32_True_True_False_0.5_20_20_5_0.001_False_False_False_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_transformed"]
    names.append('not_higher_lr_orig_size_jono_pc_remove_close')

    # transfor check
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_False_False_transformed_check"]
    names.append('CHECK_not_higher_lr_normal_size_jono_depth')
    experiments += ["0_kNN_500_1000_0_-1_32_True_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_False_False_transformed_check"]
    names.append('CHECK_not_higher_lr_normal_size_jono_pc')

    # stereo
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_False_True_transformed_check_add_by_densification"]
    names.append('STEREO_not_higher_lr_normal_size_jono_depth')
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_False_True_transformed_check_add_whole_pc"]
    names.append('STEREO_WHOLE_PC_not_higher_lr_normal_size_jono_depth')

    # knn l2
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_False_True_transformed_check_add_by_densification_r2"]
    names.append('STEREO_not_higher_lr_normal_size_jono_depth_R2')
    experiments += ["0_kNN_200_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_True_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_1_1_0_1_20_360_640_0.0_0.0_False_True_transformed_check_add_by_densification_r3"]
    names.append('STEREO_not_higher_lr_normal_size_jono_depth_SHORT')
    experiments += ["0_kNN_1500_750_0_-1_32_False_True_False_0.5_20_20_5_0.001_True_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_1_1_0_1_20_360_640_0.0_0.0_False_True_transformed_check_add_by_densification_r3"]
    names.append('STEREO_not_higher_lr_normal_size_jono_depth_LONG')

    # opa scales
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_True_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_1_1_0_1_20_360_640_0.0_0.0_False_False_transformed_check_r3"]
    names.append('not_higher_lr_orig_size_transformed_jono_depth_higher_opa_lr')
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_True_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_False_False_transformed_check_r3_scales_opa"]
    names.append('not_higher_lr_orig_size_transformed_jono_depth_higher_opa_scales_lr')
    experiments = list()
    names = list()
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_transformed"]
    names.append('not_higher_lr_orig_size_transformed_jono_depth')
    

    return experiments, names

def get_splatam_exps():
    experiments = list()
    names = list()
    experiments += ["../SplaTAM/experiments/dynosplatam_davis_init_params"]
    names += ["initial_splatam"]
    experiments += ["../SplaTAM/experiments/dynosplatam_davis_longer"]
    names += ["longer_splatam"]

    return experiments, names

def get_rgb_exps():
    experiments = list()
    names = list()
    ### BEST SO FAR
    experiments += ["0_kNN_200_200_0_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_128_128_0.0_0.0_0_False_False_True_False_True_0.1_0_aniso_deb_l2_emb_r2"]
    names.append('frist_trial')
    return experiments, names

def get_iphone_exps():
    experiments = list()
    names = list()
    # experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw"]
    # names += ["aligned_refined_hw_bug_rem_stride2"]
    # experiments += ["0_kNN_200_200_200_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw"]
    # names += ["aligned_cam_hw_bug_rem_stride2"]
    # experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_1_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw"]
    # names += ["aligned_refined_hw_bug_rem_stride1"]
    # experiments += ["0_kNN_200_200_200_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_1_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw"]
    # names += ["aligned_cam_hw_bug_rem_stride1"]

    # experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    # names += ["orig"]
    # experiments += ["0_kNN_200_200_0_20_20_30_0.001_False_False_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    # names += ["strong_bg_reg"]
    # experiments += ["0_kNN_200_200_200_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_1_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    # names += ["camera_optim_s1"]
    # experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_0.9_bug_rem_hw_deb"]
    # names += ["sil_0.9"]
    # experiments += ["0_kNN_200_200_0_20_20_5_0.001_True_True_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_0.5_bug_rem_hw_deb"]
    # names += ["higher_lr"]
    # experiments += ["0_kNN_200_200_200_20_20_5_0.001_False_True_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_0.5_bug_rem_hw_deb"]
    # names += ["cam_optim_s2_higher_cam_lr"]
    # experiments += ["0_kNN_200_200_200_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    # names += ["cam_optim_s2"]
    # experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_1_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    # names += ["orig_s1"]
    # experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_1_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_lidar_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    # names += ["orig_s1_lidar"]
    # experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_1_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_original_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    # names += ["orig_s1_orig_pose"]
    experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_2_0_20_1.0_1.0_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    names += ["orig_s2_orig_size"]

    return experiments, names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--eval_renderings", default=0, type=int, help="if eval renderings")
    parser.add_argument("--eval_traj", default=1, type=int, help="if eval traj")
    parser.add_argument("--vis_trajs", default=0, type=int, help="if eval traj")
    parser.add_argument("--vis_grid", default=0, type=int, help="if eval traj")
    parser.add_argument("--compute_metrics", default=1, type=int, help="if eval traj")
    parser.add_argument("--compute_flow", default=0, type=int, help="if compute flow")
    parser.add_argument("--novel_view_mode", default=None, help="if eval novel view")


    args = parser.parse_args()
    
    dataset = 'davis'
    # dataset = 'davis_splatam'
    dataset = 'jono'
    # dataset = 'jono_baseline'
    # dataset = 'rgb_stacking'
    dataset = 'iphone'

    if dataset == 'jono_baseline':
        experiments, names = get_jono_base_exps()
    elif dataset == 'jono':
        experiments, names = get_jono_exps()
    elif dataset == 'davis':
        experiments, names = get_davis_exps()
    elif dataset == 'davis_splatam':
        experiments, names = get_splatam_exps()
    elif dataset == 'rgb_stacking':
        experiments, names = get_rgb_exps()
    elif dataset == 'iphone':
        experiments, names = get_iphone_exps()

    load_gaussian_tracks = False
    vis_trajs = args.vis_trajs
    vis_gird = args.vis_grid
    compute_flow = args.compute_flow

    get_gauss_wise3D_track = True
    get_from3D = False
    vis_trajs_best_x = False
    queries_first_t = True if dataset != 'iphone' else False
    vis_thresh = 0.5
    
    best_x = 1
    traj_len = 0
    vis_all = True
    vis_gt = True
    primary_device = "cuda:1"
    novel_view_mode = 'circle' # 'circle', 'test_cam'
    stereo = False
    novel_view_mode = None # 'zoom_out'

    print(f"Evaluating gauss-wise-track {get_gauss_wise3D_track} and get from 3D {get_from3D} with vis thresh {vis_thresh}.")

    for name, experiment in zip(names, experiments):
        if dataset == 'davis':
            exp_dirs = f"experiments/dynosplatam_davis/*/*_{experiment}"
        elif dataset == 'rgb_stacking':
            exp_dirs = f"experiments/dynosplatam_rgb/*/*_{experiment}"
        elif dataset == 'davis_splatam':
            exp_dirs = f"{experiment}/*"
        elif dataset == "jono":
            exp_dirs = f"experiments/dynosplatam_jono/*/ims/*_{experiment}"
        elif dataset == 'iphone':
            exp_dirs = f"experiments/dynosplatam_iphone/*/*_{experiment}"
        else:
            exp_dirs = f"../Dynamic3DGaussians/{experiment}/exp1/*"

        print(f"\nEvaluating experiment {name}: {experiment}")
        paths = glob.glob(exp_dirs)
        for i, p in enumerate(paths):
            val_dict = dict()
            seq = p.split('/')[-2][8:]
            run_name = '/'.join(p.split('/')[-2:])

            if dataset == 'davis' or dataset == 'rgb_stacking':
                seq = p.split('/')[-2][8:]
                run_name = '/'.join(p.split('/')[-2:])
            elif dataset == 'davis_splatam':
                seq = p.split('/')[-1][:-2]
                run_name = p.split('/')[-1]
            elif dataset == "jono":
                seq = '/'.join(p.split('/')[-3:])
                seq = seq[8:].split('_')[0]
                run_name = '/'.join(p.split('/')[-3:])
            elif dataset == 'iphone':
                seq = p.split('/')[-1].split('_')[1]
                run_name = p.split('/')[-1]
            else:
                seq = p.split('/')[-1]
                run_name = p.split('/')[-1]

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
            elif "128_128" in experiment:
                seq_experiment.config['data']['desired_image_height'] = 128
                seq_experiment.config['data']['desired_image_width'] = 128

            if 'transformed' not in experiment:
                do_transform = True
            else:
                do_transform = False

            if args.eval_traj:
                if not os.path.isfile(os.path.join(p, 'params.npz')) and not "SplaTAM" in experiment:
                    print(f"Experiment not done {run_name} yet.")
                    continue
                elif not os.path.isfile(f"{experiment}/{run_name}/params.npz") and "SplaTAM" in experiment:
                    print(f"Experiment not done {run_name} yet.")
                    continue

                if not get_gauss_wise3D_track and (not get_from3D or 'jono' not in dataset):
                    add_on = '_alpha'
                elif get_from3D:
                    add_on = '_from_3D'
                else:
                    add_on = ''

                if best_x != 1:
                    add_on = add_on + f'_{best_x}'
                
                if not queries_first_t:
                    add_on = add_on + '_not_only_first'
                
                if vis_thresh != 0.5:
                    add_on = add_on + f'_{vis_thresh}'
                
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
                    stereo=stereo,
                    queries_first_t=queries_first_t,
                    vis_thresh=vis_thresh,
                    vis_thresh_start=vis_thresh,
                    primary_device=primary_device)

                if args.compute_metrics:
                    metrics = evaluator.eval_traj()
                    print("Trajectory metrics:", metrics)
                    with open(os.path.join(p, 'eval', f'traj_metrics{add_on}.json'), 'w') as f:
                        json.dump(metrics, f)
                    with open(os.path.join(p, 'eval', f'traj_metrics{add_on}.txt'), 'w') as f:
                        f.write(f"Trajectory metrics: {metrics}")

                if vis_gird:
                    evaluator.vis_grid_trajs()
                    print(f"Stored visualizations to {p}...")
                
                if compute_flow:
                    evaluator.vis_flow()

            if args.eval_renderings or args.novel_view_mode is not None:
                if not os.path.isfile(os.path.join(p, 'eval', 'traj_metrics.txt')) and not "SplaTAM" in experiment:
                    print(f"Experiment not done {run_name} yet.")
                    continue
                seq_experiment.config['use_wandb'] = False
                seq_experiment.config['checkpoint'] = True
                seq_experiment.config['just_eval'] = True
                seq_experiment.config['viz']['vis_all'] = vis_all if novel_view_mode != '' else False
                seq_experiment.config['viz']['vis_gt'] = vis_gt if novel_view_mode != '' else False
                seq_experiment.config['run_name'] = run_name
                seq_experiment.config['data']['sequence'] = seq

                rgbd_slammer = RBDG_SLAMMER(seq_experiment.config)
                rgbd_slammer.eval(
                    novel_view_mode=args.novel_view_mode,
                    eval_traj=0,
                    vis_trajs=0)
