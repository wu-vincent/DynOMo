import os
import glob
import sys 
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.pyplot as plt
import numpy as np
import json


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

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


def get_jono_base_exps():
    experiments = list()
    names = list()
    experiments += ["experiments/output_orig_jono"]
    names.append("original_jono")
    experiments += ["experiments/output_one_cam_jono"]
    names.append("jono_one_cam")
    experiments += ["experiments/output_one_cam_jono_fewer_epochs"]
    names.append("jono_one_cam_fewer_epochs")
    experiments += ["experiments/output_stereo"]
    names.append("jono_two_cam")

    return experiments, names

def get_davis_exps():
    experiments = list()
    names = list()

    # per point 
    experiments += ["0_per_point_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_False_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb"]
    names.append('less_16_128_16_20_20_5_bug_rem_per_point_no_dyno_weight')
    # fix cam
    experiments += ["0_kNN_200_200_0_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append('less_16_128_16_20_20_5_bug_rem_fix_cam')
    # per point 
    experiments += ["0_per_point_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append('less_16_128_16_20_20_5_bug_rem_per_point')
    # l2 weighting knn 
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3_l2_knn"]
    names.append('less_16_128_16_20_20_5_bug_rem_kNN_l2')

    # more ablations
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_nohook"]
    names.append('r3_nohook')
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_False_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append("r3_no_early_stop")
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_480_910_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append("r3_orig_size")
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_False_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append("r3_no_restart_if_fail")
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_480_910_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3_opa"]
    names.append("r3_opa_lr_higher")

    experiments = list()
    names = list()
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append('r3')
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_480_910_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3_opa_scale"]
    names.append("r3_opa_scale_lr_higher")
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_1_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append('r3_stride1')
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3_scale"]
    names.append('r3_scale')
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_0_0_0_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3"]
    names.append('r3_no_fixing_no_l1')
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_20_20_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_20_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3_l1_all"]
    names.append('r3_no_fixing_all_l1')
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_0_0_5_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3_fix_all"]
    names.append('r3_all_fixing_no_l1')
    experiments += ["0_kNN_200_200_200_-1_32_False_0.5_0_0_0_0.001_True_True_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_240_455_0.0_0.0_0_False_False_True_False_True_0.1_3_aniso_deb_l2_emb_r3_fix_all"]
    names.append('r3_all_fixing_no_l1_2')

    return experiments, names


def get_jono_exps():
    experiments = list()
    names = list()
    
    #### higher rigidity ####
    # jono depth
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_180_320_0.0_0.0_transformed"]
    names.append('not_higher_lr_normal_size_transformed_jono_depth')
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_transformed"]
    names.append('not_higher_lr_orig_size_transformed_jono_depth')

    # orig size
    experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_transformed"]
    names.append('not_higher_lr_original_size_transformed')

    # normal size
    experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_180_320_0.0_0.0_transformed"]
    names.append('not_higher_lr_normal_size_transformed')

    # jono pc
    experiments += ["0_kNN_500_1000_0_-1_32_True_True_False_0.5_20_20_5_0.001_False_False_False_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_180_320_0.0_0.0_transformed"]
    names.append('not_higher_lr_normal_size_jono_pc')
    experiments += ["0_kNN_500_1000_0_-1_32_True_True_False_0.5_20_20_5_0.001_False_False_False_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_transformed"]
    names.append('not_higher_lr_normal_size_jono_pc_orig_size')

    # transfor check
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_False_False_transformed_check"]
    names.append('CHECK_not_higher_lr_normal_size_jono_depth')
    experiments += ["0_kNN_500_1000_0_-1_32_True_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_False_False_transformed_check"]
    names.append('CHECK_not_higher_lr_normal_size_jono_pc')
    
    # opa exps
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_True_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_1_1_0_1_20_360_640_0.0_0.0_False_False_transformed_check_r3"]
    names.append('not_higher_lr_orig_size_transformed_jono_depth_higher_opa_lr_stride1')
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_True_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_1_1_0_1_20_360_640_0.0_0.0_False_False_transformed_check_r3_opa"]
    names.append('not_higher_lr_orig_size_transformed_jono_depth_higher_opa_lr')

    experiments = list()
    names = list()
    # Different exps
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_transformed"]
    names.append('not_higher_lr_orig_size_transformed_jono_depth')
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_True_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_False_False_transformed_check_r3_scales_opa"]
    names.append('not_higher_lr_orig_size_transformed_jono_depth_higher_opa_scales_lr')
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_True_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_2_1_0_1_20_360_640_0.0_0.0_False_False_transformed_check_r3_scales"]
    names.append('not_higher_lr_orig_size_transformed_jono_depth_scales')
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_20_20_5_0.001_False_False_True_True_True_2000_16_16_128_16_0.1_True_False_False_True_True_1_1_0_1_20_360_640_0.0_0.0_False_False_transformed_check_r3_scales"]
    names.append('not_higher_lr_orig_size_transformed_jono_depth_stride1')

    return experiments, names

def get_splatam_exps():
    experiments = list()
    names = list()
    experiments += ["../SplaTAM/experiments/dynosplatam_davis_init_params"]
    names += ["initial_splatam"]
    experiments += ["../SplaTAM/experiments/dynosplatam_davis_longer"]
    names += ["initial_splatam_longer"]
    return experiments, names

def get_iphone_exps():
    experiments = list()
    names = list()

    experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_1_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw_deb_scales"]
    names += ["orig_s1_scales"]
    experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    names += ["orig"]
    experiments += ["0_kNN_200_200_200_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_1_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    names += ["camera_optim_s1"]
    experiments += ["0_kNN_200_200_0_20_20_30_0.001_False_False_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    names += ["strong_bg_reg"]
    experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_0.9_bug_rem_hw_deb"]
    names += ["sil_0.9"]
    experiments += ["0_kNN_200_200_0_20_20_5_0.001_True_True_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_0.5_bug_rem_hw_deb"]
    names += ["higher_lr"]
    experiments += ["0_kNN_200_200_200_20_20_5_0.001_False_True_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_0.5_bug_rem_hw_deb"]
    names += ["cam_optim_s2_higher_cam_lr"]
    experiments += ["0_kNN_200_200_200_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_2_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    names += ["cam_optim_s2"]
    experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_1_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    names += ["orig_s1"]
    experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_1_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_lidar_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    names += ["orig_lidar_s1"]
    experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_1_0_20_0.5_0.5_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_original_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    names += ["orig_orig_pose_s1"]
    experiments += ["0_kNN_200_200_0_20_20_5_0.001_False_False_True_True_True_16_16_128_16_0.1_True_2_0_20_1.0_1.0_False_False_False_0.1_3_aniso_deb_l2_emb_aligned_depth_anything_colmap_refined_segment_bug_rem_factor_False_bug_rem_hw_deb"]
    names += ["orig_s2_orig_size"]

    return experiments, names


if __name__ == "__main__":
    files_to_load = ["traj_metrics.txt", "psnr.txt", "ssim.txt", "lpips.txt"]

    get_gauss_wise3D_track = True
    use_round_pix = False
    get_from3D = False
    queries_first_t = False
    best_x = 1
    vis_thresh = 0.5

    dataset = 'davis'
    # dataset = 'davis_splatam'
    # dataset = 'jono'
    # dataset = 'jono_baseline'
    dataset = "iphone"

    if dataset == 'jono_baseline':
        experiments, names = get_jono_base_exps()
    elif dataset == 'jono':
        experiments, names = get_jono_exps()
    elif dataset == 'davis':
        experiments, names = get_davis_exps()
    elif dataset == 'davis_splatam':
        experiments, names = get_splatam_exps()
    elif dataset == 'iphone':
        experiments, names = get_iphone_exps()

    base_exp = "not_higher_lr" # "original_jono" # "less_ep_16_16_16" # "scale_fix"

    if dataset != 'iphone':
        overall_df_mte = pd.DataFrame(columns=davis_seqs_1)
        overall_df_sur = pd.DataFrame(columns=davis_seqs_1)
        overall_df_d_avg = pd.DataFrame(columns=davis_seqs_1)
        overall_df_aj = pd.DataFrame(columns=davis_seqs_1)
        overall_df_oa = pd.DataFrame(columns=davis_seqs_1)
    
    if dataset == 'iphone':
        df_mean = pd.DataFrame(columns=['duration [min]', 'FPS', 'AJ', 'APCK', 'occ_acc', 'epe', 'pck_3d_50cm', 'pck_3d_10cm', 'pck_3d_5cm', 'psnr', 'ssim', 'lpips'])
    elif dataset != 'jono' and dataset != "jono_baseline":
        df_mean = pd.DataFrame(columns=['duration [min]', 'FPS', 'median_l2', 'survival', 'd_avg', 'occlusion_accuracy', 'pts_within_1', 'jaccard_1', 'pts_within_2', 'jaccard_2', 'pts_within_4', 'jaccard_4', 'pts_within_8', 'jaccard_8', 'pts_within_16', 'jaccard_16', 'average_jaccard', 'average_pts_within_thresh', 'psnr', 'ssim', 'lpips'])
    else:
        df_mean = pd.DataFrame(columns=['duration [min]', 'FPS', 'median_l2', 'survival', 'd_avg', 'median_l2_3D', 'survival_3D', 'd_avg_3D', 'occlusion_accuracy', 'pts_within_1', 'jaccard_1', 'pts_within_2', 'jaccard_2', 'pts_within_4', 'jaccard_4', 'pts_within_8', 'jaccard_8', 'pts_within_16', 'jaccard_16', 'average_jaccard', 'average_pts_within_thresh', 'psnr', 'ssim', 'lpips'])
        

    print(f"Evaluating gauss-wise-track {get_gauss_wise3D_track} and round pixel {use_round_pix}.")

    for name, experiment in zip(names, experiments):
        if dataset == 'davis' and not 'SplaTAM' in experiment:
            exp_dirs = f"experiments/dynosplatam_davis/*/*_{experiment}/eval"
        elif dataset == 'davis_splatam':
            exp_dirs = f"{experiment}/*/eval"
        elif dataset == "jono":
            exp_dirs = f"experiments/dynosplatam_jono/*/ims/*_{experiment}/eval"
        elif dataset == 'iphone':
            exp_dirs = f"experiments/dynosplatam_iphone/*/*_{experiment}/eval"
        else:
            exp_dirs = f"../Dynamic3DGaussians/{experiment}/exp1/*/eval"

        print(f"\nSummarazing experiment {name}: {experiment}")
        if dataset == 'iphone':
            df_display = pd.DataFrame(columns=['duration [min]', 'FPS', 'AJ', 'APCK', 'occ_acc', 'epe', 'pck_3d_50cm', 'pck_3d_10cm', 'pck_3d_5cm', 'psnr', 'ssim', 'lpips'])
            df = pd.DataFrame(columns=['duration [min]', 'FPS', 'AJ', 'APCK', 'occ_acc', 'epe', 'pck_3d_50cm', 'pck_3d_10cm', 'pck_3d_5cm', 'psnr', 'ssim', 'lpips'], index=davis_seqs_1).fillna(0).astype('float64')
        elif dataset != 'jono' and dataset != "jono_baseline":
            df_display = pd.DataFrame(columns=['duration [min]', 'FPS', 'median_l2', 'survival', 'd_avg', 'occlusion_accuracy', 'pts_within_1', 'jaccard_1', 'pts_within_2', 'jaccard_2', 'pts_within_4', 'jaccard_4', 'pts_within_8', 'jaccard_8', 'pts_within_16', 'jaccard_16', 'average_jaccard', 'average_pts_within_thresh', 'psnr', 'ssim', 'lpips'])
            df = pd.DataFrame(columns=['duration [min]', 'FPS', 'median_l2', 'survival', 'd_avg', 'occlusion_accuracy', 'pts_within_1', 'jaccard_1', 'pts_within_2', 'jaccard_2', 'pts_within_4', 'jaccard_4', 'pts_within_8', 'jaccard_8', 'pts_within_16', 'jaccard_16', 'average_jaccard', 'average_pts_within_thresh', 'psnr', 'ssim', 'lpips'], index=davis_seqs_1).fillna(0).astype('float64')
        else:
            df_display = pd.DataFrame(columns=['duration [min]', 'FPS', 'median_l2', 'survival', 'd_avg', 'median_l2_3D', 'survival_3D', 'd_avg_3D', 'occlusion_accuracy', 'pts_within_1', 'jaccard_1', 'pts_within_2', 'jaccard_2', 'pts_within_4', 'jaccard_4', 'pts_within_8', 'jaccard_8', 'pts_within_16', 'jaccard_16', 'average_jaccard', 'average_pts_within_thresh', 'psnr', 'ssim', 'lpips'])
            df = pd.DataFrame(columns=['duration [min]', 'FPS', 'median_l2', 'survival', 'd_avg', 'median_l2_3D', 'survival_3D', 'd_avg_3D', 'occlusion_accuracy', 'pts_within_1', 'jaccard_1', 'pts_within_2', 'jaccard_2', 'pts_within_4', 'jaccard_4', 'pts_within_8', 'jaccard_8', 'pts_within_16', 'jaccard_16', 'average_jaccard', 'average_pts_within_thresh', 'psnr', 'ssim', 'lpips'], index=davis_seqs_1).fillna(0).astype('float64')

        paths = glob.glob(exp_dirs)
        for i, p in enumerate(paths):
            val_dict = dict()
            if dataset == 'davis' and not 'SplaTAM' in experiment:
                seq = p.split('/')[-3]
            elif dataset == 'davis_splatam':
                seq = p.split('/')[-2][:-2]
            elif dataset == "jono":
                seq = p.split('/')[-4]
            else:
                seq = p.split('/')[-2]

            # if seq not in davis_seqs_1:
            #     continue

            # if "judo" not in seq:
            #     continue

            for file_name in files_to_load:

                # different choices of Gaussians for evaluation of trajectoy
                if not get_gauss_wise3D_track and (not get_from3D or 'jono' not in dataset):
                    add_on = '_alpha'
                elif get_from3D:
                    add_on = '_from_3D'
                elif use_round_pix:
                    add_on = '_round'
                else:
                    add_on = ''
                
                if best_x != 1:
                    add_on = add_on + f'_{best_x}'
                
                if not queries_first_t:
                    add_on = add_on + '_not_only_first'
                
                if vis_thresh != 0.5:
                    add_on = add_on + f'_{vis_thresh}'

                if not os.path.isfile(os.path.join(p, f'{file_name[:-4]}{add_on}.txt')):
                    break
                
                if os.path.isfile(os.path.join(p, f'{file_name[:-4]}{add_on}.json')):
                    with open(os.path.join(p, f'{file_name[:-4]}{add_on}.json'), 'r') as jf:
                        val_dict = json.load(jf)
                    if dataset != 'iphone':
                        val_dict = {k1: v1 for k, v in val_dict.items() for k1, v1 in val_dict[k].items() if k1 in df.columns}
                else:
                    with open(os.path.join(p, f'{file_name[:-4]}{add_on}.txt'), 'r') as f:
                        data = f.read()
                        if file_name == "traj_metrics.txt":
                            data = data.strip("Trajectory metrics: ")
                            data = data.split(',')
                            for val in ['AJ\'', 'APCK\'', 'occ_acc\'', 'epe\'', 'pck_3d_50cm\'', 'pck_3d_10cm\'', 'pck_3d_5cm\'']:
                                for d in data:
                                    if val in d:
                                        value = float(d.split(': ')[-1].strip('}'))
                                        val_dict[val[:-1]] = value

                            for val in ['median_l2\'', 'survival\'', 'd_avg\'', 'median_l2_3D\'', 'survival_3D\'', 'd_avg_3D\'']:
                                for d in data:
                                    if val in d:
                                        value = float(d.split(': ')[-1].strip('}'))
                                        val_dict[val[:-1]] = value
                            for val in ["occlusion_accuracy\'", 'pts_within_1\'', 'jaccard_1\'', 'pts_within_2\'', 'jaccard_2\'', 'pts_within_4\'', 'jaccard_4\'', 'pts_within_8\'', 'jaccard_8\'', 'pts_within_16\'', 'jaccard_16\'', 'average_jaccard\'', 'average_pts_within_thresh\'']:
                                for d in data:
                                    if val in d:
                                        value = float(d.split('[')[-1].split(']')[0])
                                        val_dict[val[:-1]] = value
                        else:
                            val = sum([float(d) for d in data.strip('\n').split('\n')])/len([float(d) for d in data.strip('\n').split('\n')])
                            val_dict[file_name[:-4]] = val
        
            if os.path.isfile(os.path.join(os.path.dirname(p), 'params.npz')):
                params = np.load(os.path.join(os.path.dirname(p), 'params.npz'))
            else:
                params = dict()

            if 'duration' in params.keys():
                val_dict['FPS']  = (params['duration'].item()) # +2.3)
                val_dict['duration [min]']  = params['overall_duration'].item() / 60
            else:
                val_dict['FPS']  = 0 
                val_dict['duration [min]']  =  -1
            
            df.loc[seq] = val_dict
            df_display.loc[seq] = val_dict
        
        # df = df.sort_index()
        df.loc['mean'] = df.mean()
        df_display = df_display.sort_index()
        df_display.loc['mean'] = df_display.mean()
        print(df_display)
        print(df_display.shape)

        if 'SplaTAM' not in experiment:
            os.makedirs(f'experiments_eval/{dataset}', exist_ok=True)
            df.to_csv(f'experiments_eval/{dataset}/{os.path.basename(experiment)}.csv')
        else:
            df.to_csv(f'{experiment}.csv')
        
        df_mean.loc[name] = df_display.loc['mean']
        
        if dataset != 'iphone':
            overall_df_mte.loc[name] = df['median_l2'].transpose()
            overall_df_sur.loc[name] = df['survival'].transpose()
            overall_df_d_avg.loc[name] = df['d_avg'].transpose()
            overall_df_aj.loc[name] = df['average_jaccard'].transpose()
            overall_df_oa.loc[name] = df['occlusion_accuracy'].transpose()

    print(df_mean)

    if dataset != 'iphone' and base_exp in overall_df_aj.index:
        overall_df_aj = overall_df_aj-overall_df_aj.loc[base_exp]
        overall_df_sur = overall_df_sur-overall_df_sur.loc[base_exp]
        overall_df_d_avg = overall_df_d_avg-overall_df_d_avg.loc[base_exp]
        overall_df_aj = overall_df_aj-overall_df_aj.loc[base_exp]
        overall_df_oa = overall_df_oa-overall_df_oa.loc[base_exp]

        os.makedirs('summ_res', exist_ok=True)

        plot = overall_df_mte.transpose().plot(title="MTE")
        plt.xticks(ticks=list(range(len(davis_seqs_1))), labels=davis_seqs_1, rotation=90)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('summ_res/mte.png', bbox_inches='tight')

        plot = overall_df_sur.transpose().plot(title="Sur")
        plt.xticks(ticks=list(range(len(davis_seqs_1))), labels=davis_seqs_1, rotation=90)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('summ_res/survival.png', bbox_inches='tight')

        plot = overall_df_d_avg.transpose().plot(title="d_avg")
        plt.xticks(ticks=list(range(len(davis_seqs_1))), labels=davis_seqs_1, rotation=90)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('summ_res/d_avg.png', bbox_inches='tight')

        plot = overall_df_aj.transpose().plot(title="AJ")
        plt.xticks(ticks=list(range(len(davis_seqs_1))), labels=davis_seqs_1, rotation=90)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('summ_res/average_jaccard.png', bbox_inches='tight')

        plot = overall_df_oa.transpose().plot(title="OA")
        plt.xticks(ticks=list(range(len(davis_seqs_1))), labels=davis_seqs_1, rotation=90)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('summ_res/occlusion_accuracy.png', bbox_inches='tight')

        
