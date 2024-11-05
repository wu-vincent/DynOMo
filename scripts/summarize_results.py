import os
import glob
import sys 
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="which dataset to summarize", choices=['davis', 'panoptic_sports', 'iphone'])
    parser.add_argument("--get_gauss_wise3D_track", action='store_false', help="if using one gaussian only or using alpha composition")
    parser.add_argument("--get_from3D", action='store_true', help="if searching in 3D instead of 2D")
    parser.add_argument("--queries_first_t", action='store_true', help="if queries only from first frae")
    parser.add_argument("--best_x", defualt=1, type=int, help="take best of x Gaussians")
    args = parser.parse_args()

    files_to_load = ["traj_metrics.txt", "psnr.txt", "ssim.txt", "lpips.txt"]

    get_gauss_wise3D_track = args.get_gauss_wise3D_track
    get_from3D = args.get_from3D
    queries_first_t = True if args.dataset != 'iphone' else args.queries_first_t
    best_x = args.best_x
    
    if args.dataset == 'iphone':
        df_mean = pd.DataFrame(columns=['duration [min]', 'FPS', 'AJ', 'APCK', 'occ_acc', 'epe', 'pck_3d_50cm', 'pck_3d_10cm', 'pck_3d_5cm', 'psnr', 'ssim', 'lpips'])
        base_exp_dirs = f"experiments/dynosplatam_iphone/*/*/eval"
    elif args.dataset == 'davis':
        df_mean = pd.DataFrame(columns=['duration [min]', 'FPS', 'median_l2', 'survival', 'd_avg', 'occlusion_accuracy', 'pts_within_1', 'jaccard_1', 'pts_within_2', 'jaccard_2', 'pts_within_4', 'jaccard_4', 'pts_within_8', 'jaccard_8', 'pts_within_16', 'jaccard_16', 'average_jaccard', 'average_pts_within_thresh', 'psnr', 'ssim', 'lpips'])
        base_exp_dirs = glob.glob(f"experiments/dynosplatam_davis/*/*/eval")
    else:
        df_mean = pd.DataFrame(columns=['duration [min]', 'FPS', 'median_l2', 'survival', 'd_avg', 'median_l2_3D', 'survival_3D', 'd_avg_3D', 'occlusion_accuracy', 'pts_within_1', 'jaccard_1', 'pts_within_2', 'jaccard_2', 'pts_within_4', 'jaccard_4', 'pts_within_8', 'jaccard_8', 'pts_within_16', 'jaccard_16', 'average_jaccard', 'average_pts_within_thresh', 'psnr', 'ssim', 'lpips'])
        base_exp_dirs = f"experiments/dynosplatam_jono/*/ims/*/eval"
        

    print(f"Evaluating gauss-wise-track {get_gauss_wise3D_track} and round pixel.")

    for exp_dirs in base_exp_dirs:
        print(f"\nSummarazing experiment: {exp_dirs}")
        if args.dataset == 'iphone':
            df_display = pd.DataFrame(columns=['duration [min]', 'FPS', 'AJ', 'APCK', 'occ_acc', 'epe', 'pck_3d_50cm', 'pck_3d_10cm', 'pck_3d_5cm', 'psnr', 'ssim', 'lpips'])
            df = pd.DataFrame(columns=['duration [min]', 'FPS', 'AJ', 'APCK', 'occ_acc', 'epe', 'pck_3d_50cm', 'pck_3d_10cm', 'pck_3d_5cm', 'psnr', 'ssim', 'lpips'], index=davis_seqs_1).fillna(0).astype('float64')
        elif args.dataset == 'davis':
            df_display = pd.DataFrame(columns=['duration [min]', 'FPS', 'median_l2', 'survival', 'd_avg', 'occlusion_accuracy', 'pts_within_1', 'jaccard_1', 'pts_within_2', 'jaccard_2', 'pts_within_4', 'jaccard_4', 'pts_within_8', 'jaccard_8', 'pts_within_16', 'jaccard_16', 'average_jaccard', 'average_pts_within_thresh', 'psnr', 'ssim', 'lpips'])
            df = pd.DataFrame(columns=['duration [min]', 'FPS', 'median_l2', 'survival', 'd_avg', 'occlusion_accuracy', 'pts_within_1', 'jaccard_1', 'pts_within_2', 'jaccard_2', 'pts_within_4', 'jaccard_4', 'pts_within_8', 'jaccard_8', 'pts_within_16', 'jaccard_16', 'average_jaccard', 'average_pts_within_thresh', 'psnr', 'ssim', 'lpips'], index=davis_seqs_1).fillna(0).astype('float64')
        else:
            df_display = pd.DataFrame(columns=['duration [min]', 'FPS', 'median_l2', 'survival', 'd_avg', 'median_l2_3D', 'survival_3D', 'd_avg_3D', 'occlusion_accuracy', 'pts_within_1', 'jaccard_1', 'pts_within_2', 'jaccard_2', 'pts_within_4', 'jaccard_4', 'pts_within_8', 'jaccard_8', 'pts_within_16', 'jaccard_16', 'average_jaccard', 'average_pts_within_thresh', 'psnr', 'ssim', 'lpips'])
            df = pd.DataFrame(columns=['duration [min]', 'FPS', 'median_l2', 'survival', 'd_avg', 'median_l2_3D', 'survival_3D', 'd_avg_3D', 'occlusion_accuracy', 'pts_within_1', 'jaccard_1', 'pts_within_2', 'jaccard_2', 'pts_within_4', 'jaccard_4', 'pts_within_8', 'jaccard_8', 'pts_within_16', 'jaccard_16', 'average_jaccard', 'average_pts_within_thresh', 'psnr', 'ssim', 'lpips'], index=davis_seqs_1).fillna(0).astype('float64')

        paths = glob.glob(exp_dirs)
        for i, p in enumerate(paths):
            val_dict = dict()
            if args.dataset == 'davis':
                seq = p.split('/')[-3]
            elif args.dataset == "panoptic_sports":
                seq = p.split('/')[-4]
            else:
                seq = p.split('/')[-2]

            for file_name in files_to_load:

                # different choices of Gaussians for evaluation of trajectoy
                if not get_gauss_wise3D_track and (not get_from3D or 'panoptic_sports' not in args.dataset):
                    add_on = '_alpha'
                elif get_from3D:
                    add_on = '_from_3D'
                else:
                    add_on = ''
                if best_x != 1:
                    add_on = add_on + f'_{best_x}'
                if not queries_first_t:
                    add_on = add_on + '_not_only_first'

                if not os.path.isfile(os.path.join(p, f'{file_name[:-4]}{add_on}.txt')):
                    break
                
                if os.path.isfile(os.path.join(p, f'{file_name[:-4]}{add_on}.json')):
                    with open(os.path.join(p, f'{file_name[:-4]}{add_on}.json'), 'r') as jf:
                        val_dict = json.load(jf)
                    if args.dataset != 'iphone':
                        val_dict = {k1: v1 for k, v in val_dict.items() for k1, v1 in val_dict[k].items() if k1 in df.columns}
        
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

        os.makedirs(f'experiments_eval/{args.dataset}', exist_ok=True)
        df.to_csv(f'experiments_eval/{args.dataset}/{os.path.basename(exp_dirs)}.csv')
        
        df_mean.loc[exp_dirs] = df_display.loc['mean']

    print(df_mean)

        
