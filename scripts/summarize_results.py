import os
import glob
import sys 
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)


if __name__ == "__main__":
    files_to_load = ["traj_metrics.txt"] #, "psnr.txt", "ssim.txt", "lpips.txt"

    dataset = 'jono'
    experiments = list()
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True"]  # mag iso longer
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_True_True_True_0.95"] # mag iso longer jono depth
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True_False_0.5"] # mag iso longer no depth rem, less add 
    # experiments += ["0_kNN_100_100_0_-1_32_False_True_False_True_False_0.5_20_20"] # mag iso no depth rem, less add l1 losses + bg reg
    # experiments += ["0_kNN_100_100_0_-1_32_False_True_False_True_True_0.95_20_20"] # mag iso l1 losses + bg reg
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True_True_0.95_0_20"] # mag iso longer bg reg
    # experiments += ["0_kNN_100_100_0_-1_32_False_True_False_True_True_0.95_0_20"] # mag iso bg reg 
    # experiments += ["0_kNN_100_100_0_-1_32_False_True_False_True_True_0.95_0_20_0.001"] # mag iso bg reg embeddings
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True_True_0.5_0_0_0"] # mag iso longer add less
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True_True_0.95_0_0_0.001"] # mag iso longer embeddings
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True_False_0.95_0_0_0"] # mag iso no add
    # experiments += ['0_kNN_500_1000_0_-1_32_False_True_False_True_True_0.5_0_0_0.001'] # mag iso longer add less embeddings
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True_True_0.5_0_20_0.001"] # mag iso longer add less embeddings bg reg
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True_True_0.5_20_0_0.001"] # mag iso longer add less embeddings l1 on rgb and embeddings
    # experiments += ["0_kNN_100_100_0_-1_32_False_True_False_True_True_0.5_0_0_0"] # mag iso add less opa/means/rot/cam/embeddings higher lr
    # experiments += ["0_kNN_100_100_0_-1_32_False_True_False_True_True_0.5_0_0_0.001"] # mag iso add less embeddings opa/means/rot/cam/embeddings higher lr
    # experiments += ["0_kNN_100_500_0_-1_32_False_True_False_True_True_0.5_0_0_0.001"] # 500 init mag iso add less embeddings opa/means/rot/cam/embeddings higher lr
    # experiments += ["0_kNN_100_500_0_-1_32_False_True_False_True_True_0.5_0_0_0"] # 500 init mag iso add less opa/means/rot/cam
    # experiments += ["0_kNN_500_1000_500_-1_32_False_True_False_True_False_0.5_0_0_0.001_False"] # mag iso longer add less embeddings no rem
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_True_False_0.5_0_0_0.001_False"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_True_True_True_False_0.5_0_0_0.001_False"] # jono depth latest setting
    # experiments += ["0_kNN_500_1000_500_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_False_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_False_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_1_1_5_1"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_0_1"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_0_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_0_1"]

    ### BEST SO FAR
    experiments += ["0_kNN_500_1000_0_-1_32_False_False_False_0.5_5_5_0.001_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    experiments += ["0_kNN_500_1000_0_-1_32_False_True_False_0.5_5_5_0.001_False_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    experiments += ["0_kNN_500_1000_0_-1_32_True_True_False_0.5_5_5_5_0.001_False_False_False_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]

    dataset = 'davis'
    experiments = list()
    # experiments += ["0_kNN_100_500_100_-1_32_True_True_True_0.95"] # mag iso
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_True_0.95"] # mag iso longer
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_False_0.5"] # mag iso longer, add less, no rem
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_False_0.5_0_0_0_initcam"] # mag iso longer, add less, no rem, no cam forward
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_True_0.5_0_0_0_initcam"] # mag iso longer, add less, rem, no cam forward
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_True_0.95_0_0_0_initcam"] # mag iso longer, no cam forward --> STOPEED
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_True_0.5_0_0_0.001_initcam"] # mag iso longer, add less, rem, no cam forward, embeddings
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_True_0.5_0_0_0_True_initcam"] # mag iso add less opa/means/rot/cam higher lr
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0_True_initcam"] # mag iso add less no rem opa/means/rot/cam higher lr
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_True_0.5_0_0_0.001_True_initcam"] # mag iso add less embeddings opa/means/rot/cam/embeddings higher lr
    # experiments += ["0_kNN_100_500_100_-1_32_True_True_True_0.5_0_0_0.001_True_initcam"] # mag iso add less embeddings opa/means/rot/cam/embeddings higher lr 500 init
    # experiments += ["0_kNN_100_500_100_-1_32_True_True_True_0.5_0_0_0.001_True_True_initcam"] # mag iso add less embeddings opa/means/rot/cam/embeddings higher lr 500 init grad instead of 0
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_False_initcam"] # mag iso add less no rem embeddings opa/means/rot/cam/embeddings higher lr red
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_True_initcam"] # mag iso add less no rem embeddings opa/means/rot/cam/embeddings higher lr red grad instead of 0
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_0_1"]  # mag iso add less no rem embeddings opa/means/rot/cam/embeddings higher lr red outlier_rem
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_False_True"] # l1 loss embeddings 5
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_True_False"] # use embeddings but with cosine distance and exp weight and no seg masks for kNN
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_True_True"] # use embeddings but with cosine distance and exp weight
    # experiments += ["0_kNN_100_100_0_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_False_True"] # no cam forwars
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_True_True_True_2000"] # cosine distance as distance
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_0_0_0.001_True_0_100_True_True_True_2000_16_0.1_True"]  # cosine distance as distance, cosine similarity as weight, iso weight 16, use weight for iso, cam prop, 0.1 depth weight cam
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_False"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_True"]
    # experiments += ["0_kNN_100_100_0_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_False"]
    # experiments += ["0_kNN_100_100_0_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_True"]
    # experiments += ["0_kNN_100_100_0_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_True_no_sc"]

    # experiments += ["0_kNN_100_100_0_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_False_False_True_False"]
    # experiments += ["0_kNN_100_100_0_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_False_True"]
     #experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True"]

    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_no_sc"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_0_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_no_sc_rgb"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_5_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_no_sc_rgb"]

    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_5_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_all"]
    # experiments += ["0_kNN_100_100_100_-1_32_True_True_False_0.5_5_5_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_all_updaterestart"]
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_False_0.5_5_5_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_all_updaterestart"]
    # experiments += ["0_kNN_500_1000_500_-1_32_True_True_False_0.5_5_5_0.001_True_0_100_True_True_True_2000_16_0.1_True_False_False_True_all_updaterestart_othermask"]
    
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_100000_5"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_8"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_8_5"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_1_1_5"]

    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5"]
    # experiments += ["0_kNN_500_1000_0_-1_32_False_0.5_5_0_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_False_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    
    experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_10_1"]

    # experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_0.001_True_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1"]
    experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1_aniso"]
    experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_0_4_4_0.1_True_False_False_True_True_2_1_5_1_20_240_455_aniso"]
    experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_0_0_4_0.1_True_False_False_True_True_2_1_5_1_aniso"]
    experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_0_0_0_0.1_True_False_False_True_True_2_1_5_1_aniso"]
    experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_4_4_0.1_True_False_False_True_True_2_1_5_1_60_240_455_aniso"]
    experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_4_4_0.1_True_False_False_True_True_2_1_5_1_60_480_910_aniso"]
    experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_4_4_0.1_True_False_False_True_True_2_1_5_1_20_120_227_aniso"]
    experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_4_0.1_True_False_False_True_True_2_1_5_1_20_240_455_aniso"]
    experiments += ["0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_16_16_0.1_True_False_False_True_True_2_1_5_1_20_240_455_aniso"]


    # experiments += ["../SplaTAM/experiments/dynosplatam_davis_init_params"]

    # dataset = 'jono_baseline'
    # experiments = list()
    # experiments += ["output_one_cam_jono"]
    # experiments += ["output_all_cam_fewer_epochs"]
    # experiments += ["output_orig"]
    # experiments += ["output"]


    davis_seqs_1 = [
        'splatam_motocross-jump',
        'splatam_goat',
        'splatam_car-roundabout',
        'splatam_breakdance',
        'splatam_drift-chicane',
        'splatam_drift-straight',
        'splatam_judo',
        'splatam_soapbox',
        'splatam_dogs-jump',
        'splatam_parkour',
        'splatam_india',
        'splatam_pigs',
        'splatam_cows',
        'splatam_gold-fish',
        'splatam_paragliding-launch',
        'splatam_camel',
        'splatam_blackswan',
        # 'splatam_dog',
        'splatam_bike-packing',
        'splatam_shooting',
        'splatam_lab-coat',
        # 'splatam_kite-surf',
        'splatam_bmx-trees',
        'splatam_dance-twirl',
        'splatam_car-shadow',
        # 'splatam_libby',
        'splatam_scooter-black',
        # 'splatam_mbike-trick',
        'splatam_loading',
        # 'splatam_horsejump-high'
        ]

    for experiment in experiments:
        if dataset == 'davis' and not 'SplaTAM' in experiment:
            exp_dirs = f"experiments/dynosplatam_davis/*/*_{experiment}/eval"
        elif dataset == 'davis':
            exp_dirs = f"../SplaTAM/experiments/dynosplatam_davis/*/eval"
        elif dataset == "jono":
            exp_dirs = f"experiments/dynosplatam_jono/*/ims/*_{experiment}/eval"
        else:
            exp_dirs = f"../Dynamic3DGaussians/{experiment}/exp1/*/eval"

        print(f"\nSummarazing experiment {experiment}")
        df = pd.DataFrame(columns=['median_l2', 'survival', 'd_avg', 'average_pts_within_thresh', 'average_jaccard', 'occlusion_accuracy'])

        paths = glob.glob(exp_dirs)
        for i, p in enumerate(paths):
            val_dict = dict()
            if dataset == 'davis' and not 'SplaTAM' in experiment:
                seq = p.split('/')[-3]
            elif dataset == 'davis':
                seq = p.split('/')[-2][:-2]
            elif dataset == "jono":
                seq = p.split('/')[-4]
            else:
                seq = p.split('/')[-2]

            # if seq not in davis_seqs_1:
            #     continue

            # if "shooting" in seq or 'parkour' in seq:
            #     continue

            for file_name in files_to_load:
                if not os.path.isfile(os.path.join(p, file_name)):
                    break
                with open(os.path.join(p, file_name), 'r') as f:
                    data = f.read()
                    if file_name == "traj_metrics.txt":
                        data = data.strip("Trajectory metrics: ")
                        data = data.split(',')
                        for val in ['median_l2', 'survival', 'd_avg']:
                            for d in data:
                                if val in d:
                                    value = float(d.split(': ')[-1].strip('}'))
                                    val_dict[val] = value

                        for val in ['average_pts_within_thresh', 'average_jaccard', 'occlusion_accuracy']:
                            for d in data:
                                if val in d:
                                    value = float(d.split('[')[-1].split(']')[0])
                                    val_dict[val] = value
                    df.loc[seq] = val_dict
        df = df.sort_index()
        df.loc['mean'] = df.mean()
        print(df)
        print(df.shape)
        if 'SplaTAM' not in experiment:        
            os.makedirs(f'experiments_eval/{dataset}', exist_ok=True)
            df.to_csv(f'experiments_eval/{dataset}/{experiment}.csv')
        else:
            df.to_csv(f'{experiment}.csv')

    
