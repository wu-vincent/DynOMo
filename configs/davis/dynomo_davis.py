import os
from os.path import join as p_join

scenes = ['motocross-jump', 'camel', 'pigs', 'car-roundabout', 'dance-twirl', 'goat', 'breakdance', 'drift-chicane', 'drift-straight', 'judo', 'soapbox', 'dogs-jump', 'parkour', 'india', 'pigs', 'cows', 'gold-fish', 'paragliding-launch', 'blackswan', 'dog', 'bike-packing', 'shooting', 'lab-coat', 'kite-surf', 'bmx-trees', 'car-shadow', 'libby', 'scooter-black', 'mbike-trick', 'loading', 'horsejump-high']

primary_device="cuda:5"
seed = 0
scene_name = scenes[0]

red_lr = False
add_every = 1
tracking_iters = 500 if not red_lr else 100
tracking_iters_init = 1000 if not red_lr else 100
delta_optim_iters = 0
tracking_iters_cam = 500 if not red_lr else 100
refine_iters = 0
num_frames = -1
feature_dim = 32
load_embeddings = True
ssmi_all_mods = False
mag_iso = True
dyno_losses = True
forward_prop = False
mapping_iters = 0
mov_init_by = 'kNN'
l1_losses = 0
bg_reg = 0
embeddings_lr = 0.001
make_grad_bg_smaller = False
trafo_mat = False
# remove_gaussians = False
# sil_thres_gaussians = 0.5

remove_gaussians = False
sil_thres_gaussians = 0.5
remove_gaussians_drift = False
early_stop = False
last_x = 1
densify_post = False
use_sil_for_loss = True if densify_post else False
remove_close = False

remove_outliers_l2 = 1

vis_all = False
gt_w2c = False

group_name = "dynosplatam_davis"
if load_embeddings:
    run_name = f"splatam_{scene_name}/splatam_{scene_name}_{seed}_{mov_init_by}_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}_{num_frames}_{feature_dim}_{dyno_losses}_{mag_iso}_{remove_gaussians}_{sil_thres_gaussians}_{l1_losses}_{bg_reg}_{embeddings_lr}_{red_lr}_{make_grad_bg_smaller}_{remove_outliers_l2}_debug"
else:
    run_name = f"splatam_{scene_name}/splatam_{scene_name}_{seed}_{mov_init_by}_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}_{num_frames}_{dyno_losses}_{mag_iso}_{remove_gaussians}_{sil_thres_gaussians}_{l1_losses}_{bg_reg}_{embeddings_lr}_{red_lr}_{make_grad_bg_smaller}_{remove_outliers_l2}_debug"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    num_threads=1,
    checkpoint=False,
    just_eval=False,
    seed=seed,
    primary_device=primary_device,
    add_every=add_every, # Mapping every nth frame
    eval_every=1, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    gaussian_distribution="anisotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=19, # Checkpoint Interval
    use_wandb=False,
    compute_normals=False,
    mov_static_init=False,
    make_bg_static=False,
    mov_init_by=mov_init_by, #'seg', 'kNN', 'per_point' 'learned_flow
    determine_mov='kNN', #'seg', 'kNN', 'per_point'
    use_seg_for_nn=True,
    mov_thresh=0.01,
    mode='static_dyno', # 'just_static', 'splatam' 'static_dyno'
    zeodepth=False,
    dist_to_use='l2', # 'rgb', 'dinov2', 'l2'
    neighbors_init='post',
    exp_weight=2000,
    eval_during=False,
    motion_lr=0.1,
    early_stop=early_stop,
    stride=1,
    kNN=20,
    norm_embeddings=True,
    remove_close=remove_close,
    ema=0,
    gt_w2c=gt_w2c,
    wandb=dict(
        project="DynoSplaTAM",
        group=group_name,
        name=run_name,
        save_qual=False,    
        eval_save_qual=True,
    ),
    data=dict(
        basedir="/scratch/jseidens/data/DAVIS/JPEGImages/480p",
        gradslam_data_cfg="./configs/data/davis.yaml",
        sequence=scene_name,
        desired_image_height=240, #480,
        desired_image_width=455, #910,
        start=0,
        end=num_frames,
        stride=1,
        num_frames=num_frames,
        load_embeddings=load_embeddings,
        embedding_dim=feature_dim,
        start_from_complete_pc=False,
        novel_view_mode=None
    ),
    add_gaussians=dict(
        add_new_gaussians=True,
        depth_error_factor=5,
        use_depth_error_for_adding_gaussians=False,
        sil_thres_gaussians=sil_thres_gaussians, # For Addition of new Gaussians
    ),
    prune_densify=dict(
        prune_gaussians=False, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=50,
            remove_big_after=0,
            stop_after=100000,
            prune_every=50,
            removal_opacity_threshold=-10,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
            kNN_rel_drift=1000000
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=int(tracking_iters/2), #100,
            grad_thresh=0.1,#0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=-10,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000, # Doesn't consider iter 0
            scale_clone_thresh='scene_radius',
            scale_split_thresh='scene_radius',
            kNN_dist_thresh_min=0.5,
            kNN_dist_thresh_max=0.35
        ),
    ),
    tracking_obj=dict(
        num_iters=tracking_iters,
        num_iters_init=tracking_iters_init,
        sil_thres=0.95,
        use_l1=True,
        use_sil_for_loss=use_sil_for_loss,
        ignore_outlier_depth_loss=False,
        dyno_losses=dyno_losses,
        use_seg_loss=False,
        take_best_candidate=False,
        disable_rgb_grads_old=True,
        make_grad_bg_smaller=make_grad_bg_smaller,
        make_grad_bg_smaller_weight=0, 
        calc_ssmi=True,
        bg_reg=True,
        rgb_attention_bg=False,
        attention_bg='attention',
        rgb_attention_embeddings=False,
        attention='multihead',
        attention_layers=1,
        attention_lrs=0.001,
        use_flow='', # 'rendered' # None
        depth_cam='cam',
        embedding_cam='cam',
        dyno_weight='bg',
        mag_iso=mag_iso,
        weight_rot=True,
        weight_rigid=True,
        weight_iso=False,
        ssmi_all_mods=ssmi_all_mods,
        gt_w2c=gt_w2c,
        loss_weights=dict(
            im=1.0,
            depth=0.1,
            rot=4.0,
            rigid=4.0,
            iso=2.0,
            flow=0.0,
            embeddings=16.0,
            bg_reg=bg_reg,
            bg_loss=3,
            l1_bg=0,
            l1_embeddings=l1_losses,
            l1_scale=l1_losses,
            l1_rgb=l1_losses,
            l1_opacity=l1_losses,
            coeff=1.0,
            instseg=0.0,
            smoothness=0.0
        ),
        lrs=dict(
            means3D=0.0016 if not red_lr else 0.016,
            rgb_colors=0.0025,
            unnorm_rotations=0.01 if not red_lr else 0.1,
            logit_opacities=0.0005 if not red_lr else 0.005,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            instseg=0.000,
            embeddings=embeddings_lr if not red_lr and not embeddings_lr == 0 else 0.01,
            bg=0.0001
        ),
    ),
    tracking_cam=dict(
        num_iters=tracking_iters_cam,
        forward_prop=forward_prop,
        calc_ssmi=False,
        sil_thres=0.9,
        use_l1=True,
        bg_reg=False,
        use_sil_for_loss=True,
        take_best_candidate=True,
        gt_w2c=gt_w2c,
        restart_if_fail=False,  
        loss_weights=dict(
            im=1.0,
            depth=0.05,
            embeddings=1.0,
        ),
        lrs=dict(
            means3D=0.0000,
            rgb_colors=0.000,
            unnorm_rotations=0.000,
            logit_opacities=0.000,
            log_scales=0.000,
            cam_unnorm_rots=0.001 if not red_lr else 0.01,
            cam_trans=0.001 if not red_lr else 0.01,
            instseg=0.000,
            embeddings=0.00,
            bg=0.000
        ),
    ),
    viz=dict(
        vis_grid=True,
        vis_tracked=True,
        save_pc=False,
        save_videos=False,
        vis_gt=False,
        vis_all=vis_all
    ),
)
