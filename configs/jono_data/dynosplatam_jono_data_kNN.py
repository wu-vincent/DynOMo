import os
from os.path import join as p_join

scenes = ["softball/ims/27", "basketball/ims/21", "football/ims/18", "juggle/ims/23", "boxes/ims/27", "tennis/ims/8"]

primary_device="cuda:1"
seed = 0
scene_name = scenes[0]

add_every = 1
tracking_iters = 100
tracking_iters_init = 100
num_frames = -1
feature_dim = 32
ssmi_all_mods = False
load_embeddings = True
init_pc_jono = False
dyno_losses = True
jono_depth = False
mag_iso = True
delta_optim_iters = 0
tracking_iters_cam = 0
mapping_iters = 0
mov_init_by = 'kNN'
l1_losses = 0
bg_reg = 0
embeddings_lr = 0 #0.001

# remove_gaussians = False
# sil_thres_gaussians = 0.5

remove_gaussians = False
sil_thres_gaussians = 0.5

vis_all = False

group_name = "dynosplatam_jono"
run_name = f"splatam_{scene_name}_{seed}_{mov_init_by}_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}_{num_frames}_{feature_dim}_{init_pc_jono}_{dyno_losses}_{jono_depth}_{mag_iso}_{remove_gaussians}_{sil_thres_gaussians}_{l1_losses}_{bg_reg}_{embeddings_lr}_debug_jono"

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
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    save_checkpoints=True, # Save Checkpoints
    checkpoint_interval=10, # Checkpoint Interval
    use_wandb=False,
    compute_normals=False,
    mov_static_init=False,
    make_bg_static=False,
    mov_init_by=mov_init_by, #'seg', 'kNN', 'per_point' 'learned_flow
    determine_mov='kNN', #'seg', 'kNN', 'per_point'
    use_seg_for_nn=True,
    mov_thresh=0.01,
    mode='static_dyno', # 'just_static', 'splatam' 'static_dyno'
    filter_small_segments=25,
    zeodepth=False,
    dist_to_use='l2', # 'rgb', 'dinov2', 'l2'
    init_scale=1.0,
    neighbors_init='post', # 'pre', 'first_post', 'post', 'reset_first_post'
    time_window=1,
    remove_outliers_l2=100, # 0.01,
    eval_during=False,
    motion_mlp=False,
    motion_lr=0.0001,
    re_init_scale=False,
    wandb=dict(
        project="DynoSplaTAM",
        group=group_name,
        name=run_name,
        save_qual=False,    
        eval_save_qual=True,
    ),
    data=dict(
        basedir="/scratch/jseidens/data/data",
        gradslam_data_cfg="./configs/data/jono_data.yaml",
        sequence=scene_name,
        desired_image_height=180, #180, #360,
        desired_image_width=320, #320, #640,
        start=0,
        end=num_frames,
        stride=1,
        num_frames=num_frames,
        load_embeddings=load_embeddings,
        embedding_dim=feature_dim,
        get_pc_jono=init_pc_jono,
        jono_depth=jono_depth
    ),
    add_gaussians=dict(
        add_new_gaussians=True,
        depth_error_factor=50,
        use_depth_error_for_adding_gaussians=False,
        sil_thres_gaussians=sil_thres_gaussians, # For Addition of new Gaussians
    ),
    remove_gaussians=dict(
        remove=remove_gaussians,
        remove_factor=15,
        rem_opa_thresh=0.5,
        rem_scale_thresh=100000
    ),
    prune_densify=dict(
        prune_gaussians=False, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=tracking_iters-1,
            remove_big_after=0,
            stop_after=100000,
            prune_every=tracking_iters-1,
            removal_opacity_threshold=-10,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
            kNN_rel_drift=0.8 # * scene radius in code
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
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        dyno_losses=dyno_losses,
        use_seg_loss=False,
        take_best_candidate=False,
        disable_rgb_grads_old=True,
        make_grad_bg_smaller=False,
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
            l1_rgb=l1_losses
        ),
        lrs=dict(
            means3D=0.0016,
            rgb_colors=0.0025,
            unnorm_rotations=0.01,
            logit_opacities=0.0005,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            instseg=0.00016,
            embeddings=embeddings_lr, #0.001,
            bg=0.0001
        ),
    ),
    init_next=dict(
        num_iters=tracking_iters,
        sil_thres=0.9,
        use_l1=True,
        use_sil_for_loss=True,
        ignore_outlier_depth_loss=False,
        dyno_losses=True,
        use_seg_loss=False,
        take_best_candidate=False,
        disable_rgb_grads_old=True,
        calc_ssmi=True,
        bg_reg=True,
        use_flow='', # 'rendered' # None
        loss_weights=dict(
            im=1.0,
            depth=0,
            rot=8.0,
            rigid=8.0,
            iso=4.0,
            flow=0.0,
            embeddings=25.0,
            bg_reg=0.0001,
            bg_loss=0.0
        ),
        lrs=dict(
            means3D=0.0016,
            rgb_colors=0.00,
            unnorm_rotations=0.01,
            logit_opacities=0.00,
            log_scales=0.00,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            instseg=0.000,
            moving=0.00,
            embeddings=0.0,
            bg=0.00
        ),
    ),
    tracking_cam=dict(
        num_iters=tracking_iters_cam,
        forward_prop=True,
        sil_thres=0.9,
        use_l1=True,
        use_sil_for_loss=True,
        take_best_candidate=True,
        loss_weights=dict(
            im=1.0,
            depth=0.05,
            embeddings=1.0
        ),
        lrs=dict(
            means3D=0.0000,
            rgb_colors=0.000,
            unnorm_rotations=0.000,
            logit_opacities=0.000,
            log_scales=0.000,
            cam_unnorm_rots=0.001,
            cam_trans=0.001,
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
