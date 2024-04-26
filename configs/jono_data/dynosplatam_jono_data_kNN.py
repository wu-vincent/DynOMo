import os
from os.path import join as p_join

scenes = ["softball/ims/27"]

primary_device="cuda:0"
seed = 0
scene_name = scenes[0]

add_every = 1
tracking_iters = 1000
delta_optim_iters = 0
tracking_iters_cam = 0
mapping_iters = 0
mov_init_by = 'kNN'

group_name = "dynosplatam_jono"
run_name = f"splatam_{scene_name}_{seed}_{mov_init_by}"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    checkpoint='',
    seed=seed,
    primary_device=primary_device,
    add_every=add_every, # Mapping every nth frame
    eval_every=1, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    save_checkpoints=True, # Save Checkpoints
    checkpoint_interval=1, # Checkpoint Interval
    use_wandb=False,
    compute_normals=False,
    mov_static_init=False,
    make_bg_static=False,
    mov_init_by=mov_init_by, #'seg', 'kNN', 'per_point' 'learned_flow
    determine_mov='kNN', #'seg', 'kNN', 'per_point'
    use_seg_for_nn=False,
    mov_thresh=0.01,
    mode='static_dyno', # 'just_static', 'splatam' 'static_dyno'
    filter_small_segments=25,
    zeodepth=False,
    dist_to_use='rgb', # 'rgb', 'dinov2', 'l2'
    init_scale=1.0,
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
        desired_image_height=360,
        desired_image_width=640,
        start=0,
        end=-1,
        stride=1,
        num_frames=95,
        load_embeddings=False
    ),
    add_gaussians=dict(
        add_new_gaussians=True,
        depth_error_factor=50,
        use_depth_error_for_adding_gaussians=False,
        sil_thres_gaussians=0.5, # For Addition of new Gaussians
    ),
    remove_gaussians=dict(
        remove=False,
        remove_factor=100,
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
        sil_thres=0.95,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        dyno_losses=True,
        use_seg_loss=False,
        take_best_candidate=False,
        disable_rgb_grads_old=True,
        calc_ssmi=True,
        bg_reg=False,
        use_flow=False, # 'rendered' # None
        loss_weights=dict(
            im=1.0,
            delta_im=1.0,
            next_im=1.0,
            depth=0.5,
            delta_depth=0.5,
            next_depth=0.5,
            rot=4.0,
            rigid=4.0,
            iso=2.0,
            flow=10.0,
            instseg=1.0,
            scale_reg=0.000000,
            normals=0.00000,
            normals_neighbors=0.00000,
            mean_dx=0.0,
            moving=1.0,
            embeddings=0.0,
            bg_reg=0.0001
        ),
        lrs=dict(
            means3D=0.00016,
            delta_means3D=0.00016,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            delta_unnorm_rotations=0.001,
            logit_opacities=0.005,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            instseg=0.00016,
            scale_reg=0.00000,
            normals=0.00000,
            normals_neighbors=0.00000,
            moving=0.001,
            embeddings=0.0,
        ),
    ),
    tracking_cam=dict(
        num_iters=tracking_iters_cam,
        sil_thres=0.95,
        use_l1=True,
        use_sil_for_loss=True,
        take_best_candidate=True,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0000,
            rgb_colors=0.000,
            unnorm_rotations=0.000,
            logit_opacities=0.000,
            log_scales=0.000,
            cam_unnorm_rots=0.0005,
            cam_trans=0.0005,
        ),
    ),
    viz=dict(
        vis_grid=True,
        vis_tracked=True,
        save_pc=False,
        save_videos=False
    ),
)
