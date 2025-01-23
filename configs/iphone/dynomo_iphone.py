import os
from os.path import join as p_join

scenes = ['paper-windmill']

primary_device="cuda:0"
seed = 0
scene_name = scenes[0]

tracking_iters = 200
tracking_iters_init = 200
tracking_iters_cam = 0
num_frames = -1

group_name = "iphone"
run_name = f"{scene_name}/{scene_name}_{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    checkpoint=False,
    just_eval=False,
    seed=seed,
    primary_device=primary_device,
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    gaussian_distribution="anisotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    save_checkpoints=True, # Save Checkpoints
    checkpoint_interval=50, # Checkpoint Interval
    use_wandb=False,
    make_bg_static=False,
    mov_init_by='kNN', #'seg', 'kNN', 'per_point' 'learned_flow
    dist_to_use='embeddings', # 'rgb', 'dinov2', 'l2'
    eval_during=False,
    early_stop=True,
    stride=1,
    kNN=20,
    gt_w2c=False,
    wandb=dict(
        project="DynoSplaTAM",
        group=group_name,
        name=run_name,
        save_qual=False,    
        eval_save_qual=True,
    ),
    data=dict(
        name='iphone',
        basedir="/data3/jseidens/iphone",
        gradslam_data_cfg="./configs/data/iphone.yaml",
        sequence=scene_name,
        desired_image_height=0.5, #480,
        desired_image_width=0.5, #910,
        start=0,
        end=num_frames,
        num_frames=num_frames,
        load_embeddings=True,
        embedding_dim=32,
        start_from_complete_pc=False,
        novel_view_mode=None,
        depth_type='aligned_depth_anything_colmap', # 'lidar', DepthAnything, DepthAnythingV2
        cam_type='refined',
        do_scale=False,
        every_x_frame=1,
        online_emb=None,
        online_depth=None
    ),
    add_gaussians=dict(
        add_new_gaussians=True,
        depth_error_factor=5,
        use_depth_error_for_adding_gaussians=False,
        sil_thres_gaussians=0.5, # For Addition of new Gaussians
    ),
    prune_densify=dict(
        prune_gaussians=False, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=int(tracking_iters/2),
            remove_big_after=0,
            stop_after=100000,
            prune_every=int(tracking_iters/2)+1,
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
        use_sil_for_loss=False,
        take_best_candidate=False,
        disable_rgb_grads_old=True,
        forward_prop=True,
        make_grad_bg_smaller=False,
        make_grad_bg_smaller_weight=0, 
        calc_ssmi=True,
        bg_reg=True,
        gt_w2c=False,
        loss_weights=dict(
            im=1.0,
            depth=0.1,
            rot=16.0,
            rigid=128.0,
            iso=16.0,
            embeddings=16.0,
            bg_reg=5,
            bg_loss=3,
            l1_bg=0,
            l1_scale=0,
            l1_opacity=0,
            l1_embeddings=20,
            l1_rgb=20,
        ),
        lrs=dict(
            means3D=0.0016,
            rgb_colors=0.0025,
            unnorm_rotations=0.01,
            logit_opacities=0.0005,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            embeddings=0.001,
            instseg=0.0,
            bg=0.0001
        ),
    ),
    tracking_cam=dict(
        num_iters=tracking_iters_cam,
        forward_prop=True,
        calc_ssmi=False,
        sil_thres=0.9,
        use_l1=True,
        bg_reg=False,
        use_sil_for_loss=True,
        take_best_candidate=True,
        gt_w2c=False,
        restart_if_fail=True,  
        loss_weights=dict(
            im=1.0,
            depth=0.1,
            embeddings=16.0,
        ),
        lrs=dict(
            means3D=0.0000,
            rgb_colors=0.000,
            unnorm_rotations=0.000,
            logit_opacities=0.000,
            log_scales=0.000,
            cam_unnorm_rots=0.001,
            cam_trans=0.001,
            embeddings=0.00,
            instseg=0.0,
            bg=0.000
        ),
    ),
    viz=dict(
        vis_grid=False,
        vis_trajs=False,
        vis_tracked=True,
        save_pc=False,
        save_videos=False,
        vis_gt=False,
        vis_all=False,
        vis_fg_only=True
    ),
)
