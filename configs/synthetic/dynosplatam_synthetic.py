import os
from os.path import join as p_join

scenes = ["pendulums"]

primary_device="cuda:0"
seed = 0
scene_name = scenes[0]

map_every = 1
keyframe_every = 1
mapping_window_size = 24
tracking_iters = 80
mapping_iters = 0

group_name = "dynosplatam_synthetic"
run_name = f"splatam_{scene_name}_{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # Report Global Progress every nth frame
    eval_every=1, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=100, # Checkpoint Interval
    use_wandb=True,
    compute_normals=False,
    mov_static_init=False,
    seg_for_mov=True,
    use_seg_for_nn=True,
    mov_thresh=0.001,
    mode='static_dyno', # 'just_static', 'splatam' 'static_dyno'
    assign_instseg='3D',
    filter_small_segments=25,
    use_dbscan_filter=False,
    remove_background=True,
    use_rendered_moving=False,
    wandb=dict(
        project="DynoSplaTAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="./data/synthetic",
        gradslam_data_cfg="./configs/data/synthetic.yaml",
        sequence=scene_name,
        desired_image_height=480,
        desired_image_width=640,
        start=0,
        end=-1,
        stride=1,
        num_frames=30,
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
            instseg=0.00016,
        ),
    ),
    tracking_obj=dict(
        num_iters=tracking_iters,
        forward_prop=True, # Forward Propagate Poses
        sil_thres=0.99,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        dyno_losses=True,
        use_seg_loss=False,
        take_best_candidate=True,
        disable_rgb_grads_old=True,
        disable_grads_stat=False,
        mask_moving=False,
        separate_static=False,
        calc_ssmi=False,
        use_rendered_moving_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            rot=4.0,
            rigid=4.0,
            iso=2.0,
            instseg=1.0,
            scale_reg=0.000000,
            normals=0.00000,
            normals_neighbors=0.00000,
            mean_dx=0.0,
            moving=1.0
        ),
        lrs=dict(
            means3D=0.00016,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.005,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            instseg=0.00016,
            scale_reg=0.00000,
            normals=0.00000,
            normals_neighbors=0.00000,
            moving=0.001,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        use_depth_error_for_adding_gaussians=False,
        sil_thres_gaussians=0.9, # For Addition of new Gaussians
        sil_thres=0.99,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        dyno_losses=True,
        use_seg_loss=False,
        disable_mov_grads=False,
        disable_rgb_grads_old=True,
        disable_rgb_grads_mov=False,
        mask_moving=False,
        separate_static=False,
        calc_ssmi=True,
        use_rendered_moving_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            rot=4.0,
            rigid=4.0,
            iso=2.0,
            instseg=1.0,
            scale_reg=0.000000,
            normals=0.00000,
            normals_neighbors=0.00000,
            mean_dx=0.0,
            moving=1.0
        ),
        lrs=dict(
            means3D=0.00016,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.005,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            instseg=0.00016,
            scale_reg=0.000001,
            normals=0.00000,
            normals_neighbors=0.00000,
            moving=0.001,
        ),
        prune_gaussians=False, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
    ),
)
