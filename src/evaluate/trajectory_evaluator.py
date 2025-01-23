import os
from src.utils.get_data import get_gt_traj, load_scene_data
import numpy as np
import torch
from src.utils.camera_helpers import get_projection_matrix
from src.utils.gaussian_utils import three2two, unnormalize_points, normalize_points
from src.utils.viz_utils import vis_trail
from src.utils.camera_helpers import setup_camera
from src.model.renderer import RenderHelper
from src.utils.gaussian_utils import build_rotation
from datasets.datasets.geometryutils import relative_transformation
import copy
import imageio
import cv2
import flow_vis
from src.utils.viz_utils import make_vid


class TrajEvaluator():
    def __init__(
            self,
            config,
            params=None,
            results_dir='out',
            cam=None,
            vis_trajs=True,
            use_norm_pix=False,
            vis_thresh=0.5,
            vis_thresh_start=0.5,
            best_x=1,
            traj_len=0,
            get_gauss_wise3D_track=True,
            get_from3D=False,
            vis_trajs_best_x=False,
            queries_first_t=True,
            primary_device='cuda:0'):
        self.render_helper = RenderHelper()
        self.config = config
        self.results_dir = results_dir
        self.params = params
        self.cam = cam
        self.use_norm_pix = use_norm_pix
        self.vis_thresh = vis_thresh
        self.vis_thresh_start = vis_thresh_start
        self.best_x = best_x
        self.get_gauss_wise3D_track = get_gauss_wise3D_track
        self.get_from3D = get_from3D
        self.vis_trajs_best_x = vis_trajs_best_x
        self.traj_len = traj_len
        self.vis_trajs = vis_trajs
        self.get_proj_and_params(primary_device)
        self.dev = self.params['means3D'].device
        self.queries_first_t = queries_first_t
        print(f"\nEvaluating queries for time only {self.queries_first_t}")
        if 'davis' in self.config['data']["gradslam_data_cfg"].lower():
            self.fps = 24
            if traj_len == 0:
                self.N = 1024
            else:
                self.N = 2048
        elif 'iphone' in self.config['data']["gradslam_data_cfg"].lower():
            self.fps = 30 if self.config['data']['sequence'] not in ['haru-sit', 'mochi-high-five'] else 60
            if traj_len == 0:
                self.N = 8192
            else:
                self.N = 4096
        elif 'panoptic_sport' in self.config['data']["gradslam_data_cfg"].lower():
            self.fps = 30
            if traj_len == 0:
                self.N = 1024
            else:
                self.N = 2048

    def eval_traj(self):
        self.visuals = (self.vis_trajs_best_x or self.vis_trajs) and self.traj_len > 0

        # get gt data
        data = get_gt_traj(self.config, in_torch=True, device=self.dev)
        if 'davis' in self.config['data']["gradslam_data_cfg"].lower():
            dataset = 'davis'
        elif 'iphone' in self.config['data']["gradslam_data_cfg"].lower():
            dataset = 'iphone'
        elif 'panoptic_sport' in self.config['data']["gradslam_data_cfg"].lower():
            dataset = 'panoptic_sport'

        # get metrics
        metrics = self._eval_traj(
            data,
            dataset=dataset)

        return metrics
    
    def _eval_traj(
            self,
            data,
            dataset='panoptic_sport'
        ):

        copied_params = copy.deepcopy(self.params)
        # get GT
        gt_traj_2D = data['points']
        gt_traj_3D = data['trajs'] if 'trajs' in data.keys() else None
        occluded = data['occluded']
        valids = 1-occluded.float()

        # move by one pix
        if dataset == 'panoptic_sport':
            gt_traj_2D[:, :, 0] = ((gt_traj_2D[:, :, 0] * self.w) - 1)/self.w
            gt_traj_2D[:, :, 1] = ((gt_traj_2D[:, :, 1] * self.h) - 1)/self.h
            search_fg_only = True
        else:
            search_fg_only = False
        
        if 'time_pairs' not in data.keys() or self.queries_first_t:
            start_time = torch.zeros(gt_traj_2D.shape[0]).to(self.dev).long()
            start_pixels = gt_traj_2D[:, 0].clone()
            start_3D = gt_traj_3D[:, 0].clone().cuda() if gt_traj_3D is not None else gt_traj_3D
        else:
            N, T, D = data['points'].shape
            start_time = data['time_ids'][None].long().repeat((N, 1)).flatten()
            start_pixels = data["points"].reshape(N*T, -1)
            start_3D = data['trajs'].reshape(N*T, -1)

        # self.params['visibility'] = (self.params['visibility'] > vis_thresh).float()
        # get trajectories of Gaussians

        gs_traj_2D, gs_traj_3D, pred_visibility, gs_traj_2D_for_vis = self.get_gs_traj_pts(
            start_pixels,
            start_pixels_normalized=True,
            search_fg_only=search_fg_only,
            start_3D=start_3D,
            start_time=start_time)

        # get bet our of x
        gs_traj_2D, gs_traj_3D, gs_traj_2D_for_vis, pred_visibility = self.best_x_idx(
            occluded, dataset, gt_traj_2D, gs_traj_2D, gs_traj_3D, gs_traj_2D_for_vis, pred_visibility, valids)

        # make predicted visinbility bool
        pred_visibility = (pred_visibility > self.vis_thresh).float()

        # unnormalize gt to image pixels
        gt_traj_2D = unnormalize_points(gt_traj_2D, self.h, self.w)

        metrics = dict()
        if dataset != "iphone":
            # mask valid ids
            if valids.sum() != 0:
                gs_traj_2D, gt_traj_2D, valids, occluded, pred_visibility, gt_traj_3D, gs_traj_3D, gs_traj_2D_for_vis = mask_valid_ids(
                    valids.unsqueeze(0),
                    gs_traj_2D.unsqueeze(0),
                    gt_traj_2D.unsqueeze(0),
                    occluded.unsqueeze(0),
                    pred_visibility.unsqueeze(0),
                    gt_traj_3D.unsqueeze(0) if gt_traj_3D is not None else None,
                    gs_traj_3D.unsqueeze(0) if gs_traj_3D is not None else None,
                    gs_traj_2D_for_vis=gs_traj_2D_for_vis.unsqueeze(0) if self.visuals else None)
            # compute metrics from pips
            pips_metrics = compute_metrics(
                self.h,
                self.w,
                gs_traj_2D.to(self.dev),
                gt_traj_2D.to(self.dev),
                valids.to(self.dev))
            print("-----------------------------")
            print(f"2D Survivial: {pips_metrics['survival']}")
            print(f"2D Median L2: {pips_metrics['median_l2']}")
            print(f"2D delta average: {pips_metrics['d_avg']}")
            print("-----------------------------")
            metrics.update({'pips': pips_metrics})

            if 'trajs' in data.keys():
                # multiply by 100 because cm evaluation
                metrics3D = compute_metrics(
                    None,
                    None,
                    (gs_traj_3D*100).to(self.dev),
                    (gt_traj_3D*100).to(self.dev),
                    valids.to(self.dev),
                    sur_thr=50,
                    norm_factor=None)
                print("-----------------------------")
                print(f"3D Survivial: {metrics3D['survival']}")
                print(f"3D Median L2: {metrics3D['median_l2']}")
                print(f"3D Delta Vverage: {metrics3D['d_avg']}")
                print("-----------------------------")
                metrics.update({'pips_3D': {f'{k}_3D': v for k, v in metrics3D.items()}})
                
            if (1-occluded.long()).sum() != 0:
                # compute metrics from tapvid
                samples = sample_queries_first(
                    occluded.cpu().bool().numpy().squeeze(),
                    gt_traj_2D.cpu().numpy().squeeze())
                tapvid_metrics = compute_tapvid_metrics(
                    samples['query_points'],
                    samples['occluded'],
                    samples['target_points'],
                    (1-pred_visibility).cpu().numpy(),
                    gs_traj_2D.cpu().numpy(),
                    W=self.w,
                    H=self.h)
                print("-----------------------------")
                print(f"2D Delta Average: {tapvid_metrics['average_pts_within_thresh']}")
                print(f"2D AJ: {tapvid_metrics['average_jaccard']}")
                print(f"2D OA: {tapvid_metrics['occlusion_accuracy']}")
                print("-----------------------------")
                metrics.update({'tapvid': tapvid_metrics})

        else:
            metrics = dict()
            metrics.update(compute_metrics_iphone(data, pred_visibility, gs_traj_2D, self.queries_first_t))
            metrics.update(compute_metrics_iphone3D(data, gs_traj_3D, self.queries_first_t))

        if self.vis_trajs:
            print('Visualizeing tracked points')
            gs_traj_2D_for_vis = gs_traj_2D_for_vis if gs_traj_2D_for_vis is not None else gs_traj_2D
            if dataset == "iphone" and self.traj_len > 0:
                gs_traj_2D_for_vis = gs_traj_2D_for_vis[:, start_time==0]
                pred_visibility = pred_visibility[start_time==0]
            elif dataset == "iphone":
                gs_traj_2D_for_vis = gs_traj_2D_for_vis[start_time==0]
                pred_visibility = pred_visibility[start_time==0]

            data['points'] = normalize_points(gs_traj_2D_for_vis, self.h, self.w).squeeze()
            data['occluded'] = occluded.squeeze()
            data = {k: v.detach().clone().cpu().numpy() for k, v in data.items()}
            vis_trail(
                os.path.join(self.results_dir, 'tracked_points_vis'),
                data,
                pred_visibility=pred_visibility.squeeze(),
                vis_traj=True if self.traj_len > 0 else False,
                traj_len=self.traj_len,
                fps=self.fps)
        
        self.params = copy.deepcopy(copied_params)    
        return metrics
    
    def best_x_idx(
            self,
            occluded,
            dataset,
            gt_traj_2D,
            gs_traj_2D,
            gs_traj_3D,
            gs_traj_2D_for_vis,
            pred_visibility,
            valids,
            get_best_jaccard=True,
            use_heuristics=False,
            just_take_mean=False,
            just_take_max_vis=False):
        
        if self.best_x > 1 and self.vis_trajs_best_x:
            print('Visualizeing tracked points')
            data['points'] = normalize_points(gs_traj_2D_for_vis, self.h, self.w).squeeze()
            data['occluded'] = occluded.squeeze()
            data = {k: v.detach().clone().cpu().numpy() for k, v in data.items()}
            vis_trail(
                os.path.join(self.results_dir, 'tracked_points_vis'),
                data,
                pred_visibility=pred_visibility.squeeze(),
                vis_traj=True if self.traj_len > 0 else False,
                traj_len=self.traj_len,
                fps=self.fps)

        # N*best_x, T, D 
        _, T, D = gs_traj_2D.shape
        gs_traj_2D = gs_traj_2D.reshape(-1, self.best_x, T, 2).permute(1, 0, 2, 3)
        gs_traj_3D = gs_traj_3D.reshape(-1, self.best_x, T, 3).permute(1, 0, 2, 3)
        pred_visibility = pred_visibility.reshape(-1, self.best_x, T).permute(1, 0, 2)
        if self.visuals:
            gs_traj_2D_for_vis = gs_traj_2D_for_vis.reshape(
                T, -1, self.best_x, T, 2).permute(0, 2, 1, 3, 4)
        
        if dataset != "iphone":
            if just_take_mean:
                torch.use_deterministic_algorithms(False)
                gs_traj_2D = gs_traj_2D.median(dim=0).values
                pred_visibility = pred_visibility.median(dim=0).values
                gs_traj_3D = gs_traj_3D.median(dim=0).values
                if self.visuals:
                    gs_traj_2D_for_vis = gs_traj_2D_for_vis.median(dim=1).values
                torch.use_deterministic_algorithms(True)
                return gs_traj_2D, gs_traj_3D, gs_traj_2D_for_vis, pred_visibility   
        
            if just_take_max_vis:
                min_idx = pred_visibility[:, :, 0].max(dim=0).indices
            elif use_heuristics:
                min_idx = heuristic_best_of_x(gs_traj_2D, pred_visibility, self.w, self.h)
            elif get_best_jaccard:
                samples = sample_queries_first(
                    occluded.cpu().bool().numpy().squeeze(),
                    unnormalize_points(copy.deepcopy(gt_traj_2D.cpu().numpy().squeeze()), self.h, self.w),
                    ignore_invalid=True)
                samples = {k: v.repeat(self.best_x, axis=0) for k, v in samples.items()}
                min_idx = torch.from_numpy(get_smallest_AJ(
                    samples['query_points'],
                    samples['occluded'],
                    samples['target_points'],
                    (1-(pred_visibility > self.vis_thresh).float()).cpu().numpy(),
                    gs_traj_2D.cpu().numpy(),
                    W=self.w,
                    H=self.h)).to(self.dev)
            else:
                # Take the best of several Gaussials
                min_idx = get_smallest_l2(
                    gs_traj_2D,
                    pred_visibility > self.vis_thresh,
                    gt_traj_2D.unsqueeze(0).repeat(self.best_x, 1, 1, 1).to(self.dev),
                    valids.unsqueeze(0).repeat(self.best_x, 1, 1).to(self.dev),
                    W=self.w, H=self.h,
                    median=True)
        else:
            min_idx = 0

        _, N, _, _ = gs_traj_2D.shape
        pred_visibility = pred_visibility[min_idx, torch.arange(N), :]
        gs_traj_2D = gs_traj_2D[min_idx, torch.arange(N), :, :]
        gs_traj_3D = gs_traj_3D[min_idx, torch.arange(N), :, :]
        if self.visuals:
            gs_traj_2D_for_vis = gs_traj_2D_for_vis[:, min_idx, torch.arange(N), :, :]
        
        return gs_traj_2D, gs_traj_3D, gs_traj_2D_for_vis, pred_visibility
            
    def get_proj_and_params(self, primary_device):
        # get projectoin matrix
        if self.cam is None:
            self.params, _, k, w2c = load_scene_data(self.config,  os.path.dirname(self.results_dir), device=primary_device)
            if len(self.params['visibility'].shape) == 3:
                self.params['visibility'] = self.params['visibility'][:, 0, :]
            elif len(self.params['visibility'].shape) == 3:
                self.params['visibility'] = self.params['visibility'][:, 1, :]
            if "desired_height" in self.params.keys():
                self.h, self.w = self.params["desired_height"].cpu().item(), self.params["desired_width"].cpu().item()
            else:
                self.h, self.w = self.config["data"]["desired_image_height"], self.config["data"]["desired_image_width"]
            self.proj_matrix = get_projection_matrix(self.w, self.h, k, w2c, device=self.params['means3D'].device).squeeze()
            self.cam = setup_camera(self.w, self.h, k, w2c, device=self.params['means3D'].device)
        else:
            self.proj_matrix = self.cam.projmatrix.squeeze()
            self.h = self.cam.image_height
            self.w = self.cam.image_width
        self.w = int(self.w)
        self.h = int(self.h)
        print(f"Used image height {self.h} and width {self.w}...")

    def format_start_pix(self, start_pixels, start_pixels_normalized):
        if not self.use_norm_pix or self.get_gauss_wise3D_track:
            if start_pixels_normalized:
                start_pixels = unnormalize_points(start_pixels, self.h, self.w, do_round=False)
            start_pixels = start_pixels.to(self.dev).float()
        else:
            if not start_pixels_normalized:
                start_pixels = normalize_points(start_pixels.float(), self.h, self.w)
            start_pixels = start_pixels.to(self.dev).float()

        return start_pixels

    def gauss_wise3D_track(self, search_fg_only, start_pixels, start_time, start_3D, do_3D=False, only_t0=True):
        first_occurance = self.params['timestep']
        if search_fg_only:
            fg_mask = (self.params['bg'] < 0.5).squeeze()
            first_occurance = first_occurance[fg_mask]
            for k in self.params.keys():
                try:
                    self.params[k] = self.params[k][fg_mask]
                except:
                    self.params[k] = self.params[k]

        # only search gaussians inializaed at t=0
        params_gs_traj_3D = copy.deepcopy(self.params)
        if only_t0:
            first = first_occurance==first_occurance.min().item()
        else:
            first = torch.ones_like(first_occurance, dtype=bool, device=self.dev)
        params_gs_traj_3D['means3D'] = params_gs_traj_3D['means3D'][first]
        params_gs_traj_3D['unnorm_rotations'] = params_gs_traj_3D['unnorm_rotations'][first]
        params_vis = params_gs_traj_3D['visibility'][first]
        # get Gauss IDs
        gauss_ids = torch.zeros(start_pixels.shape[0] * self.best_x, device=self.dev).long()
        start_time_best_x = start_time[..., None].repeat((1, self.best_x)).flatten()
        for time in start_time.unique():
            gaussians_start_time_t = self.render_helper.transform_to_frame(
                    params_gs_traj_3D,
                    time,
                    gaussians_grad=False,
                    camera_grad=False)
            
            means3D_start = gaussians_start_time_t['means3D']
            if not do_3D:
                means2D_start= three2two(
                    self.proj_matrix, means3D_start, self.w, self.h, do_normalize=self.use_norm_pix)
                gauss_ids[start_time_best_x==time] = self.find_closest_to_start_pixels(
                    means2D_start.float(),
                    params_vis[:, time],
                    start_pixels[start_time==time])
            else:
                gauss_ids[start_time_best_x==time] = self.find_closest_to_start_pixels(
                    means3D_start.float(),
                    params_vis[:, time],
                    start_3D[start_time==time])
        # get Gauss tracks
        gs_traj_3D, unnorm_rotations, visibility = self.get_3D_trajs_for_track(
            gauss_ids, return_all=True)
        
        return gs_traj_3D, unnorm_rotations, visibility

    def get_2D_track_from_3D(self, gs_traj_3D, unnorm_rotations):
        params_gs_traj_3D = copy.deepcopy(self.params)
        params_gs_traj_3D['means3D'] = gs_traj_3D

        params_gs_traj_3D['unnorm_rotations'] = unnorm_rotations
        gs_traj_2D = list()
        for time in range(gs_traj_3D.shape[-1]):
            if gs_traj_3D[:, :, time].sum() == 0:
                continue
            transformed_gs_traj_3D = self.render_helper.transform_to_frame(
                    params_gs_traj_3D,
                    time,
                    gaussians_grad=False,
                    camera_grad=False)
            gs_traj_2D.append(
                three2two(self.proj_matrix, transformed_gs_traj_3D['means3D'], self.w, self.h, do_normalize=False))
        gs_traj_2D = torch.stack(gs_traj_2D).permute(1, 0, 2)
        gs_traj_3D = gs_traj_3D.permute(0, 2, 1)

        return gs_traj_2D

    def get_2D_track_from_3D_for_vis(self, gs_traj_3D, unnorm_rotations):
        params_gs_traj_3D = copy.deepcopy(self.params)
        params_gs_traj_3D['means3D'] = gs_traj_3D

        params_gs_traj_3D['unnorm_rotations'] = unnorm_rotations
        gs_traj_2D_per_time = list()
        for cam_time in range(gs_traj_3D.shape[-1]):
            gs_traj_2D = list()
            for gauss_time in range(gs_traj_3D.shape[-1]):
                if gs_traj_3D[:, :, gauss_time].sum() == 0:
                    continue
                transformed_gs_traj_3D = self.render_helper.transform_to_frame(
                        params_gs_traj_3D,
                        cam_time,
                        gaussians_grad=False,
                        camera_grad=False,
                        gauss_time_idx=gauss_time)
                gs_traj_2D.append(
                    three2two(self.proj_matrix, transformed_gs_traj_3D['means3D'], self.w, self.h, do_normalize=False))
            gs_traj_2D = torch.stack(gs_traj_2D).permute(1, 0, 2)
            gs_traj_2D_per_time.append(gs_traj_2D)
        gs_traj_2D_per_time = torch.stack(gs_traj_2D_per_time)
        return gs_traj_2D_per_time

    def get_2D_and_3D_from_sum(self, start_pixels, start_time):
        # initialize tensors
        num_frames = self.params['means3D'].shape[-1]
        num_pix = start_pixels.shape[0]
        all_trajs_3D = torch.zeros((num_pix, num_frames, 3))
        all_trajs_2D = torch.zeros((num_pix, num_frames, 2))
        all_visibilities = torch.zeros((num_pix, num_frames))
        gs_traj_2D_per_time = torch.zeros((num_frames, num_pix, num_frames, 2))

        for time in start_time.unique():
            start_pixels_time = start_pixels[start_time==time]
            with torch.no_grad():
                _, im, _, _, _, _, _, visible, weight, time_mask, _, _, _, _, _, _, _ = self.render_helper.get_renderings(
                    self.params,
                    self.params,
                    time,
                    data={'cam': self.cam},
                    config=None, 
                    disable_grads=True,
                    track_cam=False,
                    get_depth=False,
                    get_embeddings=False)
            visible_means_start_pix = visible[
                :, torch.round(start_pixels_time).long()[:, 1], torch.round(start_pixels_time).long()[:, 0]]
            weight_means_start_pix = weight[
                :, torch.round(start_pixels_time).long()[:, 1], torch.round(start_pixels_time).long()[:, 0]]

            _all_trajs_3D = list()
            _all_trajs_2D = list()
            _all_visibilities = list()
            _gs_traj_2D_per_time = list()
            for i in range(start_pixels_time.shape[0]):
                visible_means = visible_means_start_pix[:, i][torch.nonzero(visible_means_start_pix[:, i])].squeeze().long()
                weight_means = weight_means_start_pix[:, i][torch.nonzero(visible_means_start_pix[:, i])].squeeze()
                traj_3D = list()
                traj_2D = list()
                visibility = list()
                gs_traj_2D_per_time_per_start_pix = list()
                start_pix_params = copy.deepcopy(self.params)
                for cam_time in range(self.params['means3D'].shape[2]):
                    gs_traj_2D_per_time_per_start_pix_per_cam_time = list()
                    for gauss_time in range(self.params['means3D'].shape[2]):
                        if not self.visuals and cam_time != gauss_time:
                            continue
                        loc_3D = ((weight_means/weight_means.sum()).unsqueeze(1) * self.params['means3D'][visible_means, :, gauss_time]).sum(dim=0).unsqueeze(0).unsqueeze(-1)
                        start_pix_params['means3D'] = loc_3D
                        start_pix_params['unnorm_rotations'] = torch.zeros(1, 4, 1).to(self.dev)
                        transformed_loc_3D = self.render_helper.transform_to_frame(
                                start_pix_params,
                                cam_time,
                                gaussians_grad=False,
                                camera_grad=False,
                                gauss_time_idx=0)
                        loc_2D = three2two(self.proj_matrix, transformed_loc_3D['means3D'], self.w, self.h, do_normalize=False).float()
                        if cam_time == gauss_time:
                            visibility.append(((weight_means/weight_means.sum()) * self.params['visibility'][visible_means, gauss_time].squeeze()).sum())
                            traj_3D.append(loc_3D)
                            traj_2D.append(loc_2D)
                        if self.visuals:
                            gs_traj_2D_per_time_per_start_pix_per_cam_time.append(loc_2D)
                    if self.visuals:
                        gs_traj_2D_per_time_per_start_pix.append(
                            torch.stack(gs_traj_2D_per_time_per_start_pix_per_cam_time))

                _all_trajs_2D.append(torch.stack(traj_2D).squeeze())
                _all_trajs_3D.append(torch.stack(traj_3D).squeeze())
                _all_visibilities.append(torch.stack(visibility))
                if self.visuals:
                    _gs_traj_2D_per_time.append(torch.stack(gs_traj_2D_per_time_per_start_pix))

            _all_trajs_2D = torch.stack(_all_trajs_2D).squeeze()
            _all_trajs_3D = torch.stack(_all_trajs_3D).squeeze()
            _all_visibilities = torch.stack(_all_visibilities).squeeze()
            if self.visuals:
                _gs_traj_2D_per_time = torch.stack(_gs_traj_2D_per_time)

            all_trajs_2D[start_time==time] = _all_trajs_2D
            all_trajs_3D[start_time==time] = _all_trajs_3D
            all_visibilities[start_time==time] = all_visibilities
            if self.visuals:
                gs_traj_2D_per_time[start_time==time] =_gs_traj_2D_per_time
    
        if self.visuals:
            gs_traj_2D_per_time = gs_traj_2D_per_time.squeeze().permute(1, 0, 2, 3)
        else:
            gs_traj_2D_per_time = None
        return all_trajs_2D, all_trajs_3D, all_visibilities, gs_traj_2D_per_time

    def get_gs_traj_pts(
            self,
            start_pixels,
            start_time,
            start_3D=None,
            start_pixels_normalized=True,
            search_fg_only=False):
        
        # get start pixels in right format
        start_pixels = self.format_start_pix(
            start_pixels,
            start_pixels_normalized=start_pixels_normalized)

        gs_traj_2D_for_vis = None
        if self.get_gauss_wise3D_track: 
            gs_traj_3D, unnorm_rotations, visibility = \
                self.gauss_wise3D_track(
                    search_fg_only,
                    start_pixels,
                    start_time,
                    start_3D,
                    do_3D=self.get_from3D and start_3D is not None)
            
            if self.visuals:
                gs_traj_2D_for_vis = self.get_2D_track_from_3D_for_vis(
                    copy.deepcopy(gs_traj_3D),
                    copy.deepcopy(unnorm_rotations),
                )
            gs_traj_2D = self.get_2D_track_from_3D(
                gs_traj_3D, unnorm_rotations)
            gs_traj_3D = gs_traj_3D.permute(0, 2, 1)
        else:
            gs_traj_2D, gs_traj_3D, visibility, gs_traj_2D_for_vis = self.get_2D_and_3D_from_sum(
                start_pixels, start_time)
            
        return gs_traj_2D, gs_traj_3D, visibility, gs_traj_2D_for_vis

    def find_closest_not_round(self, means2D, pix, params_vis, from_closest=0, topk=30):
        dist = torch.cdist(pix.unsqueeze(0).unsqueeze(0), means2D.unsqueeze(0)).squeeze()
        dist_top_k = dist.topk(largest=False, k=topk)
        for i, k in enumerate(dist_top_k.indices[from_closest:]):
            if params_vis[k] >= self.vis_thresh_start:
                return k
        return dist_top_k.indices[0]

    def find_closest_to_start_pixels(
            self,
            means2D,
            params_vis,
            start_pixels,
            topk=30):
        gauss_ids = list()
        for j, pix in enumerate(start_pixels):
            best_x_gauss_ids = list()
            for i in range(self.best_x):
                gauss_id = self.find_closest_not_round(
                    means2D,
                    pix,
                    params_vis,
                    from_closest=i,
                    topk=topk).unsqueeze(0)
                best_x_gauss_ids.append(gauss_id)

            if best_x_gauss_ids[0] is not None:
                gauss_ids.extend(best_x_gauss_ids)
            else:
                gauss_ids.extend([torch.tensor([0]).to(self.dev)]*self.best_x)
        return torch.stack(gauss_ids).squeeze()
            
    def get_3D_trajs_for_track(self, gauss_ids, return_all=False):
        gs_traj_3D = list()
        unnorm_rotations = list()
        visibility = list()

        for gauss_id in gauss_ids:
            if gauss_id != -1:
                gs_traj_3D.append(
                        self.params['means3D'][gauss_id].squeeze())
                unnorm_rotations.append(self.params['unnorm_rotations'][gauss_id].squeeze())
                visibility.append(self.params['visibility'][gauss_id].squeeze())
            else:
                gs_traj_3D.append(
                        (torch.ones_like(self.params['means3D'][0]).squeeze()*-1).to(self.dev))
                unnorm_rotations.append(
                    (torch.ones_like(self.params['unnorm_rotations'][0]).squeeze()*-1).to(self.dev))
                visibility.append(
                        (torch.zeros(self.params['means3D'].shape[2])).to(self.dev))

        if return_all:
            return torch.stack(gs_traj_3D), torch.stack(unnorm_rotations), torch.stack(visibility)
        else:
            return torch.stack(gs_traj_3D)
    
    def vis_grid_trajs(
            self,
            mask=None,
            vis_vis_and_occ=True):
        
        # store best_x for later
        best_x = self.best_x
        self.best_x = 1
        self.visuals = self.traj_len > 0
        search_fg_only = False if mask is None else True

        # get trajectories to track
        start_pixels = get_xy_grid(
            self.h,
            self.w,
            N=self.N,
            device=self.dev,
            mask=mask).squeeze().long()
        np.save(
            os.path.join(self.results_dir, f'start_pixels_grid_{self.traj_len}.npy'),start_pixels.cpu().numpy())

        # no_bg
        gs_traj_2D, gs_traj_3D, pred_visibility, gs_traj_2D_for_vis = self.get_gs_traj_pts(
            start_pixels,
            start_pixels_normalized=False,
            start_time=torch.zeros(start_pixels.shape[0]).to(self.dev).long(),
            search_fg_only=search_fg_only)
        pred_visibility = (pred_visibility > self.vis_thresh).float()
        if vis_vis_and_occ:
            pred_visibility = torch.ones_like(
                pred_visibility, device=pred_visibility.device, dtype=float)

        if gs_traj_2D_for_vis is None:
            gs_traj_2D_for_vis = gs_traj_2D
        
        if isinstance(gs_traj_2D_for_vis, np.ndarray):
            gs_traj_2D_for_vis = torch.from_numpy(gs_traj_2D_for_vis)

        # get gt data for visualization (actually only need rgb here)
        data = get_gt_traj(self.config, in_torch=True)
        data['points'] = normalize_points(gs_traj_2D_for_vis, self.h, self.w).squeeze()   
        data['occluded'] = torch.zeros(data['points'].shape[:-1]).to(self.dev)
        data = {k: v.detach().clone().cpu().numpy() for k, v in data.items()}
        print("Visualizing grid...")
        try:
            vis_trail(
                    os.path.join(self.results_dir, 'grid_points_vis'),
                    data,
                    pred_visibility=pred_visibility.to(self.dev), # torch.ones_like(pred_visibility.squeeze()).to(self.dev)
                    vis_traj=True if self.traj_len > 0 else False,
                    traj_len=self.traj_len,
                    fg_only=search_fg_only,
                    fps=self.fps)
        except:
            print(f'failed for {self.results_dir}...')
        # reset best x
        self.best_x = best_x
    
    def vis_flow(self):
        vis_thresh, vis_thresh_start = self.vis_thresh, self.vis_thresh_start
        self.vis_thresh, self.vis_thresh_start = -0.0001, -0.0001
        # make dir to store
        os.makedirs(os.path.join(self.results_dir, 'flow'), exist_ok=True)
        print(f"Visualizing flow and storing to {os.path.join(self.results_dir, 'flow')}")

        # get trajectories to track
        x_grid, y_grid = torch.meshgrid(torch.arange(self.w).to(self.dev).float(), 
                                        torch.arange(self.h).to(self.dev).float(),
                                        indexing='xy')
        x_grid, y_grid = x_grid.flatten(), y_grid.flatten()
        start_pixels = torch.stack([x_grid, y_grid], dim=-1) # B, N_*N_, 2
        # no_bg
        start_pixels = self.format_start_pix(
            start_pixels,
            start_pixels_normalized=False)

        T = self.params['means3D'].shape[-1]
        optical_flow = torch.zeros((T-1, self.h, self.w, 2), device=self.dev)
        scene_flow = torch.zeros((T-1, self.h, self.w, 3), device=self.dev)
        for t in range(T-1):
            if t % 50 == 0:
                print(t, T-1)
            start_time = torch.ones((self.w*self.h), device=self.dev)*t
            gs_traj_3D, unnorm_rotations, visibility = \
            self.gauss_wise3D_track(
                False,
                copy.deepcopy(start_pixels),
                start_time.long(),
                None,
                do_3D=False,
                only_t0=False)
            gs_traj_2D = self.get_2D_track_from_3D(
                gs_traj_3D, unnorm_rotations)
            gs_traj_3D = gs_traj_3D.permute(0, 2, 1)
            optical_flow[t, :, :, :] = (gs_traj_2D[:, t+1, :] - gs_traj_2D[:, t, :]).reshape(self.h, self.w, 2)
            scene_flow[t, :, :, :] = (gs_traj_3D[:, t+1, :] - gs_traj_3D[:, t, :]).reshape(self.h, self.w, 3)

            flow_color = flow_vis.flow_to_color(optical_flow[t, :, :, :].cpu().numpy(), convert_to_bgr=False)
            flow_color = flow_color.astype(np.uint8)
            imageio.imwrite(os.path.join(self.results_dir, 'flow', "gs_{:04d}.png".format(t)), flow_color)
        make_vid(os.path.join(self.results_dir, 'flow'))
        torch.save(optical_flow, os.path.join(self.results_dir, 'optical_flow.pth'))
        torch.save(scene_flow, os.path.join(self.results_dir, 'scene_flow.pth'))
        self.vis_thresh, self.vis_thresh_start = vis_thresh, vis_thresh_start

    def eval_cam_traj(self):
        gt_w2c_list = self.params['gt_w2c_all_frames']
        if isinstance(gt_w2c_list, np.ndarray):
            gt_w2c_list = torch.from_numpy(gt_w2c_list)
        num_frames = self.params['cam_unnorm_rots'].shape[-1]
        latest_est_w2c = self.params['w2c'] if type(self.params['w2c']) == torch.Tensor \
             else torch.from_numpy(self.params['w2c'])
        latest_est_w2c_list = []
        latest_est_w2c_list.append(latest_est_w2c.cpu())
        valid_gt_w2c_list = []
        valid_gt_w2c_list.append(gt_w2c_list[0].cpu())
        for idx in range(1, num_frames):
            if isinstance(gt_w2c_list[idx], np.ndarray):
                gt_w2c_list[idx] = torch.from_numpy(gt_w2c_list[idx])
            # Check if gt pose is not nan for this time step
            if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                continue
            interm_cam_rot = torch.nn.functional.normalize(self.params['cam_unnorm_rots'][..., idx].detach())
            interm_cam_trans = self.params['cam_trans'][..., idx].detach()
            intermrel_w2c = torch.eye(4).cuda().float()
            intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
            intermrel_w2c[:3, 3] = interm_cam_trans
            latest_est_w2c = intermrel_w2c
            latest_est_w2c_list.append(latest_est_w2c.cpu())
            valid_gt_w2c_list.append(gt_w2c_list[idx].cpu())
        gt_w2c_list = valid_gt_w2c_list
        # Calculate ATE RMSE
        ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
        print("Final Average ATE RMSE: {:.2f} cm".format(ate_rmse*100))

        return ate_rmse*100
            

def evaluate_ate(gt_traj, est_traj):
    """
    Input : 
        gt_traj: list of 4x4 matrices 
        est_traj: list of 4x4 matrices
        len(gt_traj) == len(est_traj)
    """
    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]

    gt_traj_pts  = torch.stack(gt_traj_pts).detach().cpu().numpy().T
    est_traj_pts = torch.stack(est_traj_pts).detach().cpu().numpy().T

    _, _, trans_error = align(gt_traj_pts, est_traj_pts)

    avg_trans_error = trans_error.mean()

    return avg_trans_error


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def compute_metrics_iphone3D(data, pred_points, queries_first_t):
    keypoints_3d = data["trajs"].permute(1, 0, 2).numpy()
    visibility = ~(data['occluded_trajs'].permute(1, 0).numpy().astype(np.bool_))
    time_pairs = data["time_pairs"]
    index_pairs = data["index_pairs"]
    num_frames = len(data["time_ids"])
    num_pts = keypoints_3d.shape[1]

    # Compute 3D tracking metrics.
    pair_keypoints_3d = keypoints_3d[index_pairs]
    pair_visibility = visibility[index_pairs]
    is_covisible = (pair_visibility == 1).all(axis=1)
    target_keypoints_3d = pair_keypoints_3d[:, 1, :, :3]

    # only first timestep as queries
    if queries_first_t:
        pred_points = pred_points[:, data['time_ids']].cpu().permute(1, 0, 2).numpy()
        pred_points = pred_points[index_pairs]
        pred_points = pred_points[:, 1, :, :3]
    else:
        # choose evaluation time steps
        pred_points = pred_points.reshape(num_pts, num_frames, -1, 3)
        pred_points = pred_points[:, :, data['time_ids']].permute(1, 0, 2, 3).permute(0, 2, 1, 3)
        pred_points = pred_points.reshape(-1, num_pts, 3)
        pred_points = pred_points.cpu().numpy()

    epes = []
    for i in range(len(time_pairs)):
        epes.append(
            np.linalg.norm(
                target_keypoints_3d[i][is_covisible[i]]
                - pred_points[i][is_covisible[i]],
                axis=-1,
            )
        )

    epe = np.mean(
        [frame_epes.mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    pck_3d_50cm = np.mean(
        [(frame_epes < 0.5).mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    pck_3d_10cm = np.mean(
        [(frame_epes < 0.1).mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    pck_3d_5cm = np.mean(
        [(frame_epes < 0.05).mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    print(f"3D tracking EPE: {epe:.4f}")
    print(f"3D tracking PCK (50cm): {pck_3d_50cm:.4f}")
    print(f"3D tracking PCK (10cm): {pck_3d_10cm:.4f}")
    print(f"3D tracking PCK (5cm): {pck_3d_5cm:.4f}")
    print("-----------------------------")
    return {'epe': epe, 'pck_3d_50cm': pck_3d_50cm, 'pck_3d_10cm': pck_3d_10cm, 'pck_3d_5cm': pck_3d_5cm}


def compute_metrics_iphone(data, pred_visibilities, pred_points, queries_first_t=False):
    target_points = data["points"].permute(1, 0, 2).numpy()
    visibilities = ~(data['occluded'].permute(1, 0).numpy().astype(np.bool_))
    time_ids = data["time_ids"]
    num_frames = len(time_ids)
    num_pts = target_points.shape[1]

    target_points = target_points[None].repeat(num_frames, axis=0)[..., :2]
    target_visibilities = visibilities[None].repeat(num_frames, axis=0)

    if queries_first_t:
        # only first timestep as queries
        pred_points = pred_points[:, data['time_ids']].cpu().permute(1, 0, 2).numpy()
        pred_points = pred_points[None].repeat(num_frames, axis=0)
        pred_visibilities = pred_visibilities[:, data['time_ids']].cpu().permute(1, 0).numpy().astype(np.bool_)
        pred_visibilities = pred_visibilities[None].repeat(num_frames, axis=0)
        pred_visibilities = pred_visibilities.reshape(
            num_frames, -1, num_pts)
    else:
        # choose evaluation time steps
        pred_points = pred_points.reshape(num_pts, num_frames, -1, 2)
        pred_points = pred_points[:, :, data['time_ids']].permute(1, 0, 2, 3).permute(0, 2, 1, 3)
        pred_visibilities = pred_visibilities.reshape(num_pts, num_frames, -1)
        pred_visibilities = pred_visibilities[:, :, data['time_ids']].permute(1, 0, 2).permute(0, 2, 1)
        # to numpyr
        pred_points = pred_points.cpu().numpy()
        pred_visibilities = pred_visibilities.cpu().numpy().astype(np.bool_)

    one_hot_eye = np.eye(target_points.shape[0])[..., None].repeat(num_pts, axis=-1)
    evaluation_points = one_hot_eye == 0

    for i in range(num_frames):
        evaluation_points[i, :, ~visibilities[i]] = False

    occ_acc = np.sum(
        np.equal(pred_visibilities, target_visibilities) & evaluation_points
    ) / np.sum(evaluation_points)

    all_frac_within = []
    all_jaccard = []
    for thresh in [4, 8, 16, 32, 64]:
        within_dist = np.sum(
            np.square(pred_points - target_points),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, target_visibilities)
        count_correct = np.sum(is_correct & evaluation_points)
        count_visible_points = np.sum(target_visibilities & evaluation_points)
        frac_correct = count_correct / count_visible_points
        all_frac_within.append(frac_correct)
        true_positives = np.sum(is_correct & pred_visibilities & evaluation_points)
        gt_positives = np.sum(target_visibilities & evaluation_points)
        false_positives = (~target_visibilities) & pred_visibilities
        false_positives = false_positives | ((~within_dist) & pred_visibilities)
        false_positives = np.sum(false_positives & evaluation_points)
        jaccard = true_positives / (gt_positives + false_positives)
        all_jaccard.append(jaccard)
    AJ = np.mean(all_jaccard)
    APCK = np.mean(all_frac_within)

    print("-----------------------------")
    print(f"2D tracking AJ: {AJ:.4f}")
    print(f"2D tracking avg PCK: {APCK:.4f}")
    print(f"2D tracking occlusion accuracy: {occ_acc:.4f}")
    print("-----------------------------")
    return {'AJ': AJ, 'APCK': APCK, 'occ_acc': occ_acc}


def mask_valid_ids(valids, gs_traj_2D, gt_traj_2D, occluded, pred_visibility, gt_traj_3D, gs_traj_3D, gs_traj_2D_for_vis=None):
    # only keep points that are visible at time 0
    vis_ok = valids[:, :, 0] > 0

    # flatten along along batch * num points and mask
    shape = gs_traj_2D.shape
    vis_ok = vis_ok.reshape(shape[0]*shape[1])
    gs_traj_2D = gs_traj_2D.reshape(
        shape[0]*shape[1], shape[2], shape[3])[vis_ok].reshape(
            shape[0], -1, shape[2], shape[3])
    gt_traj_2D = gt_traj_2D.reshape(
        shape[0]*shape[1], shape[2], shape[3])[vis_ok].reshape(
            shape[0], -1, shape[2], shape[3])
    valids = valids.reshape(
        shape[0]*shape[1], shape[2])[vis_ok].reshape(
            shape[0], -1, shape[2])
    occluded = occluded.reshape(
        shape[0]*shape[1], shape[2])[vis_ok].reshape(
            shape[0], -1, shape[2])
    pred_visibility = pred_visibility.reshape(
        shape[0]*shape[1], shape[2])[vis_ok].reshape(
            shape[0], -1, shape[2])
    if gs_traj_2D_for_vis is not None:
        gs_traj_2D_for_vis = gs_traj_2D_for_vis.permute(0, 2, 1, 3, 4).reshape(
            shape[0]*shape[1], shape[2], shape[2], shape[3])[vis_ok]
        gs_traj_2D_for_vis = gs_traj_2D_for_vis.reshape(
                shape[0], -1, shape[2], shape[2], shape[3]).permute(0, 2, 1, 3, 4)
    if gt_traj_3D is not None:
        shape = gt_traj_3D.shape
        gt_traj_3D = gt_traj_3D.reshape(
        shape[0]*shape[1], shape[2], shape[3])[vis_ok].reshape(
            shape[0], -1, shape[2], shape[3])
        gs_traj_3D = gs_traj_3D.reshape(
        shape[0]*shape[1], shape[2], shape[3])[vis_ok].reshape(
            shape[0], -1, shape[2], shape[3])

    return gs_traj_2D, gt_traj_2D, valids, occluded, pred_visibility, gt_traj_3D, gs_traj_3D, gs_traj_2D_for_vis


def get_smallest_l2(gs_traj_2D, gt_traj_2D, valids, W, H, norm_factor=256, median=False):
    B, N, S = gt_traj_2D.shape[0], gt_traj_2D.shape[1], gt_traj_2D.shape[2]
    
    # permute number of points and seq len
    gs_traj_2D = gs_traj_2D.permute(0, 2, 1, 3)
    gt_traj_2D = gt_traj_2D.permute(0, 2, 1, 3)

    # get metrics
    sc_pt = torch.tensor(
        [[[W/norm_factor, H/norm_factor]]]).float().to(self.dev)
    
    dists = torch.linalg.norm(gs_traj_2D/sc_pt - gt_traj_2D/sc_pt, dim=-1, ord=2) # B,S,N
    if median:
        dists_ = dists.permute(0,2,1).reshape(B*N,S)
        valids_ = valids.permute(0,2,1).reshape(B*N,S)
        median_l2 = reduce_masked_median(dists_, valids_, keep_batch=True).reshape(B, N)
    else:
        median_l2 = dists.mean(dim=1)
    min_idx = median_l2.min(dim=0).indices
    return min_idx


def heuristic_best_of_x(gs_traj_2D, pred_visibility, w, h, mean=False, use_vars=True, AJ=True):
    if AJ:
        occluded = pred_visibility.mean(dim=0).unsqueeze(0)
        gt_traj_2D = gs_traj_2D.mean(dim=0).unsqueeze(0)
        samples = sample_queries_first(
            occluded.cpu().bool().numpy().squeeze(),
            copy.deepcopy(gt_traj_2D.cpu().numpy().squeeze()),
            ignore_invalid=True)
        samples = {k: v.repeat(gs_traj_2D.shape[0], axis=0) for k, v in samples.items()}
        min_idx = torch.from_numpy(get_smallest_AJ(
            samples['query_points'],
            samples['occluded'],
            samples['target_points'],
            (1-(pred_visibility > 0.5).float()).cpu().numpy(),
            gs_traj_2D.cpu().numpy(),
            W=w,
            H=h)).to(gs_traj_2D.device)
    else:
        torch.use_deterministic_algorithms(False)
        if mean:
            comparison = gs_traj_2D.mean(dim=0).unsqueeze(1)
        else:
            comparison = gs_traj_2D.median(dim=0).values.unsqueeze(1)
        gs_traj_2D_perm = gs_traj_2D.permute(1, 0, 2, 3)
        dists = list()              
        vars = list()                                                                                                                                                                                                             
        for p in range(gs_traj_2D_perm.shape[0]):                                                                                                                                                                                                          
            d = torch.cdist(gs_traj_2D_perm[p].permute(1, 0, 2), comparison[p].permute(1, 0, 2))                                                                                                                                                               
            dists.append(d.mean(0))
            vars.append(torch.var(d, dim=0).squeeze())
        dists = torch.stack(dists).squeeze()
        vars = torch.stack(vars).squeeze()

        # choose
        if use_vars:
            min_idx = choose(vars, min_mean=True)
        else:
            min_idx = choose(dists)
        torch.use_deterministic_algorithms(True)

    return min_idx


def choose(input_dists=None, abs_dist_to_median=False, abs_dist_to_mean=False, min_mean=False):
    if abs_dist_to_median:
        min_idx = torch.sort(torch.abs(input_dists - input_dists.median(dim=1).values.unsqueeze(1)), dim=1).indices[:, 0]  # --> with median bad
    elif abs_dist_to_mean:
        min_idx = torch.sort(torch.abs(input_dists - input_dists.mean(dim=1).unsqueeze(1)), dim=1).indices[:, 0]  # --> with median bad
    elif min_mean:
        min_idx = input_dists.min(dim=1).indices # --> with median bad 
    else:
        raise ValueError
    return min_idx


def get_smallest_AJ(
        query_points: np.ndarray,
        gt_occluded: np.ndarray,
        gt_tracks: np.ndarray,
        pred_occluded: np.ndarray,
        pred_tracks: np.ndarray,
        query_mode: str = 'first',
        norm_factor=256,
        W=256,
        H=256):
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """
    # SCALE to 256  
    sc_pt = np.array(
        [[[[W/norm_factor, H/norm_factor]]]])
    gt_tracks = gt_tracks/sc_pt
    pred_tracks = pred_tracks/sc_pt
    query_points[:, :, 1:] = query_points[:, :, 1:]/sc_pt[0]

    metrics = {}
    # Don't evaluate the query point.  Numpy doesn't have one_hot, so we
    # replicate it by indexing into an identity matrix.
    one_hot_eye = np.eye(gt_tracks.shape[2])
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = one_hot_eye[query_frame] == 0

    # If we're using the first point on the track as a query, don't evaluate the
    # other points.
    if query_mode == "first":
        for i in range(gt_occluded.shape[0]):
            index = np.where(gt_occluded[i] == 0)[0][0]
            evaluation_points[i, :index] = False
    elif query_mode != "strided":
        raise ValueError("Unknown query mode " + query_mode)

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = (
            np.sum(
                np.square(pred_tracks - gt_tracks),
                axis=-1,
            )
            < np.square(thresh)
        )
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(2))
        frac_correct = count_correct / count_visible_points
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(2)
        )
        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(2))
        jaccard = true_positives / (gt_positives + false_positives)
        all_jaccard.append(jaccard)

    all_jaccard =  np.mean(np.stack(all_jaccard, axis=1), axis=1)
    all_frac_within = np.mean(np.stack(all_frac_within, axis=1), axis=1)
    min_idx = np.argmax(all_jaccard, axis=0)
    return min_idx


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    ignore_invalid=False):
    """Package a set of frames and tracks for use in TAPNet evaluations.
    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.
    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1]
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1]
    """
    if ignore_invalid:
        target_occluded = np.zeros_like(target_occluded, dtype=bool)
    valid = np.sum(~target_occluded, axis=1) > 0

    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)
    return {
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def compute_tapvid_metrics(
        query_points: np.ndarray,
        gt_occluded: np.ndarray,
        gt_tracks: np.ndarray,
        pred_occluded: np.ndarray,
        pred_tracks: np.ndarray,
        query_mode: str = 'first',
        norm_factor=256,
        W=256,
        H=256):
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """
    # SCALE to 256  
    sc_pt = np.array(
        [[[[W/norm_factor, H/norm_factor]]]])
    gt_tracks = gt_tracks/sc_pt
    pred_tracks = pred_tracks/sc_pt
    query_points[:, :, 1:] = query_points[:, :, 1:]/sc_pt[0]

    metrics = {}

    # Don't evaluate the query point.  Numpy doesn't have one_hot, so we
    # replicate it by indexing into an identity matrix.
    one_hot_eye = np.eye(gt_tracks.shape[2])
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = one_hot_eye[query_frame] == 0

    # If we're using the first point on the track as a query, don't evaluate the
    # other points.
    if query_mode == "first":
        for i in range(gt_occluded.shape[0]):
            index = np.where(gt_occluded[i] == 0)[0][0]
            evaluation_points[i, :index] = False
    elif query_mode != "strided":
        raise ValueError("Unknown query mode " + query_mode)

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = (
        np.sum(
            np.equal(pred_occluded, gt_occluded) & evaluation_points,
            axis=(1, 2),
        )
        / np.sum(evaluation_points)
    )
    metrics["occlusion_accuracy"] = occ_acc.item()

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = (
            np.sum(
                np.square(pred_tracks - gt_tracks),
                axis=-1,
            )
            < np.square(thresh)
        )
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct.item()
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard.item()
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
        ).item()
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    ).item()
    return metrics


def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    for (a,b) in zip(x.size(), mask.size()):
        # if not b==1: 
        assert(a==b) # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x*mask

    if dim is None:
        numer = torch.sum(prod)
        denom = 1e-10+torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = 1e-10+torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer/denom
    return mean


def reduce_masked_median(x, mask, keep_batch=False):
    # x and mask are the same shape
    assert(x.size() == mask.size())
    device = x.device

    B = list(x.shape)[0]
    x = x.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    if keep_batch:
        x = np.reshape(x, [B, -1])
        mask = np.reshape(mask, [B, -1])
        meds = np.zeros([B], np.float32)
        for b in list(range(B)):
            xb = x[b]
            mb = mask[b]
            if np.sum(mb) > 0:
                xb = xb[mb > 0]
                meds[b] = np.median(xb)
            else:
                meds[b] = np.nan
        meds = torch.from_numpy(meds).to(device)
        return meds.float()
    else:
        x = np.reshape(x, [-1])
        mask = np.reshape(mask, [-1])
        if np.sum(mask) > 0:
            x = x[mask > 0]
            med = np.median(x)
        else:
            med = np.nan
        med = np.array([med], np.float32)
        med = torch.from_numpy(med).to(device)
        return med.float()
    

def compute_metrics(
        H,
        W,
        gs_traj_2D,
        gt_traj_2D,
        valids,
        sur_thr=16,
        thrs=[1, 2, 4, 8, 16],
        norm_factor=256):
    
    device = gs_traj_2D.device
    B, N, S = gt_traj_2D.shape[0], gt_traj_2D.shape[1], gt_traj_2D.shape[2]
    
    # permute number of points and seq len
    gs_traj_2D = gs_traj_2D.permute(0, 2, 1, 3)
    gt_traj_2D = gt_traj_2D.permute(0, 2, 1, 3)
    valids = valids.permute(0, 2, 1)

    # get metrics
    metrics = dict()
    d_sum = 0.0
    if norm_factor is not None:
        sc_pt = torch.tensor(
            [[[W/norm_factor, H/norm_factor]]]).float().to(device)
    else:
        sc_pt = torch.tensor(
            [[[1, 1, 1]]]).float().to(device)

    for thr in thrs:
        # note we exclude timestep0 from this eval
        d_ = (torch.linalg.norm(
            gs_traj_2D[:,1:]/sc_pt - gt_traj_2D[:,1:]/sc_pt, dim=-1, ord=2) < thr).float() # B,S-1,N
        d_ = reduce_masked_mean(d_, valids[:,1:]).item()*100.0
        d_sum += d_
        metrics['d_%d' % thr] = d_
    d_avg = d_sum / len(thrs)
    metrics['d_avg'] = d_avg
    
    dists = torch.linalg.norm(gs_traj_2D/sc_pt - gt_traj_2D/sc_pt, dim=-1, ord=2) # B,S,N
    dist_ok = 1 - (dists > sur_thr).float() * valids # B,S,N
    survival = torch.cumprod(dist_ok, dim=1) # B,S,N
    metrics['survival'] = torch.mean(survival).item()*100.0

    # get the median l2 error for each trajectory
    dists_ = dists.permute(0,2,1).reshape(B*N,S)
    valids_ = valids.permute(0,2,1).reshape(B*N,S)
    median_l2 = reduce_masked_median(dists_, valids_, keep_batch=True)
    metrics['median_l2'] = median_l2.mean().item()

    return metrics


def meshgrid2d(B, Y, X, stack=False, device='cuda:0', on_chans=False):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        if on_chans:
            grid = torch.stack([grid_x, grid_y], dim=1)
        else:
            grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


def get_xy_grid(H, W, N=2048, B=1, device='cuda:0', mask=None):
    # pick N points to track; we'll use a uniform grid
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = meshgrid2d(B, N_, N_, stack=False, device=device)
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
    xy0 = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
    if mask is not None:
        xy0 = from_fg_mask(mask, W, H, device, xy0=xy0.long())
    return xy0


def from_fg_mask(mask, W, H, device, xy0=None):
    mask = cv2.resize(
            mask.cpu().numpy().astype(float),
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        )
    mask = torch.from_numpy(mask).to(device)
    mask = ~(mask.bool())
    candidates = torch.zeros_like(mask, dtype=bool, device=device)
    candidates[xy0[0, :, 1], xy0[0, :, 0]] = True
    candidates = mask & candidates
    candidates = torch.nonzero(candidates)
    candidates = torch.stack([candidates[:, 1], candidates[:, 0]], dim=1)
    return candidates.unsqueeze(0)


