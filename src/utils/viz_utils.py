# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Visualization utility functions."""

import colorsys
import random
from typing import List, Optional, Sequence, Tuple

from absl import logging
# import jax
# import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import os
from matplotlib.collections import LineCollection
import imageio
from matplotlib import cm
import cv2
import flow_vis
import glob
import torch
import json


color_map = cm.get_cmap("jet")


def make_vid(input_path): 
    images = list()
    img_paths = glob.glob(f'{input_path}/*')
    print(input_path, len(img_paths))
    for f in sorted(img_paths):
        if 'mp4' in f:
            continue
        images.append(imageio.imread(f))

    imageio.mimwrite(os.path.join(input_path, 'vid_trails.mp4'), np.stack(images), quality=8, fps=10)
    for f in img_paths:
        if 'mp4' in f:
            continue
        os.remove(f)
        

def get_w2c(y_angle=0., center_dist=2.4, cam_height=1.3):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    return w2c

def get_circle(num_frames, device, avg_w2c, rots=1, rads=0.5, zrate=0):
    w2cs = torch.stack([avg_w2c] * num_frames)
    thetas = torch.linspace(
        0, 2 * torch.pi * rots, num_frames + 1, device=device)[:-1]
    positions = (
        torch.stack(
            [
                torch.cos(thetas),
                -torch.sin(thetas),
                -torch.sin(thetas * zrate),
            ],
            dim=-1,
        )
        * rads
    )
    positions += rads*0.5
    w2cs[:, :3, -1] += positions
    return w2cs
    

def get_cam_poses(novel_view_mode, dataset, config, num_frames, device, params):
    # w2c = torch.linalg.inv(pose)
    # train_c2ws = dataset.transformed_poses.to(device)
    train_c2ws = torch.tile(
        dataset.transformed_poses[0].unsqueeze(0),
        (dataset.transformed_poses.shape[0], 1, 1)).to(device)
    if novel_view_mode == 'circle' or novel_view_mode == 'zoom_out':
        # params['means3D'] = (N, T, 3)
        avg_w2c = params['w2c'] @ torch.linalg.inv(train_c2ws[0])

        # zoom out a bit
        scene_center = params['means3D'][:, :, :].reshape(-1, 3).mean(dim=0)
        lookat = scene_center - avg_w2c[:3, -1]
        if avg_w2c.sum() == 4:
            if 'DAVIS' in config['data']['basedir']:
                lookat = torch.tensor([0, 0, -2]).to(device)
            elif 'iphone' in config['data']['basedir']:
                lookat = torch.tensor([0, 0, -0.2]).to(device)
            else:
                lookat = torch.tensor([0, 0, -0.2]).to(device)
        avg_w2c[:3, -1] -= 1 * lookat
        rads = 0 if novel_view_mode == 'zoom_out' else 0.1
        w2cs = get_circle(num_frames, device, avg_w2c, rads=rads, rots=3)
        poses = torch.linalg.inv(w2cs)
        name = novel_view_mode
        
    elif novel_view_mode == 'test_cam':
        meta_path = os.path.join(
            config['data']['basedir'],
            os.path.dirname(os.path.dirname(config['data']['sequence'])),
            'test_meta.json')
        with open(meta_path, 'r') as jf:
            data = json.load(jf)

        # [0, 10, 15, 30] --> 10 (above) or 0 (side) most interesting I'd say
        test_cam = 10 # random.choice(data['cam_id'][0])
        print(f"Using test cam {test_cam}.")
        test_cam_idx = data['cam_id'][0].index(test_cam)
        test_cam_w2c = torch.tensor(data['w2c'][0][test_cam_idx]).float()
        test_cam_c2w = torch.linalg.inv(test_cam_w2c)
        poses = [test_cam_c2w.to(device)] * num_frames
        name = f'{test_cam}'
     
    return poses, name



def vis_trail(results_dir, data, clip=True, pred_visibility=None, vis_traj=True, traj_len=10, fg_only=True):
    """
    This function calculates the median motion of the background, which is subsequently
    subtracted from the foreground motion. This subtraction process "stabilizes" the camera and
    improves the interpretability of the foreground motion trails.
    """
    points = data['points']
    per_time = len(points.shape) == 4
    if not per_time:
        points = np.expand_dims(points, axis=0)

    # points shape B, N x T x 2
    if points.sum() == 0:
        points = data['points_projected']
    B, N, T, _ = points.shape
    rgb = data['video'][:T] # T x 480 x 854 x 3
    h, w, _ = rgb[0].shape
    occluded = data['occluded'][:, :T]
    occluded = 1 - occluded

    scale_factor = np.array([w, h])
    points = points * scale_factor
    os.makedirs(results_dir, exist_ok=True)
    
    pred_visibility = pred_visibility.transpose(1, 0)
    points = points.transpose(0, 2, 1, 3)
    num_imgs, num_pts = points.shape[1:3] # T x N x 2
    frames = []
    for i in range(num_imgs):
        # kpts = kpts_foreground - np.median(kpts_background - kpts_background[i], axis=1, keepdims=True)
        img_curr = rgb[i]
        if vis_traj:
            for t in range(max(0, i-traj_len), i):
                img1 = img_curr.copy()
                # changing opacity
                alpha = max(1 - 0.9 * ((i - t) / ((i + 1) * .99)), 0.1)

                for j in range(num_pts):
                    if not pred_visibility[i, j]:
                        continue
                    color = np.array(color_map(j/max(1, float(num_pts - 1)))[:3]) * 255

                    color_alpha = 1

                    hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
                    color = colorsys.hsv_to_rgb(hsv[0], hsv[1]*color_alpha, hsv[2])
                    if not per_time:
                        pt1 = points[0, t, j]
                        pt2 = points[0, t+1, j]
                    else:
                        pt1 = points[i, t, j]
                        pt2 = points[i, t+1, j]
                    p1 = (int(round(pt1[0])), int(round(pt1[1])))
                    p2 = (int(round(pt2[0])), int(round(pt2[1])))
                    # if p2[0] > 10000 or p2[1] > 10000:
                    #     continue
                    cv2.line(img1, p1, p2, color, thickness=1, lineType=16)

                img_curr = cv2.addWeighted(img1, alpha, img_curr, 1 - alpha, 0)

        for j in range(num_pts):
            if not pred_visibility[i, j]:
                continue
            color = np.array(color_map(j/max(1, float(num_pts - 1)))[:3]) * 255
            if not per_time:
                pt1 = points[0, i, j]
            else:
                pt1 = points[i, i, j]
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            # if p1[0] > 10000 or p1[1] > 10000:
            #     continue
            if traj_len > 0:
                cv2.circle(img_curr, p1, 2, color, -1, lineType=16)
            else:
                cv2.circle(img_curr, p1, 3, color, -1, lineType=16)


        frames.append(img_curr.astype(np.uint8))

    if not fg_only:
        save_path = os.path.join(results_dir, f'vid_trails_{traj_len}_fg_and_bg.mp4')
    else:
        save_path = os.path.join(results_dir, f'vid_trails_{traj_len}.mp4')

    imageio.mimwrite(save_path, frames, quality=8, fps=10)
    print('stored vis', save_path)

# flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
