import json
import torch
import os
from natsort import natsorted
import glob
import imageio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import argparse
import os
import shutil
import sys
from importlib.machinery import SourceFileLoader
import pickle
import mediapy as media


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

from utils.recon_helpers import setup_camera
from utils.tapnet_utils_viz import paint_point_track, plot_tracks_v2


def vis_jono(store_trajs, data='test', cam_id=0, vis=False):

    results_dir = 'vis_gt/jono'
    traj_len = 10
    cam_idxs = os.listdir('data/data/tennis/ims')
    with open('data/data/annotations/3dgt.json', 'r') as jf:
        gt_3d_all = json.load(jf)
    with open('data/data/annotations/2dgt.json', 'r') as jf:
        gt_2d_all = json.load(jf)

    for cam_id in cam_idxs:
        all_2d_gt_converted_from_2d_cam = dict()
        for sequence in os.listdir('data/data'):
            if sequence == 'annotations':
                continue

            # combine train and test meta
            if not os.path.isfile(f'data/data/{sequence}/meta.json'):
                with open(f'data/data/{sequence}/train_meta.json', 'r') as jf:
                    meta = json.load(jf)
                with open(f'data/data/{sequence}/test_meta.json', 'r') as jf:
                    test_meta = json.load(jf)
                    for k in meta.keys():
                        if type(meta[k]) == list:
                            for time in range(len(meta[k])):
                                meta[k][time].extend(test_meta[k][time])

                with open(f'data/data/{sequence}/meta.json', 'w') as jf:
                    json.dump(meta, jf)
            
            # load joint meta information
            with open(f'data/data/{sequence}/meta.json', 'r') as jf:
                meta = json.load(jf)
            
            # get meta variables
            cam_idx = meta['cam_id'][0].index(int(cam_id))
            time_idx = 0
            w2c = torch.tensor(meta['w2c'][time_idx][cam_idx])
            k = torch.tensor(meta['k'][time_idx][cam_idx])
            h = meta['h']
            w = meta['w']

            print(f'Processing Sequence {sequence} on cam {cam_id}...')

            # fill cam dict with sequence data
            all_2d_gt_converted_from_2d_cam[sequence] = dict()
            rgbs = np.stack([imageio.imread(p) for p in glob.glob(f'data/data/{sequence}/ims/{cam_id}/*.jpg')])
            all_2d_gt_converted_from_2d_cam[sequence]['video'] = rgbs
            gt_3d = gt_3d_all[sequence]     

            # get 2d points             
            seq_len = 148
            points_list = list()
            occluded_list = list()
            for traj in gt_3d.keys():
                if str(cam_id) not in gt_2d_all[sequence][traj].keys():
                    occluded = np.ones(seq_len)
                    points = np.zeros([seq_len, 2])
                else:
                    occluded = np.zeros(seq_len)
                    points = np.asarray(gt_2d_all[sequence][traj][str(cam_id)])
                    points[:, 0] = (points[:, 0] + 1)/w
                    points[:, 1] = (points[:, 1] + 1)/h

                points_list.append(points)
                occluded_list.append(occluded)
            points_list = np.stack(points_list)
            occluded_list = np.stack(occluded_list)

            all_2d_gt_converted_from_2d_cam[sequence]['points'] = points_list
            all_2d_gt_converted_from_2d_cam[sequence]['occluded'] = occluded_list

            # get projected 2d points from 3d points
            cam = setup_camera(w, h, k, w2c)

            points_xy_list = list()
            traj_list = list()
            for traj_name, traj in gt_3d.items():
                traj = torch.tensor(traj)
                points_xy = cam.projmatrix.cpu().squeeze().T.matmul(
                    torch.cat([traj, torch.ones(traj.shape[0], 1)], dim=1).T)
                points_xy = points_xy / points_xy[3, :]
                points_xy = points_xy[:2].T
                points_xy[:, 0] = ((points_xy[:, 0]+1)*w-1) * 0.5
                points_xy[:, 1] = ((points_xy[:, 1]+1)*h-1) * 0.5
                points_xy[:, 0] = (points_xy[:, 0] + 1)/w
                points_xy[:, 1] = (points_xy[:, 1] + 1)/h
                traj_list.append(traj.numpy())
                points_xy_list.append(points_xy)
            
            points_xy_list = np.stack([p.numpy() for p in points_xy_list])
            traj_list = np.stack(traj_list)

            # add projected points and 3D trajectories
            all_2d_gt_converted_from_2d_cam[sequence]['points_projected'] = points_xy_list
            all_2d_gt_converted_from_2d_cam[sequence]['trajs'] = traj_list

            # visualize
            if vis:
                store_figs(traj_len, sequence, cam_id, points_xy_list, results_dir)

        # store tapvid format
        with open('data/data/annotations/traj_tap_vid_format_gs_{:04d}.pickle'.format(int(cam_id)), 'wb') as pf:
            pickle.dump(all_2d_gt_converted_from_2d_cam, pf)

def store_figs(traj_len, sequence, cam_id, points_xy_list, results_dir):
    os.makedirs(os.path.join(results_dir, sequence), exist_ok=True)
    # pad with zeros
    points_xy_list_padded = list()
    for i, points_xy in enumerate(points_xy_list):
        points_xy = torch.cat([
            torch.ones([traj_len, 2]) * points_xy[0], points_xy], dim=0)
        points_xy_list_padded.append(points_xy.numpy())

    points_xy_list = np.stack(points_xy_list_padded)
    for time, img in enumerate(natsorted(glob.glob(f'data/data/{sequence}/ims/{cam_id}/*.jpg'))):
        img = imageio.imread(img)

        fig, ax = plt.subplots()
        ax.imshow(img)
        
        for i in range(points_xy_list.shape[0]):
            x = np.clip(points_xy_list[i, time:time+traj_len, 0], a_min=0, a_max=w-1)
            y = np.clip(points_xy_list[i, time:time+traj_len, 1], a_min=0, a_max=h-1)
            color_len = np.arange(traj_len-1)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a continuous norm to map from data points to colors
            norm = plt.Normalize(color_len.min(), color_len.max())
            lc = LineCollection(segments, cmap='hsv', norm=norm)
            # Set the values used for colormapping
            lc.set_array(color_len)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            plt.scatter(x[0], y[0], c='white', marker='D', edgecolors='black', s=2, linewidths=0.5)
            # fig.colorbar(line, ax=ax)

            # plt.plot(x, y, color='orange') #, marker="D")

        # Add the patch to the Axes
        plt.axis('off')
        plt.savefig(os.path.join(results_dir, sequence, "gs_{:04d}.png".format(time)), bbox_inches='tight', pad_inches = 0)


def vis_davis(store_trajs):
    results_dir = 'vis_gt/davis'
    traj_len = 10
    with open('data/tapvid_davis/tapvid_davis.pkl', 'rb') as jf:
        gt = pickle.load(jf)

    for sequence in gt.keys():
        points = gt[sequence]['points'] # N x T x 2
        N, T, _ = points.shape
        rgb = gt[sequence]['video'] # T x 480 x 854 x 3
        h, w, _ = rgb[0].shape
        occluded = gt[sequence]['occluded']
        scale_factor = np.array([w, h])
        points = points * scale_factor

        print(f'Processing Sequence {sequence}...')

        painted_frames = paint_point_track(rgb,
            points,
            ~occluded,)
        painted_frames = plot_tracks_v2(
                rgb,
                points,
                occluded)
        print(painted_frames.shape)
        os.makedirs(os.path.join(results_dir, sequence), exist_ok=True)
        # media.write_video(os.path.join(results_dir, sequence, 'vid.mp4'), painted_frames, fps=25)

        # pad with zeros
        for time, img in enumerate(painted_frames):
            fig, ax = plt.subplots()
            ax.imshow(img)
            from_time = max(0, time-traj_len)
            for i in range(points.shape[0]):
                x = np.clip(points[i, from_time:time+1, 0], a_min=0, a_max=w-1)
                y = np.clip(points[i, from_time:time+1, 1], a_min=0, a_max=h-1)

                color_len = np.arange(traj_len-1)
                pts = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([pts[:-1], pts[1:]], axis=1)

                # Create a continuous norm to map from data points to colors
                norm = plt.Normalize(color_len.min(), color_len.max())
                lc = LineCollection(segments, cmap='hsv', norm=norm)
                # Set the values used for colormapping
                lc.set_array(color_len)
                lc.set_linewidth(2)
                line = ax.add_collection(lc)

            plt.axis('off')
            plt.savefig(os.path.join(results_dir, sequence, "gs_{:04d}.png".format(time)), bbox_inches='tight', pad_inches = 0)
            plt.close(fig)


if __name__ == "__main__":
    data = 'jono'
    if data == 'jono':
        vis_jono(store_trajs=True)
    elif data == 'DAVIS':
        vis_davis(store_trajs=True)