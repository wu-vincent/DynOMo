import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .replica import ReplicaDataset
import imageio
import pickle


class DavisDynoSplatamDataset(ReplicaDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        load_support_trajs=False,
        **kwargs,
    ):  
        with open('data/tapvid_davis/tapvid_davis.pkl', 'rb') as f:
            data = pickle.load(f)
            dat = data[sequence]

        rgbs = dat['video'] # list of H,W,C uint8 images
        self.max_len = rgbs.shape[0]
        trajs = dat['points'] # N,S,2 array
        valids = 1-dat['occluded'] # N,S array

        desired_height = rgbs.shape[1]
        desired_width = rgbs.shape[2]
                
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        super().__init__(config_dict,
            basedir,
            sequence,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        self.instseg_paths = self.get_instsegpaths()
        self.bg_paths = self.get_bg_paths()
        self.load_instseg = True
        self.load_support_trajs()
        print(f"Length of sequence {len(self.color_paths)}")
    
    def load_support_trajs(self):
        # load
        self.support_trajs = np.load(f"{self.input_folder.replace('JPEGImages', 'SUPPORT_TRAJS')}/48_trajs.npy").squeeze()
        # sclae 
        support_traj_shape = (512,896)
        scale_factors = (
            self.desired_width/support_traj_shape[1], self.desired_height/support_traj_shape[0])
        self.support_trajs[:, :, :, 0] = self.support_trajs[:, :, :, 0] * scale_factors[0]
        self.support_trajs[:, :, :, 1] = self.support_trajs[:, :, :, 1] * scale_factors[1]
        self.support_trajs = np.round(self.support_trajs)
        # clip to image boundaries
        self.support_trajs[:, :, :, 0] = np.clip(self.support_trajs[:, :, :, 0], a_min=0, a_max=self.desired_width-1)
        self.support_trajs[:, :, :, 1] = np.clip(self.support_trajs[:, :, :, 1], a_min=0, a_max=self.desired_height-1)

    def get_instsegpaths(self):
        instseg_paths = natsorted(glob.glob(f"{self.input_folder.replace('JPEGImages', 'Annotations')}/*.png"))[:self.max_len]
        # instseg_paths = natsorted(glob.glob(f"{self.input_folder}/*sam_big_area.npy"))
        return instseg_paths
    
    def get_bg_paths(self):
        return self.instseg_paths[:self.max_len]
    
    def read_embedding_from_file(self, idx):
        embedding = torch.from_numpy(np.load(self.embedding_paths[idx]).astype(dtype=np.int64))
        embedding = torch.cat([torch.zeros(480, 80, 384), embedding, torch.zeros(480, 80, 384)], dim=1)
        return embedding

    def _load_instseg(self, instseg_path):
        # instseg = np.load(instseg_path, mmap_mode="r").astype(dtype=np.int64)
        instseg = np.asarray(imageio.imread(instseg_path), dtype=int)
        instseg[:, :, 1][instseg[:, :, 1]!=0] += 50
        instseg[:, :, 2][instseg[:, :, 2]!=0] += 50
        instseg = np.sum(instseg, axis=2)
        return instseg
    
    def _load_bg(self, bg_path):
        # instseg = np.load(instseg_path, mmap_mode="r").astype(dtype=np.int64)
        bg = np.asarray(imageio.imread(bg_path), dtype=int)
        bg[:, :, 1][bg[:, :, 1]!=0] += 50
        bg[:, :, 2][bg[:, :, 2]!=0] += 50
        bg = np.sum(bg, axis=2)
        bg = bg == 0
        return bg
    
    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/*.jpg"))[:self.max_len]
        depth_paths = natsorted(glob.glob(f"{self.input_folder.replace('JPEGImages', 'DEPTH')}/depth*.npy"))[:self.max_len]
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('JPEGImages', 'Embeddings')}/embeddings*.png"))[:self.max_len]
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        for i in range(self.num_imgs):
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.eye(4).float()
            poses.append(c2w)
        return poses