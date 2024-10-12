import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset
import imageio
import pickle
from sklearn.decomposition import PCA


class RGBDynoSplatamDataset(GradSLAMDataset):
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
        load_instseg=True,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        load_support_trajs=False,
        feats_224=False,
        **kwargs,
    ):  
        with open('/data3/jseidens/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl', 'rb') as f:
            data = pickle.load(f)
            dat = data[int(sequence)]

        self.rgbs = dat['video'] # list of H,W,C uint8 images
        # desired_height = rgbs.shape[1]
        # desired_width = rgbs.shape[2]
        self.start_frame = 0 #18
        self.max_len = self.rgbs.shape[0]
        # trajs = dat['points'] # N,S,2 array
        # valids = 1-dat['occluded'] # N,S array
        
        self.feats_224 = feats_224
        self.input_folder = os.path.join(basedir, 'ims', '{:04d}'.format(sequence))
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        self.load_instseg = True
        super().__init__(config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            load_instseg=load_instseg,
            **kwargs,
        )
        print(f"Length of sequence {len(self.color_paths)}")
    
    def load_support_trajs(self):
        self.support_trajs = None

    def get_instsegpaths(self):
        instseg_paths = list(range(len(self.rgbs)))
        return instseg_paths
    
    def get_bg_paths(self):
        return self.instseg_paths
    
    def read_embedding_from_file(self, embedding_path):
        embedding = np.load(embedding_path)
        if self.embedding_downscale is not None:
            shape = embedding.shape
            embedding = self.embedding_downscale.transform(embedding.reshape(-1, shape[2]))
            embedding = embedding.reshape((shape[0], shape[1], self.embedding_dim))
        return torch.from_numpy(embedding)

    def _load_instseg(self, instseg_path):
        instseg = np.zeros((self.desired_height, self.desired_height), dtype=int)
        return instseg
    
    def _load_bg(self, bg_path):
        # instseg = np.load(instseg_path, mmap_mode="r").astype(dtype=np.int64)
        bg = np.ones((self.desired_height, self.desired_height), dtype=int)
        return bg
    
    def get_filepaths(self):
        self.rgbs = self.rgbs[self.start_frame:self.max_len]
        color_paths = list(range(len(self.rgbs)))
        depth_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'depth')}/*.npy"))[self.start_frame:self.max_len]
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'feats')}/*dino_img_quat_4_1_32_128_128.npy"))[self.start_frame:self.max_len]
            features = np.load(embedding_paths[0])
            if self.embedding_dim != features.shape[2]:
                pca = PCA(n_components=self.embedding_dim)
                self.embedding_downscale = pca.fit(features.reshape(-1, features.shape[2]))
            else:
                print('Features already have the right size...')
                self.embedding_downscale = None

        return self.rgbs, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        for i in range(self.num_imgs):
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.eye(4).float()
            poses.append(c2w)
        return poses
