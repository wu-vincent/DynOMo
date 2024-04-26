import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .replica import ReplicaDataset
import imageio
import json
from sklearn.decomposition import PCA


class JonoDynoSplatamDataset(ReplicaDataset):
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
        embedding_dim: Optional[int] = 64,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.sequence = sequence
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
    
    def get_bg_paths(self):
        if os.path.isdir(f"{self.input_folder.replace('ims', 'seg')}"):
            bg_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'seg')}/*.png"))[2:36]
        else:
            bg_paths = None
        return bg_paths
    
    def get_instsegpaths(self):
        # instseg_paths = natsorted(glob.glob(f"{self.input_folder}/*sam_big_area.npy"))[2:36]
        instseg_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'seg')}/*.png"))[2:36]
        return instseg_paths
    
    def read_embedding_from_file(self, embedding_path):
        embedding = np.load(embedding_path)
        if self.embedding_downscale is not None:
            shape = embedding.shape
            embedding = self.embedding_downscale.transform(embedding.reshape(-1, shape[2]))
            embedding = embedding.reshape((shape[0], shape[1], self.embedding_dim))
        return torch.from_numpy(embedding)

    def _load_instseg(self, instseg_path):
        if 'npy' in instseg_path:
            instseg = np.load(instseg_path, mmap_mode="r").astype(dtype=np.int64)
        else:
            instseg = np.asarray(imageio.imread(instseg_path), dtype=int)
        return instseg

    def _load_bg(self, bg_path):
        bg = ~(imageio.imread(bg_path))
        return bg
    
    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/*.jpg"))[2:36]
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth*.npy"))[2:36]
        embedding_paths = None
        if self.load_embeddings:
            # embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'feats')}/*sam_features.npy"))[2:]
            embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'feats')}/*sam_img.npy"))[2:36]
            features = np.load(embedding_paths[0])
            if self.embedding_dim != features.shape[2]:
                pca = PCA(n_components=self.embedding_dim)
                self.embedding_downscale = pca.fit(features.reshape(-1, features.shape[2]))
            else:
                print('Features already have the right size...')
                self.embedding_downscale = None
        return color_paths, depth_paths, embedding_paths
    
    def load_support_trajs(self):
        # load
        self.support_trajs = np.load(f"{self.input_folder.replace('ims', 'support_trajs')}/48_trajs.npy").squeeze()[2:]
        # sclae 
        support_traj_shape = (352,640)
        scale_factors = (
            self.desired_width/support_traj_shape[1], self.desired_height/support_traj_shape[0])
        self.support_trajs[:, :, :, 0] = self.support_trajs[:, :, :, 0] * scale_factors[0]
        self.support_trajs[:, :, :, 1] = self.support_trajs[:, :, :, 1] * scale_factors[1]
        self.support_trajs = np.round(self.support_trajs)

        # clip to image boundaries
        self.support_trajs[:, :, :, 0] = np.clip(self.support_trajs[:, :, :, 0], a_min=0, a_max=self.desired_width-1)
        self.support_trajs[:, :, :, 1] = np.clip(self.support_trajs[:, :, :, 1], a_min=0, a_max=self.desired_height-1)

    def load_poses(self):
        poses = []
        for i in range(self.num_imgs):
            # c2w = torch.tensor([
            #     [0.05361535669, 0.02356944191, 0.9982834642, 0.06302501384],
            #     [0.6379545649, 0.7682889931, -0.05240225469, 1.060031278],
            #     [-0.7682052894, 0.6396690586, 0.02615585534, 3.608606891],
            #     [0.0, 0.0, 0.0, 1.0]])
            c2w = torch.eye(4).float()
            poses.append(c2w)
        return poses