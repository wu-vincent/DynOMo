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


class DavisDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        every_x_frame: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dim: Optional[int] = 512,
        online_depth=None,
        online_emb=None,
        **kwargs,
    ):  
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(basedir))), 'tapvid_davis/tapvid_davis.pkl'), 'rb') as f:
            data = pickle.load(f)
            dat = data[sequence]

        rgbs = dat['video'] # list of H,W,C uint8 images
        start = 0 #18
        end = rgbs.shape[0]
        del rgbs
        
        self.input_folder = os.path.join(basedir, sequence)
        super().__init__(config_dict,
            every_x_frame=every_x_frame,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dim=embedding_dim,
            online_depth=online_depth,
            online_emb=online_emb,
            **kwargs,
        )
        print(f"Length of sequence {len(self.color_paths)}")
    
    def read_embedding_from_file(self, embedding_path):
        embedding = np.load(embedding_path)
        if self.embedding_downscale is not None:
            shape = embedding.shape
            embedding = self.embedding_downscale.transform(embedding.reshape(-1, shape[2]))
            embedding = embedding.reshape((shape[0], shape[1], self.embedding_dim))
        return torch.from_numpy(embedding)
    
    def _load_bg(self, bg_path):
        bg = np.asarray(imageio.imread(bg_path), dtype=int)
        bg[:, :, 1][bg[:, :, 1]!=0] += 50
        bg[:, :, 2][bg[:, :, 2]!=0] += 50
        bg = np.sum(bg, axis=2)
        bg = bg == 0
        return bg
    
    def _load_instseg(self, instseg_path):
        instseg = np.asarray(imageio.imread(instseg_path), dtype=int).sum(-1)
        return instseg
    
    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/*.jpg"))[self.start:self.end]
        if self.online_depth is None:
            depth_paths = natsorted(glob.glob(f"{self.input_folder.replace('JPEGImages', 'Depth')}/*.npy"))[self.start:self.end]
        else:
            depth_paths = None
        bg_paths = natsorted(glob.glob(f"{self.input_folder.replace('JPEGImages', 'Annotations')}/*.png"))[self.start:self.end]
        instseg_paths = natsorted(glob.glob(f"{self.input_folder.replace('JPEGImages', 'Annotations')}/*.png"))[self.start:self.end]

        if self.load_embeddings and self.online_emb is None:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('JPEGImages', 'Feats')}/*.npy"))[self.start:self.end]
            features = np.load(embedding_paths[0])
            if self.embedding_dim != features.shape[2]:
                pca = PCA(n_components=self.embedding_dim)
                self.embedding_downscale = pca.fit(features.reshape(-1, features.shape[2]))
            else:
                print('Features already have the right size...')
                self.embedding_downscale = None
        else:
            embedding_paths = None

        return color_paths, depth_paths, embedding_paths, bg_paths, instseg_paths

    def load_poses(self):
        poses = []
        for i in range(self.num_imgs):
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.eye(4).float()
            poses.append(c2w)
        self.first_time_w2c = torch.linalg.inv(poses[0])
        return poses
