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


class DavisDynoSplatamDataset(GradSLAMDataset):
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
        with open('/scratch/jseidens/data/tapvid_davis/tapvid_davis.pkl', 'rb') as f:
            data = pickle.load(f)
            print(data.keys())
            dat = data[sequence]

        rgbs = dat['video'] # list of H,W,C uint8 images
        # desired_height = rgbs.shape[1]
        # desired_width = rgbs.shape[2]
        self.start_frame = 0 #18
        self.max_len = rgbs.shape[0]
        del rgbs
        # trajs = dat['points'] # N,S,2 array
        # valids = 1-dat['occluded'] # N,S array
        
        self.feats_224 = feats_224
        self.input_folder = os.path.join(basedir, sequence)
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
        try:
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
        except:
            self.support_trajs = None

    def get_instsegpaths(self):
        instseg_paths = natsorted(glob.glob(f"{self.input_folder.replace('JPEGImages', 'Annotations')}/*.png"))[self.start_frame:self.max_len]
        # instseg_paths = natsorted(glob.glob(f"{self.input_folder}/*sam_big_area.npy"))
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
        color_paths = natsorted(glob.glob(f"{self.input_folder}/*.jpg"))[self.start_frame:self.max_len]
        depth_paths = natsorted(glob.glob(f"{self.input_folder.replace('JPEGImages', 'DEPTH')}/depth*.npy"))[self.start_frame:self.max_len]
        embedding_paths = None
        if self.load_embeddings:
            if not self.feats_224:
                embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('JPEGImages', 'FEATS')}/*dino_img_quat_4_1.npy"))[self.start_frame:self.max_len]
            else:
                embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('JPEGImages', 'FEATS')}/*dino_img_quat_4_2_64_224.npy"))[self.start_frame:self.max_len]
            features = np.load(embedding_paths[0])
            if self.embedding_dim != features.shape[2]:
                pca = PCA(n_components=self.embedding_dim)
                self.embedding_downscale = pca.fit(features.reshape(-1, features.shape[2]))
            else:
                print('Features already have the right size...')
                self.embedding_downscale = None

        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        for i in range(self.num_imgs):
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.eye(4).float()
            poses.append(c2w)
        return poses
