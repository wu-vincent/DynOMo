import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .replica import ReplicaDataset
from .basedataset import GradSLAMDataset
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
        get_pc_jono=False,
        **kwargs,
    ):
        self.get_pc_jono = get_pc_jono
        if self.get_pc_jono:
            kwargs['relative_pose'] = False
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
            bg_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'seg')}/*.png"))[2:35]
        else:
            bg_paths = None
        return bg_paths
    
    def get_instsegpaths(self):
        # instseg_paths = natsorted(glob.glob(f"{self.input_folder}/*sam_big_area.npy"))[2:35]
        instseg_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'seg')}/*.png"))[2:35]
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
        color_paths = natsorted(glob.glob(f"{self.input_folder}/*.jpg"))[2:35]
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth*.npy"))[2:35]
        embedding_paths = None
        if self.load_embeddings:
            # embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'feats')}/*gs_*.npy"))[2:35]
            # embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'feats')}/*sam_img.npy"))[2:35]
            # embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'feats')}/*sam_img_quat.npy"))[2:35]
            embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'feats')}/*dino_img_quat.npy"))[2:35]
            if len(embedding_paths) != len(color_paths):
                print("Using SAM features")
                embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'feats')}/*sam_img_quat.npy"))[2:35]
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
        self.support_trajs = np.load(
            f"{self.input_folder.replace('ims', 'support_trajs')}/48_trajs.npy").squeeze()[2:35]
        # sclae 
        support_traj_shape = (352,640)
        scale_factors = (
            self.desired_width/support_traj_shape[1], self.desired_height/support_traj_shape[0])
        self.support_trajs[:, :, :, 0] = self.support_trajs[:, :, :, 0] * scale_factors[0]
        self.support_trajs[:, :, :, 1] = self.support_trajs[:, :, :, 1] * scale_factors[1]
        self.support_trajs = np.round(self.support_trajs)

        # clip to image boundaries
        self.support_trajs[:, :, :, 0] = np.clip(
            self.support_trajs[:, :, :, 0], a_min=0, a_max=self.desired_width-1)
        self.support_trajs[:, :, :, 1] = np.clip(
            self.support_trajs[:, :, :, 1], a_min=0, a_max=self.desired_height-1)

    def load_poses(self):
        if False: # self.get_pc_jono:
            basedir = os.path.dirname(os.path.dirname(self.input_folder))
            cam_id = int(os.path.basename(self.input_folder))
            with open(f'{basedir}/meta.json', 'r') as jf:
                data = json.load(jf)
            idx = data['cam_id'][0].index(cam_id)
            w2c = torch.tensor(data['w2c'][0][idx])
            c2w = torch.linalg.inv(w2c)
        else:
            c2w = torch.eye(4).float()

        poses = []
        for i in range(self.num_imgs):
            poses.append(c2w)
        return poses