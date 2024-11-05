import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset
import imageio
import json
from sklearn.decomposition import PCA


class PanopticSportsDataset(GradSLAMDataset):
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
        depth_type='Dynamic3DGaussians',
        embedding_dim: Optional[int] = 64,
        start_from_complete_pc=False,
        do_transform=False,
        novel_view_mode=None,
        **kwargs,
    ): 
        self.input_folder = os.path.join(basedir, sequence)

        start = start + 2
        self.start_from_complete_pc = start_from_complete_pc
        self.depth_type = depth_type
        
        self.sequence = sequence
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        self.do_transform = do_transform
        self.novel_view_mode = novel_view_mode

        super().__init__(config_dict,
            every_x_frame=every_x_frame,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dim=embedding_dim,
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
        bg = ~(imageio.imread(bg_path))
        return bg
    
    def _load_instseg(self, instseg_path):
        instseg = imageio.imread(instseg_path)
        return instseg
    
    def get_filepaths(self):
        # get color paths
        color_paths = natsorted(glob.glob(f"{self.input_folder}/*.jpg"))

        # get depth paths
        if self.depth_type != 'Dynamic3DGaussians':
            depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth*.npy"))
        else:
            input_folder = self.input_folder.replace("/scratch/jseidens/data/data", "../Dynamic3DGaussians/rendered")
            input_folder = self.input_folder.replace("/data3/jseidens/data", "../Dynamic3DGaussians/rendered")
            depth_paths = natsorted(glob.glob(f"{input_folder}/depth*.npy"))
        
        # get background paths
        bg_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'seg')}/*.png"))
        instseg_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'seg')}/*.png"))
        
        # get embedding paths
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'feats')}/*dino_img_quat_4_1.npy"))
            features = np.load(embedding_paths[0])
            if self.embedding_dim != features.shape[2]:
                pca = PCA(n_components=self.embedding_dim)
                self.embedding_downscale = pca.fit(features.reshape(-1, features.shape[2]))
            else:
                print('Features already have the right size...')
                self.embedding_downscale = None

        return color_paths, depth_paths, embedding_paths, bg_paths, instseg_paths
            
    def load_poses(self):
        if self.do_transform:
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

        self.first_time_w2c = torch.linalg.inv(poses[0])
    
        return poses
