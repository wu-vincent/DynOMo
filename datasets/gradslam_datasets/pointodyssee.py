import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted
import imageio

from .replica import ReplicaDataset


class PointOdysseeDynoSplatamDataset(ReplicaDataset):
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
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")

        self.anno = np.load(f'{self.input_folder}/anno.npz')

        self.intrinsics = self.anno['intrinsics'][0]

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
            precomp_intrinsics=True,
            **kwargs,
        )
        self.instseg_paths = self.get_instsegpaths()
        self.load_instseg = True
        embedding_path = f"{self.input_folder}/Features/crop-256-dinov2-01.npy"
        if self.load_embeddings:
            self.embeddings = np.load(embedding_path, mmap_mode="r").astype(dtype=np.int64)
    
    def get_instsegpaths(self):
        instseg_paths = natsorted(glob.glob(f"{self.input_folder}/masks/mask_*.png"))
        return instseg_paths
    
    def _load_instseg(self, instseg_path):
        instseg = np.asarray(imageio.imread(instseg_path), dtype=np.int64)
        instseg = np.dot(instseg, [0.2989, 0.5870, 0.1140])
        instseg = np.round(instseg).astype(np.uint8)
        return np.round(instseg).astype(np.uint8)

    def read_embedding_from_file(self, idx):
        return self.embeddings[idx]
    
    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgbs/rgb_*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depths/depth_*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        return [torch.from_numpy(p) for p in self.anno['extrinsics']]