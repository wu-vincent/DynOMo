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


jono_seqs_stereo = \
    {"boxes/ims/27": "boxes/ims/23",
     "softball/ims/27": "softball/ims/8",
     "basketball/ims/21": "basketball/ims/24",
     "football/ims/18": "football/ims/3", 
     "juggle/ims/14": "juggle/ims/23",
     "tennis/ims/8": "tennis/ims/27"}


class JonoDynoSplatamDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stereo=False,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        load_instseg=True,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 64,
        get_pc_jono=False,
        jono_depth=False,
        feats_224=False,
        do_transform=False,
        novel_view_mode=None,
        **kwargs,
    ): 
        self.stereo = stereo
        if self.stereo:
            self.input_folder = os.path.join(basedir, jono_seqs_stereo[sequence])
        else:
            self.input_folder = os.path.join(basedir, sequence)

        start = start + 2
        self.get_pc_jono = get_pc_jono
        self.jono_depth = jono_depth
        # if self.get_pc_jono or do_transform:
        #     kwargs['relative_pose'] = False
        
        self.sequence = sequence
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        self.feats_224 = feats_224
        self.do_transform = do_transform
        self.novel_view_mode = novel_view_mode

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

    def get_bg_paths(self):
        if os.path.isdir(f"{self.input_folder.replace('ims', 'seg')}"):
            bg_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'seg')}/*.png"))
        else:
            bg_paths = None
        return bg_paths
    
    def get_instsegpaths(self):
        # instseg_paths = natsorted(glob.glob(f"{self.input_folder}/*sam_big_area.npy"))
        instseg_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'seg')}/*.png"))
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
        # get color paths
        color_paths = natsorted(glob.glob(f"{self.input_folder}/*.jpg"))

        # get depth paths
        if not self.jono_depth:
            depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth*.npy"))
        else:
            input_folder = self.input_folder.replace("/scratch/jseidens/data/data", "../Dynamic3DGaussians/rendered")
            input_folder = self.input_folder.replace("/data3/jseidens/data", "../Dynamic3DGaussians/rendered")
            depth_paths = natsorted(glob.glob(f"{input_folder}/depth*.npy"))
        
        # get embedding paths
        embedding_paths = None
        if self.load_embeddings:
            print("Using DINO 4/1 features")
            if os.path.isfile(f"{self.input_folder.replace('ims', 'feats')}/000000dino_img_quat_4_1_64.npy"):
                embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'feats')}/*dino_img_quat_4_1_64.npy"))
            else:
                embedding_paths = natsorted(glob.glob(f"{self.input_folder.replace('ims', 'feats')}/*dino_img_quat_4_1.npy"))

            features = np.load(embedding_paths[0])
            if self.embedding_dim != features.shape[2]:
                pca = PCA(n_components=self.embedding_dim)
                self.embedding_downscale = pca.fit(features.reshape(-1, features.shape[2]))
            else:
                print('Features already have the right size...')
                self.embedding_downscale = None

        return color_paths, depth_paths, embedding_paths
    
    def load_support_trajs(self):
        self.support_trajs = None
            
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
