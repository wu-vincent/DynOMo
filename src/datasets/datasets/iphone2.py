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
import json
from itertools import product
from .col_map_utils import get_colmap_camera_params
import roma


class IphoneDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        depth_type='lidar', #'depth_anything', # 'lidar',
        cam_type='refined',
        factor=2,
        do_scale=True,
        every_x_frame: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -0,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dim: Optional[int] = 512,
        relative_pose=False,
        **kwargs,
    ):  
        self.start = start
        self.end = end

        self.factor = factor
        self.do_scale = do_scale
        print(f"Doing scale {self.do_scale} and using factor {self.factor}")
        self.input_folder = os.path.join(basedir, sequence)
        self.depth_type = depth_type
        self.cam_type = cam_type
        if do_scale:
            self.scene_norm_dict = torch.load(
                os.path.join(self.input_folder, "flow3d_preprocessed/cache/scene_norm_dict.pth"))
        
            self.scale = self.scene_norm_dict["scale"]
            self.transfm = self.scene_norm_dict["transfm"]

        print(f"Using {depth_type} depth...")
        self.depths = None

        print(f"Using relative pose {relative_pose}!")
        super().__init__(config_dict,
            every_x_frame=every_x_frame,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dim=embedding_dim,
            per_seq_intrinsics=True,
            relative_pose=relative_pose,
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
        bg = (imageio.imread(bg_path) / 255).astype(bool)
        bg = ~bg
        return bg.astype(float)
    
    def _load_instseg(self, instseg_path):
        instseg = (imageio.imread(instseg_path) / 255).astype(bool)
        return instseg.astype(float)
    
    def get_filepaths(self, cam=0):
        color_paths = natsorted(glob.glob(os.path.join(self.input_folder, f'rgb/{self.factor}x', f'{cam}_*.png')))[self.start:self.end]
        if self.depth_type == 'lidar':
            depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/{self.factor}x/{cam}_*.npy"))[self.start:self.end]
        elif 'aligned' in self.depth_type:
            depth_paths = natsorted(glob.glob(f"{self.input_folder}/flow3d_preprocessed/{self.depth_type}/1x/{cam}_*.npy"))[self.start:self.end]
        else:
            depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth_anything_jenny/depth{cam}_*.npy"))[self.start:self.end]
        bg_paths = natsorted(glob.glob(os.path.join(self.input_folder, 'flow3d_preprocessed/track_anything/1x', f'{cam}_*.png')))[self.start:self.end]
        instseg_paths = natsorted(glob.glob(os.path.join(self.input_folder, 'flow3d_preprocessed/track_anything/1x', f'{cam}_*.png')))[self.start:self.end]
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(os.path.join(self.input_folder, 'feats/2x', f"{cam}_*dino_img_quat_4_1_32_240_180.npy")))[self.start:self.end]
            features = np.load(embedding_paths[0])
            if self.embedding_dim != features.shape[2]:
                pca = PCA(n_components=self.embedding_dim)
                self.embedding_downscale = pca.fit(features.reshape(-1, features.shape[2]))
            else:
                print('Features already have the right size...')
                self.embedding_downscale = None

        return color_paths, depth_paths, embedding_paths, bg_paths, instseg_paths

    def load_depth(self, depth_path, index, use_median=True, fill_remaining=True):
        if self.depths is None and (self.depth_type == 'lidar' or 'aligned' in self.depth_type):
            self.depths = torch.from_numpy(
                    np.array(
                        [
                            np.load(depth_path).squeeze()
                            for depth_path in self.depth_paths
                        ],
                        np.float32,
                    )
                )

            if 'aligned' in self.depth_type:
                self.depths[self.depths < 1e-3] = 1e-3
                self.depths = 1.0 / self.depths

            # max depth falue
            max_depth_values_per_frame = self.depths.reshape(
                    len(self.depths), -1
                ).max(1)[0]
            max_depth_value = max_depth_values_per_frame.median() * 2.5
            self.depths = torch.clamp(self.depths, 0, max_depth_value)

            # median filter to fill "voids"
            if use_median:
                torch.use_deterministic_algorithms(False)
                for i in range(len(self.depth_paths)):
                    depth = masked_median_blur(
                        self.depths[[i]].unsqueeze(1).to("cuda"),
                        (self.depths[[i]] > 0).unsqueeze(1).to("cuda").float())[0, 0].cpu()
                    self.depths[i] = depth

                    # fill remaining zeros with surounding
                    if fill_remaining and not 'aligned' in self.depth_type:
                        self.depths[i] = torch.from_numpy(fill_remaining_zeros(self.depths[i].numpy()))
                
                torch.use_deterministic_algorithms(True)
            
            if self.do_scale:
                self.depths /= self.scale

        if self.depth_type == 'lidar' or 'aligned' in self.depth_type:
            depth = self.depths[index].numpy()
        else:
            depth = np.load(depth_path)[..., 0]
            if self.do_scale:
                depth = depth / self.scale
        
        return depth

    def load_poses(self, cam=0):
        if self.cam_type == 'original':
            pose_paths = natsorted(glob.glob(os.path.join(self.input_folder, 'camera', f'{cam}_*.json')))[self.start:self.end]
            w2cs, Ks = list(), list()
            for pose_path in pose_paths:
                with open(pose_path, 'r') as jf:
                    camera_dict = json.load(jf)
                focal_length = camera_dict["focal_length"]
                principal_point = camera_dict["principal_point"]
                Ks.append(
                    [
                        [focal_length, 0.0, principal_point[0]],
                        [0.0, focal_length, principal_point[1]],
                        [0.0, 0.0, 1.0],
                    ]
                )
                orientation = np.array(camera_dict["orientation"])
                position = np.array(camera_dict["position"])
                w2cs.append(
                    np.block(
                        [
                            [orientation, -orientation @ position[:, None]],
                            [np.zeros((1, 3)), np.ones((1, 1))],
                        ]
                    ).astype(np.float32)
                )
            self.intrinsics = torch.tensor(Ks)
            self.intrinsics[:, :2] /= self.factor
            self.w2cs = torch.from_numpy(np.array(w2cs))
        else:
            Ks, w2cs = get_colmap_camera_params(
                os.path.join(self.input_folder, "flow3d_preprocessed/colmap/sparse/"),
                [os.path.basename(p) for p in self.color_paths],
            )
            self.intrinsics = torch.from_numpy(Ks[:, :3, :3].astype(np.float32))
            self.intrinsics[:, :2] /= self.factor
            self.w2cs = torch.from_numpy(w2cs.astype(np.float32))

        if self.do_scale:
            self.w2cs = self.w2cs @ torch.linalg.inv(self.transfm)
            self.w2cs[:, :3, 3] /= self.scale

        self.w2cs = self.w2cs @ torch.linalg.inv(self.w2cs[0])
        poses = [torch.linalg.inv(w2c) for w2c in self.w2cs]
        self.first_time_w2c = self.w2cs[0]

        return poses


def fill_remaining_zeros(depth):
    depth = depth.copy()  # Copy the tensor to avoid modifying the original
    rows, cols = depth.shape
    zero_positions = np.argwhere(depth == 0)
    
    for pos in zero_positions:
        r, c = pos
        surrounding_values = []
        
        # Check all eight neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and depth[nr, nc] != 0:
                    surrounding_values.append(depth[nr, nc])
        
        if surrounding_values:
            depth[r, c] = np.mean(surrounding_values)
    
    return depth


def get_binary_kernel2d(
    window_size: tuple[int, int] | int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    from kornia
    Create a binary kernel to extract the patches.
    If the window size is HxW will create a (H*W)x1xHxW kernel.
    """
    ky, kx = _unpack_2d_ks(window_size)

    window_range = kx * ky

    kernel = torch.zeros((window_range, window_range), device=device, dtype=dtype)
    idx = torch.arange(window_range, device=device)
    kernel[idx, idx] += 1.0
    return kernel.view(window_range, 1, ky, kx)


def _unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        assert len(kernel_size) == 2, "2D Kernel size should have a length of 2."
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)

    return (ky, kx)


def _compute_zero_padding(kernel_size):
    r"""Utility function that computes zero padding tuple."""
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]


def masked_median_blur(image, mask, kernel_size=11):
    """
    Args:
        image: [B, C, H, W]
        mask: [B, C, H, W]
        kernel_size: int
    """
    assert image.shape == mask.shape
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {image.shape}")

    padding = _compute_zero_padding((kernel_size, kernel_size))

    # prepare kernel
    kernel: torch.Tensor = get_binary_kernel2d((kernel_size, kernel_size)).to(image)
    b, c, h, w = image.shape

    # map the local window to single vector
    features: torch.Tensor = torch.nn.functional.conv2d(
        image.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1
    )
    masks: torch.Tensor = torch.nn.functional.conv2d(
        mask.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1
    )
    features = features.view(b, c, -1, h, w).permute(
        0, 1, 3, 4, 2
    )  # BxCxxHxWx(K_h * K_w)
    min_value, max_value = features.min(), features.max()
    masks = masks.view(b, c, -1, h, w).permute(0, 1, 3, 4, 2)  # BxCxHxWx(K_h * K_w)
    index_invalid = (1 - masks).nonzero(as_tuple=True)
    index_b, index_c, index_h, index_w, index_k = index_invalid
    features[(index_b[::2], index_c[::2], index_h[::2], index_w[::2], index_k[::2])] = (
        min_value
    )
    features[
        (index_b[1::2], index_c[1::2], index_h[1::2], index_w[1::2], index_k[1::2])
    ] = max_value
    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=-1)[0]

    return median

class IphoneDatasetKeyPoints():
    """Return a dataset view of the annotated keypoints."""

    def __init__(self, dataset: IphoneDataset):
        super().__init__()
        self.dataset = dataset

        # Load 2D keypoints.
        keypoint_paths = sorted(
            glob(os.path.join(self.dataset.input_folder, "keypoint/2x/train/0_*.json"))
        )
        keypoints = []
        for keypoint_path in keypoint_paths:
            with open(keypoint_path) as f:
                keypoints.append(json.load(f))
        time_ids = [
            int(os.path.basename(p).split("_")[1].split(".")[0]) for p in keypoint_paths
        ]
        # only use time ids that are in the dataset.
        start = self.dataset.start
        time_ids = [t - start for t in time_ids if t - start in self.dataset.time_ids]
        self.time_ids = torch.tensor(time_ids)
        self.time_pairs = torch.tensor(list(product(self.time_ids, repeat=2)))
        self.index_pairs = torch.tensor(
            list(product(range(len(self.time_ids)), repeat=2))
        )
        self.keypoints = torch.tensor(keypoints, dtype=torch.float32)
        self.keypoints[..., :2] *= 2.0 / self.dataset.factor

    def __len__(self):
        return len(self.time_pairs)

    def __getitem__(self, index: int):
        ts = self.time_pairs[index]
        return {
            "ts": ts,
            "w2cs": self.dataset.w2cs[ts],
            "Ks": self.dataset.Ks[ts],
            "imgs": self.dataset.imgs[ts],
            "keypoints": self.keypoints[self.index_pairs[index]],
        }
