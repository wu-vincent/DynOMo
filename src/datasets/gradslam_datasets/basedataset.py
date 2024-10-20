"""
PyTorch dataset classes for GradSLAM v1.0.

The base dataset class now loads one sequence at a time
(opposed to v0.1.0 which loads multiple sequences).

A few parts of this code are adapted from NICE-SLAM
https://github.com/cvg/nice-slam/blob/645b53af3dc95b4b348de70e759943f7228a61ca/src/utils/datasets.py
"""

import abc
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union  

import cv2
import imageio
import numpy as np
import torch
import yaml
from natsort import natsorted

from .geometryutils import relative_transformation
from . import datautils
import torchvision
from torchvision.transforms.functional import InterpolationMode
from utils.camera_helpers import as_intrinsics_matrix, from_intrinsics_matrix


def to_scalar(inp: Union[np.ndarray, torch.Tensor, float]) -> Union[int, float]:
    """
    Convert the input to a scalar
    """
    if isinstance(inp, float):
        return inp

    if isinstance(inp, np.ndarray):
        assert inp.size == 1
        return inp.item()

    if isinstance(inp, torch.Tensor):
        assert inp.numel() == 1
        return inp.item()


def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if "Y" not in header["channels"] else channelData["Y"]

    return Y


class GradSLAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_dict,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = False,  # If True, the pose is relative to the first frame
        load_instseg: bool = False,
        precomp_intrinsics: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        if not precomp_intrinsics:
            self.orig_height = config_dict["camera_params"]["image_height"]
            self.orig_width = config_dict["camera_params"]["image_width"]
            self.fx = config_dict["camera_params"]["fx"]
            self.fy = config_dict["camera_params"]["fy"]
            self.cx = config_dict["camera_params"]["cx"]
            self.cy = config_dict["camera_params"]["cy"]
        else:
            self.orig_height = config_dict["camera_params"]["image_height"]
            self.orig_width = config_dict["camera_params"]["image_width"]

        self.dtype = dtype

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.load_embeddings = load_embeddings
        self.load_instseg = load_instseg
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError("end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start))

        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"])
            if "distortion" in config_dict["camera_params"]
            else None
        )
        self.crop_size = (
            config_dict["camera_params"]["crop_size"] if "crop_size" in config_dict["camera_params"] else None
        )

        self.crop_edge = None
        if "crop_edge" in config_dict["camera_params"].keys():
            self.crop_edge = config_dict["camera_params"]["crop_edge"]

        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()

        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must be the same.")
        
        if self.load_embeddings:
            if len(self.color_paths) != len(self.embedding_paths):
                raise ValueError("Mismatch between number of color images and number of embedding files.")

        self.num_imgs = len(self.color_paths)
        self.poses = self.load_poses()
        if len(self.color_paths) != len(self.poses):
            raise ValueError(f"Number of color images and poses must be the same, but got {len(self.color_paths)} and {len(self.poses)}.")
        
        self.instseg_paths = self.get_instsegpaths()
        self.bg_paths = self.get_bg_paths()
        if len(self.color_paths) != len(self.instseg_paths):
            raise ValueError("Number of color images and segmentation masks must be the same.")

        if self.end == -1:
            self.end = self.num_imgs

        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.depth_paths = self.depth_paths[self.start : self.end : stride]

        if self.load_embeddings:
            self.embedding_paths = self.embedding_paths[self.start : self.end : stride]
        if self.load_instseg:
            self.instseg_paths = self.instseg_paths[self.start : self.end : stride]
        if self.bg_paths is not None:
            self.bg_paths = self.bg_paths[self.start : self.end : stride]

        self.poses = self.poses[self.start : self.end : stride]
        
        # Tensor of retained indices (indices of frames and poses that were retained)
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]

        # Update self.num_images after subsampling the dataset
        self.num_imgs = len(self.color_paths)

        # self.transformed_poses = datautils.poses_to_transforms(self.poses)
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses
        self.precomp_intrinsics = precomp_intrinsics

    def __len__(self):
        return self.num_imgs

    def get_filepaths(self):
        """Return paths to color images, depth images. Implement in subclass."""
        raise NotImplementedError

    def load_poses(self):
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth / self.png_depth_scale

    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )

    def get_cam_K(self):
        """
        Return camera intrinsics matrix K

        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        """
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K

    def read_embedding_from_file(self, embedding_path: str):
        """
        Read embedding from file and process it. To be implemented in subclass for each dataset separately.
        """
        raise NotImplementedError
    
    def load_depth(self, depth_path, index):
        if ".png" in depth_path:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path)).squeeze() #, dtype=np.int64)
        elif 'npy' in depth_path:
            depth = np.load(depth_path, mmap_mode="r").squeeze()  # .astype(dtype=np.int64)
        elif ".exr" in depth_path:
            depth = readEXR_onlydepth(depth_path)
        return depth

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]

        if isinstance(color_path, np.ndarray):
            color = color_path
        else:
            color = np.asarray(imageio.imread(color_path), dtype=float)
        if color.shape[2] > 3:
            color = color[:, :, :3]
        color = self._preprocess_color(color)
        depth = self.load_depth(depth_path, index)
        
        if len(depth.shape) > 2 and depth.shape[2] != 1:
            depth = depth[:, :, 1]

        if not self.precomp_intrinsics:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        else:
            K = self.intrinsics[index]

        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        color = torch.from_numpy(color)
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]
        return_vals = [
                color.to(self.device).type(self.dtype).permute(2, 0, 1) / 255,
                depth.to(self.device).type(self.dtype).permute(2, 0, 1),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),]

        trans_nearest = torchvision.transforms.Resize(
                (color.shape[0], color.shape[1]), InterpolationMode.NEAREST)
        trans_bilinear = torchvision.transforms.Resize(
                (color.shape[0], color.shape[1]), InterpolationMode.BILINEAR)
        
        if self.load_instseg:
            # load and downsample to rgb size
            instseg = self._load_instseg(self.instseg_paths[index])            
            instseg = trans_nearest(torch.from_numpy(instseg).unsqueeze(0))            
            return_vals = return_vals + [instseg.to(self.device).type(self.dtype)]
        else:
            return_vals = return_vals + [None]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index]).permute(2, 0, 1)
            embedding = trans_bilinear(embedding)
            return_vals = return_vals + [embedding.to(self.device)] # Allow embedding to be another dtype
        else:
            return_vals = return_vals + [None]
        
        # rest of support trajs
        return_vals = return_vals + [None]
        
        if self.bg_paths is not None:
            bg = self._load_bg(self.bg_paths[index])
            bg = trans_nearest(torch.from_numpy(bg).unsqueeze(0)) 
            return_vals = return_vals + [bg.to(self.device)]
        else:
            return_vals = return_vals + [None]

        return return_vals
