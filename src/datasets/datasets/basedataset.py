"""
PyTorch dataset classes for GradSLAM v1.0.

The base dataset class now loads one sequence at a time
(opposed to v0.1.0 which loads multiple sequences).

A few parts of this code are adapted from NICE-SLAM
https://github.com/cvg/nice-slam/blob/645b53af3dc95b4b348de70e759943f7228a61ca/src/utils/datasets.py
"""
from typing import Optional, Union  

import cv2
import imageio
import numpy as np
import torch

from .geometryutils import relative_transformation
from . import datautils
import torchvision
from torchvision.transforms.functional import InterpolationMode
from src.utils.camera_helpers import as_intrinsics_matrix
import sys
import torchvision.transforms as T
from preprocess.get_dino_prediction import generate_im_feats
import os
from sklearn.decomposition import PCA
from preprocess.get_depth_anything_prediction import infer
from PIL import Image
from torchvision.transforms.functional import to_tensor


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


def get_depth_model(model, name, device):
    if "DepthAnythingV2" in model:
        sys.path.append("Depth-Anything-V2/metric_depth")
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'DepthAnythingV2-vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'DepthAnythingV2-vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'DepthAnythingV2-vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        if name == 'davis':
            dataset = 'vkitti'
            max_depth = 80
            url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth"
        else:
            dataset = 'hypersim'
            max_depth = 20
            url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth"
        encoder = model.split('-')[-1]

        depth_model = DepthAnythingV2(**{**model_configs[model], 'max_depth': max_depth})
        model_name = f'depth_anything_v2_metric_{dataset}_{encoder}.pth'
        model_dir = f'Depth-Anything-V2/metric_depth/checkpoints'
        if not os.path.isfile(f"{model_dir}/{model_name}"):
            import subprocess
            subprocess.run(
                f"wget {url}; mkdir {model_dir}; mv {model_name} {model_dir}/",
                shell=True
            )
        depth_model.load_state_dict(torch.load(f"{model_dir}/{model_name}", map_location='cpu'))
        depth_transform = None

    elif model == "DepthAnything":
        sys.path.append("Depth-Anything/metric_depth")
        from zoedepth.models.builder import build_model
        from zoedepth.utils.config import get_config
        from zoedepth.data.diml_outdoor_test import ToTensor

        model_dir = "Depth-Anything/metric_depth/checkpoints"
        if name == 'davis':
            dataset = 'vkitti2'
            model_name = "depth_anything_metric_depth_outdoor.pt"
            pretrained_resource = f"local::{os.getcwd()}/{model_dir}/{model_name}"
            url = f"https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/{model_name}"
        else:
            dataset = 'hypersim_test'
            model_name = "depth_anything_metric_depth_indoor.pt"
            pretrained_resource = f"local::{os.getcwd()}/{model_dir}/{model_name}"
            url = f"https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/{model_name}"
        
        vit_name = "depth_anything_vitl14.pth"
        vit_url = f"https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/{vit_name}"

        if not os.path.isfile(pretrained_resource.replace("local::", '')):
            import subprocess
            subprocess.run(
                f"wget {url}; wget {vit_url}; mkdir {model_dir}; mv {model_name} {model_dir}/; mv {vit_name} {model_dir}",
                shell=True
            )

        overwrite = {"pretrained_resource": pretrained_resource}
        config = get_config("zoedepth", "eval", dataset, **overwrite)
        depth_model = build_model(config)
        depth_transform = ToTensor()

    depth_model = depth_model.eval().to(device)
    return depth_model, depth_transform

def get_emb_model(desired_img_height, desired_img_width, model='dinov2_vits14_reg', embedding_dim=384, model_input_size=896, num_crops_l0=4, crop_n_layers=1, device='cuda:0'):
    emb_model = torch.hub.load('facebookresearch/dinov2', model).to(device)
    emb_transform = T.Compose([
        T.Resize(model_input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(model_input_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    emb_kwargs = {
        'model_input_size': model_input_size,
        'num_crops_l0': num_crops_l0,
        'crop_n_layers': crop_n_layers,
        'embedding_dim': embedding_dim,
        'device': device}
    emb_model.eval()
    output_size = (desired_img_height, desired_img_width)
    emb_initial_scale = torchvision.transforms.Resize(
        output_size, InterpolationMode.BILINEAR)
    return emb_model, emb_transform, emb_kwargs, emb_initial_scale


class GradSLAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_dict,
        every_x_frame: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: float = 1.0,
        desired_width: float = 1.0,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dim: int = 512,
        relative_pose: bool = False,  # If True, the pose is relative to the first frame
        per_seq_intrinsics: bool = False,
        online_depth=False,
        online_emb=False,
        **kwargs,
    ):
        super().__init__()
        # intialialize variables
        self.name = config_dict["dataset_name"]
        self.device = device
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]
        self.dtype = dtype
        self.load_embeddings = load_embeddings
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose
        self.channels_first = channels_first
        self.normalize_color = normalize_color
        self.online_depth = online_depth
        self.online_emb = online_emb

        # initialize start and end
        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(every_x_frame))
        if not (end == -1 or end > start):
            raise ValueError("end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start))

        # initialize camera distotion and image crops
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

        # get file paths
        self.color_paths, self.depth_paths, self.embedding_paths, self.bg_paths, self.instseg_paths = self.get_filepaths()
        self.num_imgs = len(self.color_paths)
        self.poses = self.load_poses()
        
        # get image width and image height and downscale factors
        img = imageio.imread(self.color_paths[0])
        h, w, _ = img.shape
        config_dict["camera_params"]["image_height"] = h
        config_dict["camera_params"]["image_width"] = w
        self.height_downsample_ratio = desired_height
        self.width_downsample_ratio = desired_width
        self.desired_height = int(h * desired_height)
        self.desired_width = int(w * desired_width)
        
        # set scaling functions
        self.trans_nearest = torchvision.transforms.Resize(
                (self.desired_height, self.desired_width), InterpolationMode.NEAREST)
        self.trans_bilinear = torchvision.transforms.Resize(
                (self.desired_height, self.desired_width), InterpolationMode.BILINEAR)
        
        # if nnot per sequence intrinsics take intrinsics from dataset file
        self.per_seq_intrinsics = per_seq_intrinsics
        if not per_seq_intrinsics:
            self.orig_height = config_dict["camera_params"]["image_height"]
            self.orig_width = config_dict["camera_params"]["image_width"]
            self.fx = config_dict["camera_params"]["fx"]
            self.fy = config_dict["camera_params"]["fy"]
            self.cx = config_dict["camera_params"]["cx"]
            self.cy = config_dict["camera_params"]["cy"]
        else:
            self.orig_height = config_dict["camera_params"]["image_height"]
            self.orig_width = config_dict["camera_params"]["image_width"]

        # check of number of paths are the same over different inputs
        if self.online_depth is None and len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must be the same.")
        if self.online_emb is None and self.load_embeddings:
            if len(self.color_paths) != len(self.embedding_paths):
                raise ValueError("Mismatch between number of color images and number of embedding files.")
        if len(self.color_paths) != len(self.poses):
            raise ValueError(f"Number of color images and poses must be the same, but got {len(self.color_paths)} and {len(self.poses)}.")

        # apply stride to paths and poses
        if self.end == -1:
            self.end = self.num_imgs
        self.color_paths = self.color_paths[self.start : self.end : every_x_frame]
        if self.online_depth is None:
            self.depth_paths = self.depth_paths[self.start : self.end : every_x_frame]
        if self.load_embeddings and self.online_emb is None:
            self.embedding_paths = self.embedding_paths[self.start : self.end : every_x_frame]
        if self.bg_paths is not None:
            self.bg_paths = self.bg_paths[self.start : self.end : every_x_frame]
        self.poses = self.poses[self.start : self.end : every_x_frame]
        
        # Tensor of retained indices (indices of frames and poses that were retained)
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : every_x_frame]

        # Update self.num_images after subsampling the dataset
        self.num_imgs = len(self.color_paths)

        # transform poses into relative poses
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

        if self.online_depth is not None:
            self.depth_model, self.depth_transform = get_depth_model(
                self.online_depth,
                self.name,
                device=self.device)
            self.depth_model = self.depth_model.to(self.device)
        if self.online_emb is not None:
            self.emb_model, self.emb_transform, self.emb_kwargs, self.emb_initial_scale = get_emb_model(
                self.desired_height,
                self.desired_width,
                self.online_emb,
                device=self.device)
            self.pca = None

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

    def _get_online_depth(self, color_path):
        if self.online_depth == "DepthAnything":
            image = np.asarray(Image.open(color_path), dtype=np.float32)
            image = image / 255.0
            if image.shape[2] > 3:
                image = image[:, :, :3]
            depth = np.zeros((image.shape[0], image.shape[1], 1))
            sample = dict(image=image, depth=depth)
            sample = self.depth_transform(sample)
            image, depth = sample['image'], sample['depth']
            image, depth = image.to(self.device), depth.to(self.device)
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            image = image.unsqueeze(0)
            focal = sample.get('focal', torch.Tensor(
                [715.0873]).to(self.device))  # This magic number (focal) is only used for evaluating BTS model

            # get depth
            pred = infer(self.depth_model, image.float(), dataset=sample['dataset'][0], focal=focal).squeeze().cpu().numpy()
        else:
            image = cv2.imread(color_path)
            pred = self.depth_model.infer_image(image)

        return pred

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
    
    def _get_online_emb(self, image):
        image = self.emb_initial_scale(image).permute(1, 2, 0).numpy()
        features = generate_im_feats(
            image,
            self.emb_model,
            self.emb_transform,
            output_size=(self.desired_height, self.desired_width),
            **self.emb_kwargs)
        features = features.permute(0, 2, 3, 1)
        features = features.cpu().squeeze().numpy()

        if features.shape[-1] != self.embedding_dim:
            shape = features.shape
            features = features.reshape(-1, shape[2])
            if self.pca is None:
                self.pca = PCA(n_components=self.embedding_dim)
                self.pca.fit(features)
            features = self.pca.transform(features)
            features = features.reshape(shape[0], shape[1], self.embedding_dim)

        return torch.from_numpy(features)
    
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
        # load image 
        color_path = self.color_paths[index]
        if isinstance(color_path, np.ndarray):
            color = color_path
        else:
            color = np.asarray(imageio.imread(color_path), dtype=float)
        if color.shape[2] > 3:
            color = color[:, :, :3]

        # load depth
        if self.online_depth is None:
            depth_path = self.depth_paths[index]
            depth = self.load_depth(depth_path, index)
        else:
            depth = self._get_online_depth(color_path)

        if len(depth.shape) > 2 and depth.shape[2] != 1:
            depth = depth[:, :, 1]

        # load embedding
        if self.load_embeddings:
            if self.online_emb is None:
                embedding = self.read_embedding_from_file(self.embedding_paths[index])
            else:
                # embedding = self._get_online_emb(color)
                embedding = self._get_online_emb(to_tensor(Image.open(color_path))[:3])
        else:
            embedding = None

        # preprocess RGB, depth, and embeddings
        color = self._preprocess_color(color)
        if self.distortion is not None:
            color = cv2.undistort(color, K, self.distortion)
        color = torch.from_numpy(color)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)
        
        if self.load_embeddings:
            embedding = self.trans_bilinear(embedding.permute(2, 0, 1)).to(self.device)

        # get intrinsics
        if not self.per_seq_intrinsics:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        else:
            K = self.intrinsics[index]
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        # get camera poses 
        pose = self.transformed_poses[index]

        # get BG
        bg = self._load_bg(self.bg_paths[index])
        bg = self.trans_nearest(torch.from_numpy(bg).unsqueeze(0)).to(self.device)
        
        instseg = self._load_instseg(self.instseg_paths[index])
        instseg = self.trans_nearest(torch.from_numpy(instseg).unsqueeze(0)).to(self.device)
        
        return [
                color.to(self.device).type(self.dtype).permute(2, 0, 1) / 255,
                depth.to(self.device).type(self.dtype).permute(2, 0, 1),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding, 
                bg,
                instseg]
