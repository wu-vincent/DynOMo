# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import argparse

import torch
from tqdm import tqdm
import glob
import sys
sys.path.append("Depth-Anything-V2/metric_depth")
from depth_anything_v2.dpt import DepthAnythingV2
import numpy as np
import os
from PIL import Image
import cv2


@torch.no_grad()
def evaluate(model, base_path, save_dir, save_depth=True, device="cuda:4"):
    print('Starting evaluation...')
    
    # make save dir and get image paths
    os.makedirs(save_dir, exist_ok=True)
    if "DAVIS" in base_path:
        test_loader = glob.glob(f"{base_path}/*/*.jpg")
    elif "panoptic_sport" in base_path:
        test_loader = glob.glob(f"{base_path}/*/ims/*/*.jpg")
    elif experiment['data']['name'] == "iphone":
        test_loader = glob.glob(f'{base_path}/*/rgb/2x/0_*.png')
        
    for i, image_path in tqdm(enumerate(test_loader), total=len(test_loader)):
        # load data and transform
        raw_img = cv2.imread(image_path)

        # get save name
        if "DAVIS" in base_path:
            save_name = image_path.replace(base_path, save_dir)[:-3] + 'npy'
        elif "panoptic_sport" in base_path:
            save_name = image_path.replace("ims", "Depth_V2")[:-3] + 'npy'
            os.makedirs(os.path.dirname(image_path.replace("ims", "Depth_V2")), exist_ok=True)
        elif "iphone" in base_path:
            save_name = image_path.replace("rgb", "DepthAnything_V2")[:-3] + 'npy'
            os.makedirs(os.path.dirname(image_path.replace("rgb", "DepthAnything_V2")), exist_ok=True)

        pred = model.infer_image(raw_img) # HxW raw depth map in numpy

        # Save image, depth, pred for visualization
        if save_depth:
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            np.save(save_name, np.expand_dims(pred.squeeze(), axis=2))


def download_weights(dataset, encoder):
    if dataset == 'vkitti':
        url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth"
    else:
        url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth"
    
    model_name = f'depth_anything_v2_metric_{dataset}_{encoder}.pth'
    model_dir = f'Depth-Anything-V2/metric_depth/checkpoints'
    if not os.path.isfile(f"{model_dir}/{model_name}"):
        import subprocess
        print(os.getcwd())
        print(f"wget {url}; mkdir {model_dir}; mv {model_name} {model_dir}/",)
        subprocess.run(
            f"wget {url}; mkdir {model_dir}; mv {model_name} {model_dir}/",
            shell=True
        )
    return model_dir, model_name


def eval_model(model, base_path, save_dir, save_depth=True, dataset='nyu', device="cuda:4"):
    # get model
    model_configs = {
        'DepthAnythingV2-vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'DepthAnythingV2-vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'DepthAnythingV2-vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    if dataset == 'vkitti':
        max_depth = 80
    else:
        max_depth = 20
    
    encoder = model.split('-')[-1]
    model = DepthAnythingV2(**{**model_configs[model], 'max_depth': max_depth})\
    
    # download weights
    model_dir, model_name = download_weights(dataset, encoder)
    model.load_state_dict(torch.load(f"{model_dir}/{model_name}", map_location='cpu'))
    model = model.to(device).eval()
    
    # Load default pretrained resource defined in config if not set
    print(f"Evaluating {model_name} on {dataset}...")
    evaluate(model, base_path, save_dir, save_depth, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=False, default="DepthAnythingV2-vitl", help="Path to image data.")
    parser.add_argument("-b", "--base_path", type=str,
                        required=True, help="Path to image data.")
    parser.add_argument("-s", "--save_dir", type=str,
                        required=True, help="Path to store the data.")
    parser.add_argument('-d', '--device', type=str, 
                        default='cuda:0', help='Which device to use.')
    parser.add_argument('--save_depth', action="store_false",
                        help='If saving depth or not.')
    args, unknown_args = parser.parse_known_args()

    if not "panoptic" in args.base_path:
        datasets = ['vkitti']
    else:
        datasets = ['hypersim']
    
    for dataset in datasets:
        eval_model(
            model=args.model,
            base_path=args.base_path,
            save_dir=args.save_dir,
            dataset=dataset,
            save_depth=args.save_depth,
            device=args.device)
