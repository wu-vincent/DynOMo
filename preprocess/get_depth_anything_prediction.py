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
sys.path.append("Depth-Anything/metric_depth/")
from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import get_config
from zoedepth.data.diml_outdoor_test import ToTensor
import numpy as np
import os
from PIL import Image


@torch.no_grad()
def infer(model, images, do_mean_pred=False, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    if do_mean_pred:
        pred2 = model(torch.flip(images, [3]), **kwargs)
        pred2 = get_depth_from_prediction(pred2)
        pred2 = torch.flip(pred2, [3])

        mean_pred = 0.5 * (pred1 + pred2)
        return mean_pred
    else:
        return pred1


@torch.no_grad()
def evaluate(model, base_path, save_dir, config, save_depth=True, device="cuda:4"):
    transform = ToTensor()
    model.eval()
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
        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.zeros((image.shape[0], image.shape[1], 1))
        sample = dict(image=image, depth=depth)
        sample = transform(sample)

        # get save name
        if "DAVIS" in base_path:
            save_name = image_path.replace(base_path, save_dir)[:-3] + 'npy'
        elif "panoptic_sport" in base_path:
            save_name = image_path.replace("ims", "Depth")[:-3] + 'npy'
            os.makedirs(os.path.dirname(image_path.replace("ims", "Depth")), exist_ok=True)
        elif "iphone" in base_path:
            save_name = image_path.replace("rgb", "DepthAnything")[:-3] + 'npy'
            os.makedirs(os.path.dirname(image_path.replace("rgb", "DepthAnything")), exist_ok=True)

        # put sample on device
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth = sample['image'], sample['depth']
        image, depth = image.to(device), depth.to(device)
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        image = image.unsqueeze(0)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).to(device))  # This magic number (focal) is only used for evaluating BTS model

        # get depth
        pred = infer(model, image, dataset=sample['dataset'][0], focal=focal)

        # Save image, depth, pred for visualization
        if save_depth:
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            np.save(save_name, pred.squeeze().unsqueeze(2).cpu().numpy())


def main(config, base_path, save_dir, save_depth, device):
    model = build_model(config)
    model = model.to(device)
    evaluate(model, base_path, save_dir, config, save_depth, device)


def download_weights(dataset, pretrained_resource):
    model_dir = "Depth-Anything/metric_depth/checkpoints"
    if dataset == 'vkitti2':
        model_name = "depth_anything_metric_depth_outdoor.pt"
        url = f"https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/{model_name}"
    else:
        model_name = "depth_anything_metric_depth_indoor.pt"
        url = f"https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/{model_name}"
    
    vit_name = "depth_anything_vitl14.pth"
    vit_url = f"https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/{vit_name}"

    if not os.path.isfile(pretrained_resource.replace("local::", '')):
        import subprocess
        subprocess.run(
            f"wget {url}; wget {vit_url}; mkdir {model_dir}; mv {model_name} {model_dir}/; mv {vit_name} {model_dir}",
            shell=True
        )


def eval_model(model_name, base_path, save_dir, pretrained_resource, save_depth=True, dataset='nyu', device="cuda:4", **kwargs):
    # download weights
    download_weights(dataset, pretrained_resource)
    
    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    print(f"Evaluating {model_name} on {dataset}...")
    main(config, base_path, save_dir, save_depth, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate.")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-b", "--base_path", type=str,
                        required=True, help="Path to image data.")
    parser.add_argument("-s", "--save_dir", type=str,
                        required=True, help="Path to store the data.")
    parser.add_argument('-d', '--device', type=str, 
                        default='cuda:0', help='Which device to use.')
    parser.add_argument('--save_depth', action="store_false",
                        help='If saving depth or not.')
    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    if "depth_anything_metric_depth_outdoor.pt" in args.pretrained_resource:
        datasets = ['vkitti2']
    else:
        datasets = ['hypersim_test']
    
    for dataset in datasets:
        eval_model(
            args.model,
            base_path=args.base_path,
            save_dir=args.save_dir,
            pretrained_resource=args.pretrained_resource,
            dataset=dataset,
            save_depth=args.save_depth,
            device=args.device,
            **overwrite_kwargs)
