import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .ddad import ToTensor
import glob


class Jono(Dataset):
    def __init__(self, root_dir, resize_shape):

        # image paths are of the form <data_dir_root>/{outleft, depthmap}/*.png
        
        # self.image_files = glob.glob(os.path.join(data_dir_root, '*.png'))
        # self.depth_files = [r.replace("_rgb.png", "_depth.npy")
        #                     for r in self.image_files]
        self.name = 'jono'
        sequence = ['softball/ims/27', 'juggle/ims/14', 'boxes/ims/27', 'basketball/ims/21', 'football/ims/18', 'tennis/ims/8']
        sequence = ['juggle/ims/14']
        self.image_files, self.depth_files = [], []
        for seq in sequence:
            if '27' in seq:
                continue
            for file in sorted(glob.glob(os.path.join(root_dir, seq, '*.jpg'))):
                if 'sam' in str(file):
                    continue
                self.image_files.append(file)

        self.transform = ToTensor(resize_shape)
        self.root_dir = root_dir
        self.save_dir = os.path.dirname(self.image_files[0].replace('./data/../../../DynoSplatTAM/data', '/data3/jseidens'))

    def __getitem__(self, idx, dummy_depth=True):

        image_path = self.image_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.ones((image.shape[0], image.shape[1]))

        # depth[depth > 8] = -1
        depth = depth[..., None]

        sample = dict(image=image, depth=depth)
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        return sample

    def __len__(self):
        return len(self.image_files)


def get_jono_loader(data_dir_root, resize_shape, batch_size=1, **kwargs):
    dataset = Jono(data_dir_root, resize_shape)
    return DataLoader(dataset, batch_size, **kwargs)