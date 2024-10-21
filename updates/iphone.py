import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .ddad import ToTensor
import glob



class iphone(Dataset):
    def __init__(self, root_dir, resize_shape):
        # image paths are of the form <data_dir_root>/{outleft, depthmap}/*.png
        self.name = 'iphone'
        sequence = ['/'.join(p.split('/')[-3:]) for p in glob.glob(f'{root_dir}/*/rgb/*')]

        self.image_files, self.depth_files = [], []
        for seq in sequence:
            print('seq', seq)
            for file in sorted(glob.glob(os.path.join(root_dir, seq, '*.png'))):
                if 'sam' in str(file):
                    continue
                self.image_files.append(file)

        self.transform = ToTensor(resize_shape)
        self.root_dir = root_dir
        self.save_dir = os.path.dirname(self.image_files[0].replace("rgb", "depth"))

    def __getitem__(self, idx, dummy_depth=True):

        image_path = self.image_files[idx]
        image = np.asarray(Image.open(image_path), dtype=np.float32)[:, :, :3] / 255.0
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


def get_iphone_loader(data_dir_root, resize_shape, batch_size=1, **kwargs):
    dataset = iphone(data_dir_root, resize_shape)
    return DataLoader(dataset, batch_size, **kwargs)
