import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .ddad import ToTensor
import glob


class Davis(Dataset):
    def __init__(self, root_dir, resize_shape):
        # image paths are of the form <data_dir_root>/{outleft, depthmap}/*.png
        self.name = 'DAVIS'
        sequence = ['goat'] # ['bike-packing', 'tennis']
        sequence = ['car-roundabout', 'motocross-jump', 'breakdance', 'drift-chicane', 'drift-straight', 'judo', 'soapbox', 'dogs-jump', 'parkour', 'india', 'pigs', 'cows', 'gold-fish', 'paragliding-launch', 'camel', 'blackswan', 'dog', 'bike-packing', 'shooting', 'lab-coat', 'kite-surf', 'bmx-trees', 'dance-twirl', 'car-shadow', 'libby', 'scooter-black', 'mbike-trick', 'loading', 'horsejump-high']

        self.image_files, self.depth_files = [], []
        for seq in sequence:
            print('seq', seq)
            for file in sorted(glob.glob(os.path.join(root_dir, seq, '*.jpg'))):
                if 'sam' in str(file):
                    continue
                self.image_files.append(file)
        print('In total files:', len(self.image_files))
        self.transform = ToTensor(resize_shape)
        self.root_dir = root_dir
        self.save_dir = os.path.dirname(self.image_files[0].replace("JPEGImages", "DEPTH"))

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


def get_davis_loader(data_dir_root, resize_shape, batch_size=1, **kwargs):
    dataset = Davis(data_dir_root, resize_shape)
    return DataLoader(dataset, batch_size, **kwargs)
