import os
import argparse
from importlib.machinery import SourceFileLoader

import numpy as np
from plyfile import PlyData, PlyElement

# Spherical harmonic constant
C0 = 0.28209479177387814


def rgb_to_spherical_harmonic(rgb):
    return (rgb-0.5) / C0


def spherical_harmonic_to_rgb(sh):
    return sh*C0 + 0.5


def save_ply(path, means, scales, rotations, rgbs, opacities, normals=None, timestamp=1):
    if normals is None:
        normals = np.zeros_like(means)

    colors = rgb_to_spherical_harmonic(rgbs)

    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))

    attrs = ['x', 'y', 'z',
             'nx', 'ny', 'nz',
             'f_dc_0', 'f_dc_1', 'f_dc_2',
             'opacity',
             'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3',]

    dtype_full = [(attribute, 'f4') for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)

    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    print(f"Saved PLY format Splat to {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str, help="Path to experiment directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load SplaTAM config
    os.makedirs(os.path.join(os.path.dirname(args.params_path), 'splats'), exist_ok=True)

    print('Loading...')
    params = dict(np.load(args.params_path, allow_pickle=True))
    print('Loaded!!!')
    for timestamp in range(params['means3D'].shape[2]):
        if timestamp != 0 and timestamp != 100:
            continue
        means = params['means3D'][:, :, timestamp][params['timestep']<=timestamp]
        scales = params['log_scales'][:, :, timestamp][params['timestep']<=timestamp]
        rotations = params['unnorm_rotations'][:, :, timestamp][params['timestep']<=timestamp]
        rgbs = params['rgb_colors'][:, :, timestamp][params['timestep']<=timestamp]
        opacities = params['logit_opacities'][params['timestep']<=timestamp]

        ply_path = os.path.join(os.path.dirname(args.params_path), 'splats', f"splat_{timestamp}.ply")
        save_ply(ply_path, means, scales, rotations, rgbs, opacities, timestamp=timestamp)
