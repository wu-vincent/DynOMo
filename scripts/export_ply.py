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
    parser.add_argument("config", type=str, help="Path to config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load SplaTAM config
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    config = experiment.config
    work_path = config['workdir']
    run_name = "splatam_motocross-jump/splatam_motocross-jump_0_kNN_500_1000_500_-1_32_False_0.5_5_5_5_0.001_True_True_True_True_True_2000_16_0.1_True_False_False_True_True_2_1_5_1_aniso"
    params_path = os.path.join(work_path, run_name, "params.npz")
    os.makedirs(os.path.join(work_path, run_name, 'splats'), exist_ok=True)

    params = dict(np.load(params_path, allow_pickle=True))
    for timestamp in range(params['means3D'].shape[2]):
        if timestamp != 30:
            continue
        print(params['rgb_colors'].shape, params['log_scales'].shape)
        means = params['means3D'][:, :, timestamp]
        scales = params['log_scales'][:, timestamp, :]
        rotations = params['unnorm_rotations'][:, :, timestamp]
        rgbs = params['rgb_colors'][:, :, timestamp]
        opacities = params['logit_opacities']

        ply_path = os.path.join(work_path, run_name, 'splats', f"splat_{timestamp}.ply")
        save_ply(ply_path, means, scales, rotations, rgbs, opacities, timestamp=timestamp)
        quit()