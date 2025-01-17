<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">DynOMo: Online Point Tracking by Dynamic Online Monocular Gaussian Reconstruction</h1>
  <p align="center">
    <a href="https://jennyseidenschwarz.github.io/"><strong>Jenny Seidenschwarz</strong></a>
    ·
    <a href="https://research.nvidia.com/labs/dvl/author/qunjie-zhou/"><strong>Qunjie Zhou</strong></a>
    ·
    <a href="https://www.bart-ai.com/"><strong>Bardenius Duisterhof</strong></a>
    ·
    <a href="https://www.cs.cmu.edu/~deva/"><strong>Deva Ramanan</strong></a>
    ·
    <a href="https://research.nvidia.com/labs/dvl/author/laura-leal-taixe/"><strong>Laura Leal-Taixe</strong></a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2409.02104">Paper</a> | <a href="https://github.com/JennySeidenschwarz/DynOMo.github.io">Project Page</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <video class="lazy" autoplay controls muted loop playsinline poster="assets/front.png">
    <source data-src="assets/combo_tracks.mp4" type="video/mp4"></video>
</p>

[![Sehen Sie sich das Video an](assets/front.png)](assets/combo_tracks.mp)

<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#downloads">Downloads</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#developers">Developers</a>
    </li>
  </ol>
</details>

## Installation

We provide a conda environment file to create our environment. Please run the following to install all necessary dependencies including the rasterizer.

```bash
# create conda environment and install rasterizer
bash scripts/create_env.sh
```

## Downloads
We provide a script to download and preprocess TAP-VID Davis, Panoptic Sport as well as the iPhone dataset. Additionally, you can pre-compute DepthAnything depth maps as well as the DINO embeddings. However, we also provide the option to predict the depth maps and embeddings during optimization. Pleas use our script as follows:

```bash
# base command
python scripts/prepare_data.py <DATASET> --download --embeddings --embedding_model <EMBEDDING_MODEL> --depth --depth_model <DEPTH_MODEL>
```
where the flags mean the following
- ```<DATASET>```: choose either davis, panoptic_sport, or iphone
- download: will lead to downloading the data
- embeddings: will lead to pre-computing the embeddings
- embedding_model ```<EMBEDDING_MODEL>```: determined the DINO version, i.e., either dinov2_vits14_reg or dinov2_vits14
- depth: will lead to pre-computing depth predictions
- depth_model ```<DEPTH_MODEL>```: will determine the depth model, i.e., either DepthAnything or DepthAnythingV2-vitl

To preprocess the data the same way as we did, please run the following:

```bash
# Download and prepare davis dataset
python scripts/prepare_data.py davis --download --embeddings --embedding_model dinov2_vits14_reg --depth --depth_model DepthAnything

# Download and prepare panoptic sport dataset
python scripts/prepare_data.py panoptic_sport --download --embeddings --embedding_model dinov2_vits14_reg

# Download and prepare iphone dataset
python scripts/prepare_data.py iphone --download --embeddings  --embedding_model dinov2_vits14_reg
```

Pleas note, since we use depth predictions generated with [Dynamic 3D Gaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians) for Panoptic Sport as well as the depth predictions from [Shape of Motion](https://github.com/vye16/shape-of-motion) for the results in our paper we do not pre-compute depth maps here for both.

## Usage
To run DynOMo, please run the ```run_dynomo.py``` script as follows:

```bash
# base command
python scripts/run_dynomo.py <CONFIG_FILE> --gpus <GPUS_TO_USE> 
``` 
where the flags are defined as follows:
- \<CONFIG_FILE>: one of the config files for the specific datasets, i.e., either ```config/davis/dynomo_davis.py```, ```config/iphone/dynomo_iphone.py```, or ```config/panoptic_sports/dynomo_panoptic_sports.py```
- gpus: the GPUs to use for the optimization as comma seperated list, e.g., 0,1,2,3,4,5,6,7

Additionally, to predict depth and embeddings online, add the following flags:
```bash
# base command with online depth and embedding computation
python scripts/run_dynomo.py <CONFIG_FILE> --gpus <GPUS_TO_USE> --online_depth <DEPTH_MODEL> --online_emb <EMBEDDING_MODEL>
``` 
where ```<DEPTH_MODEL>``` and  ```<EMBEDDING_MODEL>``` are defined as:
- online_depth ```<DEPTH_MODEL>```: will determine the depth model, i.e., either DepthAnything or DepthAnythingV2-vitl
- online_emb ```<EMBEDDING_MODEL>```: determined the DINO version, i.e., either dinov2_vits14_reg or dinov2_vits14

Finally, for evaluation of an already optimized model please add the ```just_eval``` flag:
```bash
# base command with online depth and embedding computation
python scripts/run_dynomo.py <CONFIG_FILE> --gpus <GPUS_TO_USE> --just_eval
``` 
this will re-evaluate the trajectories and store visualizations of the tracked trajectories, a grid of tracked points from the foreground mask, as well as the online rendered training views to ```experiments/<DATASET>/<RUN_NAME>/<SEQUENCE>/eval```.

Additionally, you can set the following flags for evaluation:
- not_eval_renderings: will lead to not re-rendering the training views
- not_eval_trajs: will lead to not evaluating the trajectories
- not_vis_trajs: will lead to not visualize the tracked trajectories
- not_vis_grid: will lead to not visualize a grid of tracked point trajectories
- vis_bg_and_fg: will sample points from the foreground and background during grid visualization
- vis_gt: will visualize all gt data, i.e., depth, embeddings, background mask, and rgb
- vis_rendered: will visualize all rendered data, i.e., depth, embeddings, background mask, and rgb
- novel_view_mode: will render views from a novel view, choose from zoom_out and circle
- best_x ```<x>```: will compute oracle results bei choosing the Gaussian from the best ```<x>``` Gaussians that fits the ground truth trajectory best
- traj_len ```<l>```: will change the length of the visualized trajectories


## Acknowledgement

We thank the authors of the following repositories for their open-source code:

  - [Dynamic 3D Gaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians)
  - [SplaTAM](https://github.com/spla-tam/SplaTAM)
  - [Shape of Motion](https://github.com/vye16/shape-of-motion/)
   - [3D Gaussian Splating](https://github.com/graphdeco-inria/gaussian-splatting)


## Citation

If you find our paper and code useful, please cite us:

```bib
@article{seidenschwarz2025dynomo,
  author       = {Jenny Seidenschwarz and Qunjie Zhou and Bardienus Pieter Duisterhof and Deva Ramanan and Laura Leal{-}Taix{\'{e}}},
  title        = {DynOMo: Online Point Tracking by Dynamic Online Monocular Gaussian Reconstruction},
  journal      = {3DV},
  year         = {2025},
}
```

