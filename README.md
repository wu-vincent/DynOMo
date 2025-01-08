<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM</h1>
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
  <h3 align="center"><a href="https://arxiv.org/abs/2409.02104">Paper</a> | <a href="">Video</a> | <a href="">Project Page</a></h3>
  <div align="center"></div>
</p>

<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
    <li>
      <a href="#downloads">Downloads</a>
    </li>
    <li>
      <a href="#benchmarking">Benchmarking</a>
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

### Main Repository
We provide a conda environment file to create our environment.

```
git clone --recurse-submodules https://github.com/JennySeidenschwarz/DynoSplatTAM.git
conda env create -f environment.yml
conda activate dynomo
pip install imageio-ffmpeg
cd diff-gaussian-rasterization-w-depth-vis-weights
python setup.py install 
pip install . 
cd ../
```

## Download Data

```bash
# Download and prepare davis dataset
python scripts/prepare_data.py davis --download --embeddings --depth --depth_model DepthAnything

# Download and prepare panoptic sport dataset
python scripts/prepare_data.py panoptic_sport --download --embeddings

# Download and prepare iphone dataset
python scripts/prepare_data.py iphone --download --embeddings
```

## RUN DYNOMO PARALLEL
```bash
python scripts/run_dynomo.py configs/davis/dynomo_davis.py --gpus 0 1 2 3 4 5 6 7
``` 

## Acknowledgement

We thank the authors of the following repositories for their open-source code:

- 3D Gaussians
  - [Dynamic 3D Gaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians)
  - [3D Gaussian Splating](https://github.com/graphdeco-inria/gaussian-splatting)
- Dataloaders
  - [GradSLAM & ConceptFusion](https://github.com/gradslam/gradslam/tree/conceptfusion)
- Baselines
  - [Nice-SLAM](https://github.com/cvg/nice-slam)
  - [Point-SLAM](https://github.com/eriksandstroem/Point-SLAM)

## Citation

If you find our paper and code useful, please cite us:

```bib
@article{keetha2023splatam,
    author    = {Keetha, Nikhil and Karhade, Jay and Jatavallabhula, Krishna Murthy and Yang, Gengshan and Scherer, Sebastian and Ramanan, Deva and Luiten, Jonathan}
    title     = {SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM},
    journal   = {arXiv},
    year      = {2023},
}
```

## Developers
- [Nik-V9](https://github.com/Nik-V9) ([Nikhil Keetha](https://nik-v9.github.io/))
- [JayKarhade](https://github.com/JayKarhade) ([Jay Karhade](https://jaykarhade.github.io/))
- [JonathonLuiten](https://github.com/JonathonLuiten) ([Jonathan Luiten](https://www.vision.rwth-aachen.de/person/216/))
- [krrish94](https://github.com/krrish94) ([Krishna Murthy Jatavallabhula](https://krrish94.github.io/))
- [gengshan-y](https://github.com/gengshan-y) ([Gengshan Yang](https://gengshan-y.github.io/))
