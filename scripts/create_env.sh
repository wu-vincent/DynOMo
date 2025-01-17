#!/bin/bash

git clone --recurse-submodules https://github.com/JennySeidenschwarz/DynoSplatTAM.git
conda env create -f environment.yml
conda activate dynomo
pip install imageio-ffmpeg
cd diff-gaussian-rasterization-w-depth-vis-weights
python setup.py install 
pip install . 
cd ../
cp preprocess/Depth-Anything_Updates/depth_anything.py Depth-Anything/metric_depth/zoedepth/models/base_models/depth_anything.py
cp preprocess/Depth-Anything_Updates/dpt.py Depth-Anything/metric_depth/zoedepth/models/base_models/dpt_dinov2/dpt.py