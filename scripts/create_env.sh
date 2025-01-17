#!/bin/bash

git clone --recurse-submodules https://github.com/JennySeidenschwarz/DynoSplatTAM.git
conda env create -f environment.yml
conda activate dynomo
pip install imageio-ffmpeg
cd diff-gaussian-rasterization-w-depth-vis-weights
python setup.py install 
pip install . 
cd ../