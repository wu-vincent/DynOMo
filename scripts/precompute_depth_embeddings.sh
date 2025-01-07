#!/bin/bash
export CODE_DIR=$PWD

# DEPTH ANYTHING SETUP
# depth anything setup
cd Depth-Anything/metric_depth && makdir checkpoints && cd checkpoints 
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt
cd ../../../

# DEPTH ANYTHING V2 SETUP
# depth anything setup
cd Depth-Anything/metric_depth && makdir checkpoints && cd checkpoints 
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth
cd ../../../

# DAVIS DATA
# depth prediction DA
python preprocess/get_depth_anything_prediction.py -m zoedepth --pretrained_resource="local::$CODE_DIR/Depth-Anything/metric_depth/checkpoints/depth_anything_metric_depth_outdoor.pt" --base_path $CODE_DIR/data/DAVIS/JPEGImages/480p/ --save_dir $CODE_DIR/data/DAVIS/Depth/480p/
# depth prediction DA-V2
python preprocess/get_depth_anything_V2_prediction.py --base_path $CODE_DIR/data/DAVIS/JPEGImages/480p/ --save_dir $CODE_DIR/data/DAVIS/Depth_V2/480p/
# dino featurees
python preprocess/get_dino_prediction.py configs/davis/dynomo_davis.py --base_path $CODE_DIR/data/DAVIS/JPEGImages/480p/ --save_dir $CODE_DIR/data/DAVIS/Feats/480p/

# PANOPTIC SPORT
# depth prediction
python preprocess/get_depth_anything_prediction.py -m zoedepth --pretrained_resource="local::$CODE_DIR/Depth-Anything/metric_depth/checkpoints/depth_anything_metric_depth_indoor.pt" --base_path $CODE_DIR/data/panoptic_sport/ --save_dir $CODE_DIR/data/panoptic_sport/
# depth prediction DA-V2
python preprocess/get_depth_anything_V2_prediction.py --base_path $CODE_DIR/data/panoptic_sport/ --save_dir $CODE_DIR/data/panoptic_sport/
# dino featurees
python preprocess/get_dino_prediction.py configs/panoptic_sports/dynomo_panoptic_sports.py --base_path $CODE_DIR/data/panoptic_sport/  --save_dir $CODE_DIR/data/panoptic_sport/

# IPHONE DATASET
# dino featurees
python preprocess/get_dino_prediction.py configs/iphone/dynomo_iphone.py --base_path $CODE_DIR/data/iphone/  --save_dir $CODE_DIR/data/iphone/


