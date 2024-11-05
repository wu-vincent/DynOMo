#!/bin/bash
export CODE_DIR=$PWD

# DEPTH ANYTHING SETUP
# depth anything setup
cd Depth-Anything/metric-depth && makdir checkpoints && cd checkpoints 
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt
cd ../../
cp scripts/get_depth_anything_prediction.py  Depth-Anything/metric_depth/

# DAVIS DATA
# download data
cd data
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip && unzip tapvid_davis.zip && rm -rf tapvid_davis.zip
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip && unzip DAVIS-2017-trainval-480p.zip && rm -rf DAVIS-2017-trainval-480p.zip && cd ..
python data/process_davis.py 
cd ../
# depth prediction
cd Depth-Anything/metric_depth/  && python get_depth_anything_prediction.py -m zoedepth --pretrained_resource="local::$CODE_DIR/Depth-Anything/metric_depth/checkpoints/depth_anything_metric_depth_indoor.pt" --base_path $CODE_DIR/data/DAVIS/JPEGImages/480p/ --save_dir $CODE_DIR/data/DAVIS/Depth/480p/ && cd ../../
# dino featurees
python scripts/get_dino_prediction.py configs/davis/dynomo_davis.py --base_path $CODE_DIR/data/DAVIS/JPEGImages/480p/ --save_dir $CODE_DIR/data/DAVIS/Feats/480p/


# PANOPTIC SPORT
# download data
cd data && wget https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip && unzip data.zip && rm -rf data.zip 
cd data && makdir annotations && cd annotations
wget https://drive.google.com/file/d/1WXePud-DuR3fN5P4ThyfKWGjnIzD-0O1/view?usp=drive_link
wget https://drive.google.com/file/d/1MSYSTKhMvS-Wn-ACBxNVATzHUC5WybKS/view?usp=drive_link
cd ../
python data/process_panoptic_sport.py
mv data/data data/panoptic_sport
# depth prediction
cd Depth-Anything/metric_depth/  && python get_depth_anything_prediction.py -m zoedepth --pretrained_resource="local::$CODE_DIR/Depth-Anything/metric_depth/checkpoints/depth_anything_metric_depth_indoor.pt" --base_path $CODE_DIR/data/panoptic_sport/ --save_dir $CODE_DIR/data/panoptic_sport/ && cd ../../
# dino featurees
python scripts/get_dino_prediction.py configs/panoptic_sports/dynomo_panoptic_sports.py --base_path $CODE_DIR/data/panoptic_sport/  --save_dir $CODE_DIR/data/panoptic_sport/


# IPHONE DATASET
# download data from som https://drive.google.com/drive/folders/1xJaFS_3027crk7u36cue7BseAX80abRe
cd data && mkdir iphone && cd iphone
gdown --fuzzy https://drive.google.com/file/d/15PirJRqsT5lLjuGdLWALBDFMQanj8FTh/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/18sjQQMU6AijyXg4BoucLX82R959BYAzz/view?usp=drive_link 
