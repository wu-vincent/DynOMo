import argparse
import subprocess
import os


def download_davis(download, embeddings, depth):
    # DAVIS DATA
    # download data
    if download:
        command = "cd data; wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip && unzip tapvid_davis.zip && rm -rf tapvid_davis.zip; wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip && unzip DAVIS-2017-trainval-480p.zip && rm -rf DAVIS-2017-trainval-480p.zip && cd ..; python preprocess/process_davis.py"
        subprocess.run(
            command,
            shell=True
        )

    if embeddings:
        command = f"python preprocess/get_dino_prediction.py configs/davis/dynomo_davis.py --base_path {os.getcwd()}/data/DAVIS/JPEGImages/480p/ --save_dir {os.getcwd()}/data/DAVIS/Feats/480p/"
        subprocess.run(
            command,
            shell=True
        )
    if depth:
        command = f'python preprocess/get_depth_anything_prediction.py -m zoedepth --pretrained_resource="local::{os.getcwd()}/Depth-Anything/metric_depth/checkpoints/depth_anything_metric_depth_outdoor.pt" --base_path {os.getcwd()}/data/DAVIS/JPEGImages/480p/ --save_dir {os.getcwd()}/data/DAVIS/Depth/480p/'
        subprocess.run(
            command,
            shell=True
        )

def download_panoptic(download, embeddings, depth):
    # PANOPTIC SPORT
    # download data
    if download:
        command = "cd data && wget https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip && unzip data.zip && rm -rf data.zip && cd data; mkdir annotations && cd annotations; gdown --fuzzy https://drive.google.com/file/d/1WXePud-DuR3fN5P4ThyfKWGjnIzD-0O1/view?usp=sharing; gdown --fuzzy https://drive.google.com/file/d/1MSYSTKhMvS-Wn-ACBxNVATzHUC5WybKS/view?usp=sharing; cd ../../../; mv data/data/ data/panoptic_sport; python preprocess/process_panoptic_sport.py; cd data; wget https://vision.in.tum.de/webshare/u/seidensc/DynOMo/Dynamic3DGaussianDepth.zip; unzip Dynamic3DGaussianDepth.zip; rm Dynamic3DGaussianDepth.zip; cd ../"
        subprocess.run(
            command,
            shell=True
        )

    if embeddings:
        command = f"python preprocess/get_dino_prediction.py configs/panoptic_sports/dynomo_panoptic_sports.py --base_path {os.getcwd()}/data/panoptic_sport/  --save_dir {os.getcwd()}/data/panoptic_sport/"
        subprocess.run(
            command,
            shell=True
        )
    if depth:
        command = f'python preprocess/get_depth_anything_prediction.py -m zoedepth --pretrained_resource="local::{os.getcwd()}/Depth-Anything/metric_depth/checkpoints/depth_anything_metric_depth_indoor.pt" --base_path {os.getcwd()}/data/panoptic_sport/ --save_dir {os.getcwd()}/data/panoptic_sport/'
        subprocess.run(
            command,
            shell=True
        )

def dosnload_iphone(download, embedding, depth):
    # IPHONE DATASET
    # download data from som https://drive.google.com/drive/folders/1xJaFS_3027crk7u36cue7BseAX80abRe
    if download:
        files = [
            "gdown --fuzzy https://drive.google.com/file/d/15PirJRqsT5lLjuGdLWALBDFMQanj8FTh/view?usp=drive_link && unzip paper-windmill.zip && rm paper-windmill.zip", 
            "gdown --fuzzy https://drive.google.com/file/d/18sjQQMU6AijyXg4BoucLX82R959BYAzz/view?usp=drive_link && unzip sriracha-tree.zip && rm sriracha-tree.zip"
        ]
        
        for file in files:
            command = f"cd data && mkdir iphone && cd iphone; {file}; cd ../../"
            subprocess.run(
                command,
                shell=True
            )
    
    if embeddings:
        command = f"python preprocess/get_dino_prediction.py configs/iphone/dynomo_iphone.py --base_path {os.getcwd()}/data/iphone/  --save_dir {os.getcwd()}/data/iphone/"
        subprocess.run(
            command,
            shell=True
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["davis", "panoptic_sport", "iphone"], help="If downloading davis.")
    parser.add_argument("--download", action="store_true", help="If download needed.")
    parser.add_argument("--embeddings", action="store_true", help="If precompute embeddings.")
    parser.add_argument("--depths", action="store_true", help="If precompute depth.")
    parser.add_argument("--depth_model", default='DepthAnything', choices=["DepthAnything", "DepthAnythingV2-vitl"], help="Which Depth Model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "davis":
        download_davis(args.download, args.embeddings, args.depths)
    if args.dataset == "panoptic_sport":
        download_panoptic(args.download, args.embeddings, args.depths)
    if args.dataset == "iphone":
        download_iphone(args.download, args.embeddings, args.depths)
    
