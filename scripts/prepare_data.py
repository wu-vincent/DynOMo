import argparse
import subprocess


def download_davis(embeddings, depth):
    # DAVIS DATA
    # download data
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

def download_panoptic(embeddings, depth):
    # PANOPTIC SPORT
    # download data
    command = "cd data && wget https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip && unzip data.zip && rm -rf data.zip && cd data; mkdir annotations && cd annotations; gdown --fuzzy https://drive.google.com/file/d/1WXePud-DuR3fN5P4ThyfKWGjnIzD-0O1/view?usp=sharing; gdown --fuzzy https://drive.google.com/file/d/1MSYSTKhMvS-Wn-ACBxNVATzHUC5WybKS/view?usp=sharing; cd ../../../; mv data/data/ data/panoptic_sport; python preprocess/process_panoptic_sport.py"
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

def dosnload_iphone(embedding, depth):
    # IPHONE DATASET
    # download data from som https://drive.google.com/drive/folders/1xJaFS_3027crk7u36cue7BseAX80abRe
    
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


def get_depth_model(model, name):
    if "DepthAnythingV2" in model:
        sys.path.append("Depth-Anything-V2/metric_depth")
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'DepthAnythingV2-vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'DepthAnythingV2-vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'DepthAnythingV2-vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        if name == 'davis':
            dataset = 'vkitti'
            max_depth = 80
            url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth"
        else:
            dataset = 'hypersim'
            max_depth = 20
            url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth"
        encoder = model.split('-')[-1]

        depth_model = DepthAnythingV2(**{**model_configs[model], 'max_depth': max_depth})
        model_name = f'depth_anything_v2_metric_{dataset}_{encoder}.pth'
        model_dir = f'Depth-Anything-V2/metric_depth/checkpoints'
        if not os.path.isfile(f"{model_dir}/{model_name}"):
            import subprocess
            print(os.getcwd())
            print(f"wget {url}; mkdir {model_dir}; mv {model_name} {model_dir}/",)
            subprocess.run(
                f"wget {url}; mkdir {model_dir}; mv {model_name} {model_dir}/",
                shell=True
            )
        depth_model.load_state_dict(torch.load(f"{model_dir}/{model_name}", map_location='cpu'))
        depth_model.eval()
        depth_transform = None

    return depth_model, depth_transform



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis", action="store_true", help="If downloading davis.")
    parser.add_argument("--panoptic_sport", action="store_true", help="If downloading panoptic_sport.")
    parser.add_argument("--iphone", action="store_true", help="If downloading iphone.")
    parser.add_argument("--embeddings", action="store_true", help="If precompute embeddings.")
    parser.add_argument("--depths", action="store_true", help="If precompute depth.")
    parser.add_argument("--depth_model", default='DepthAnything', choices=["DepthAnything", "DepthAnythingV2-vitl"], help="Which Depth Model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.davis:
        download_davis(args.embeddings, args.depths)
    if args.panoptic_sport:
        download_panoptic(args.embeddings, args.depths)
    if args.iphone:
        download_iphone(args.embeddings, args.depths)
    
