import argparse
import os
import shutil
import requests
import zipfile
import subprocess
import gdown
from pathlib import Path
from tqdm import tqdm


def download_file(url, save_path):
    """Download a file from a given URL with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as file, tqdm(
            desc=f"Downloading {save_path.name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path, extract_to):
    """Extract a zip file to a given directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)


def download_davis(download, embeddings, depth, embedding_model, depth_model):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    if download:
        tapvid_zip = data_dir / "tapvid_davis.zip"
        davis_zip = data_dir / "DAVIS-2017-trainval-480p.zip"

        download_file("https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip", tapvid_zip)
        extract_zip(tapvid_zip, data_dir)

        download_file("https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip", davis_zip)
        extract_zip(davis_zip, data_dir)

        subprocess.run(["python", "preprocess/process_davis.py"], check=True)

    if embeddings:
        subprocess.run([
            "python", "preprocess/get_dino_prediction.py", "configs/davis/dynomo_davis.py",
            "--base_path", str(data_dir / "DAVIS/JPEGImages/480p/"),
            "--save_dir", str(data_dir / "DAVIS/Feats/480p/"),
            "--model", embedding_model
        ], check=True)

    if depth:
        subprocess.run([
            "python", "preprocess/get_depth_anything_prediction.py", "-m", "zoedepth",
            "--pretrained_resource",
            f"local::Depth-Anything/metric_depth/checkpoints/depth_anything_metric_depth_outdoor.pt",
            "--base_path", str(data_dir / "DAVIS/JPEGImages/480p/"),
            "--save_dir", str(data_dir / "DAVIS/Depth/480p/")
        ], check=True)


def download_panoptic(download, embeddings, depth, embedding_model, depth_model):
    """Pure Python-based implementation for downloading PANOPTIC SPORT data."""
    data_dir = Path("data")
    panoptic_sport_dir = data_dir / "panoptic_sport"
    annotations_dir = panoptic_sport_dir / "annotations"
    depth_zip_url = "https://vision.in.tum.de/webshare/u/seidensc/DynOMo/Dynamic3DGaussianDepth.zip"
    depth_zip_path = data_dir / "Dynamic3DGaussianDepth.zip"

    if download:
        # Step 1: Download and extract main data
        main_data_zip_url = "https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip"
        main_data_zip_path = data_dir / "data.zip"

        data_dir.mkdir(exist_ok=True)
        download_file(main_data_zip_url, main_data_zip_path)
        extract_zip(main_data_zip_path, data_dir)

        # Step 2: Setting up annotations and downloading required files from Google Drive
        annotations_dir.mkdir(parents=True, exist_ok=True)
        gdown.download(
            "https://drive.google.com/uc?id=1WXePud-DuR3fN5P4ThyfKWGjnIzD-0O1",
            str(annotations_dir / "2dgt.json"), fuzzy=True
        )
        gdown.download(
            "https://drive.google.com/uc?id=1MSYSTKhMvS-Wn-ACBxNVATzHUC5WybKS",
            str(annotations_dir / "3dgt.json"), fuzzy=True
        )

        # Step 3: Move and process the data
        shutil.move(str(data_dir / "data"), str(panoptic_sport_dir))
        subprocess.run(["python", "preprocess/process_panoptic_sport.py"], check=True)
        subprocess.run(["python", "preprocess/convert_panoptic_sports_to_tapvid.py"], check=True)

        # Step 4: Download and extract depth data
        download_file(depth_zip_url, depth_zip_path)
        extract_zip(depth_zip_path, data_dir)

    if embeddings:
        # Precompute embeddings using the specified model
        subprocess.run([
            "python", "preprocess/get_dino_prediction.py",
            "configs/panoptic_sports/dynomo_panoptic_sports.py",
            "--base_path", str(panoptic_sport_dir),
            "--save_dir", str(panoptic_sport_dir),
            "--model", embedding_model
        ], check=True)

    if depth:
        # Precompute depth using the specified model and pretrained resource
        pretrained_resource = f"local::Depth-Anything/metric_depth/checkpoints/depth_anything_metric_depth_indoor.pt"
        subprocess.run([
            "python", "preprocess/get_depth_anything_prediction.py",
            "-m", "zoedepth",
            "--pretrained_resource", pretrained_resource,
            "--base_path", str(panoptic_sport_dir),
            "--save_dir", str(panoptic_sport_dir)
        ], check=True)


def download_iphone(download, embeddings, depth, embedding_model, depth_model):
    # IPHONE DATASET
    # download data from som https://drive.google.com/drive/folders/1xJaFS_3027crk7u36cue7BseAX80abRe
    if download:
        files = [
            "gdown --fuzzy https://drive.google.com/file/d/15PirJRqsT5lLjuGdLWALBDFMQanj8FTh/view?usp=drive_link && unzip paper-windmill.zip && rm paper-windmill.zip",
            "gdown --fuzzy https://drive.google.com/file/d/18sjQQMU6AijyXg4BoucLX82R959BYAzz/view?usp=drive_link && unzip sriracha-tree.zip && rm sriracha-tree.zip",
            "gdown --fuzzy https://drive.google.com/file/d/1QihG5A7c_bpkse5b0OBdqqFThgX0kDyZ/view?usp=drive_link && unzip bagpack.zip && bagpack.zip",
            "gdown --fuzzy https://drive.google.com/file/d/1QJQnVw_szoy_k5x9k_BAE2BWtf_BXqKn/view?usp=drive_link && unzip apple.zip && apple.zip",
            "gdown --fuzzy https://drive.google.com/file/d/1BmuxJXKi6dVaNOjmppuETQsaspAV9Wca/view?usp=drive_link && unzip haru-sit.zip && haru-sit.zip",
            "gdown --fuzzy https://drive.google.com/file/d/1frv8miU24Dl7fqblYt7zkwj129ci-68U/view?usp=drive_link && unzip handwavy.zip && handwavy.zip",
            "gdown --fuzzy https://drive.google.com/file/d/1inkHp24an1TyWvBekBxu2wRIyLQ0gkhO/view?usp=drive_link && unzip creeper.zip && creeper.zip",
            "gdown --fuzzy https://drive.google.com/file/d/1OpgF2ILf43jcN-226wQcxImjcfMAVOwA/view?usp=drive_link && unzip mochi-high-five.zip && mochi-high-five.zip",
            "gdown --fuzzy https://drive.google.com/file/d/1b9Y-hUm9Cviuq-fl7gG-q7rUK7j0u2Rv/view?usp=drive_link && unzip block.zip && block.zip",
            "gdown --fuzzy https://drive.google.com/file/d/1055wcQk-ZfVWXa_g-dpQIRQy-kLBL_Lk/view?usp=drive_link && unzip spin.zip && spin.zip",
            "gdown --fuzzy https://drive.google.com/file/d/1Mqm4C1Oitv4AsDM2n0Ojbt5pmF_qXVfI/view?usp=drive_link && unzip teddy.zip && teddy.zip",
            "gdown --fuzzy https://drive.google.com/file/d/1Uc2BXpONnWhxKNs6tKMle0MiSVMVZsuB/view?usp=drive_link && unzip pillow.zip && pillow.zip",
        ]

        for file in files:
            command = f"cd data; mkdir iphone; cd iphone; {file}; cd ../../"
            subprocess.run(
                command,
                shell=True
            )

    if embeddings:
        command = f"python preprocess/get_dino_prediction.py configs/iphone/dynomo_iphone.py --base_path {os.getcwd()}/data/iphone/  --save_dir {os.getcwd()}/data/iphone/ --model {embedding_model}"
        subprocess.run(
            command,
            shell=True
        )

    if depth:
        command = f'python preprocess/get_depth_anything_prediction.py -m zoedepth --pretrained_resource="local::{os.getcwd()}/Depth-Anything/metric_depth/checkpoints/depth_anything_metric_depth_indoor.pt" --base_path {os.getcwd()}/data/iphone/ --save_dir {os.getcwd()}/data/iphone/'
        subprocess.run(
            command,
            shell=True
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["davis", "panoptic_sport", "iphone"],
                        help="Which dataset to use.")
    parser.add_argument("--download", action="store_true", help="If download needed.")
    parser.add_argument("--embeddings", action="store_true", help="If precompute embeddings.")
    parser.add_argument("--embedding_model", type=str, default="dinov2_vits14_reg",
                        choices=["dinov2_vits14_reg", "dinov2_vits14"], help='Which dino version to use.')
    parser.add_argument("--depths", action="store_true", help="If precompute depth.")
    parser.add_argument("--depth_model", type=str, default='DepthAnything',
                        choices=["DepthAnything", "DepthAnythingV2-vitl"], help="Which Depth Model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "davis":
        download_davis(args.download, args.embeddings, args.depths, args.embedding_model, args.depth_model)
    if args.dataset == "panoptic_sport":
        download_panoptic(args.download, args.embeddings, args.depths, args.embedding_model, args.depth_model)
    if args.dataset == "iphone":
        download_iphone(args.download, args.embeddings, args.depths, args.embedding_model, args.depth_model)
