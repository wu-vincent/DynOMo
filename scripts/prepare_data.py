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
    data_dir = Path("data/iphone")
    data_dir.mkdir(parents=True, exist_ok=True)

    if download:
        gdown_files = [
            ("15PirJRqsT5lLjuGdLWALBDFMQanj8FTh", "paper-windmill.zip"),
            ("18sjQQMU6AijyXg4BoucLX82R959BYAzz", "sriracha-tree.zip"),
            ("1QihG5A7c_bpkse5b0OBdqqFThgX0kDyZ", "bagpack.zip"),
            ("1QJQnVw_szoy_k5x9k_BAE2BWtf_BXqKn", "apple.zip"),
            ("1BmuxJXKi6dVaNOjmppuETQsaspAV9Wca", "haru-sit.zip"),
            ("1frv8miU24Dl7fqblYt7zkwj129ci-68U", "handwavy.zip"),
            ("1inkHp24an1TyWvBekBxu2wRIyLQ0gkhO", "creeper.zip"),
            ("1OpgF2ILf43jcN-226wQcxImjcfMAVOwA", "mochi-high-five.zip"),
            ("1b9Y-hUm9Cviuq-fl7gG-q7rUK7j0u2Rv", "block.zip"),
            ("1055wcQk-ZfVWXa_g-dpQIRQy-kLBL_Lk", "spin.zip"),
            ("1Mqm4C1Oitv4AsDM2n0Ojbt5pmF_qXVfI", "teddy.zip"),
            ("1Uc2BXpONnWhxKNs6tKMle0MiSVMVZsuB", "pillow.zip"),
        ]  # Add remaining files in the same format

        for file_id, filename in gdown_files:
            zip_path = data_dir / filename
            gdown.download(id=file_id, output=str(zip_path), quiet=False)
            extract_zip(zip_path, data_dir)

    if embeddings:
        subprocess.run([
            "python", "preprocess/get_dino_prediction.py", "configs/iphone/dynomo_iphone.py",
            "--base_path", str(data_dir),
            "--save_dir", str(data_dir),
            "--model", embedding_model
        ], check=True)

    if depth:
        subprocess.run([
            "python", "preprocess/get_depth_anything_prediction.py", "-m", "zoedepth",
            "--pretrained_resource",
            f"local::Depth-Anything/metric_depth/checkpoints/depth_anything_metric_depth_indoor.pt",
            "--base_path", str(data_dir),
            "--save_dir", str(data_dir)
        ], check=True)

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
