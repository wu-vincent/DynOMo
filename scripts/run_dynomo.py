from src.model.dynomo import DynOMo
from importlib.machinery import SourceFileLoader
from src.utils.common_utils import seed_everything
import os
import shutil
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--just_eval", default=0, type=int, help="if only eval")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()
    experiment.config['just_eval'] = args.just_eval

    if experiment.config['just_eval']:
        experiment.config['use_wandb'] = False
        experiment.config['checkpoint'] = True

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )

    dynomo = DynOMo(experiment.config)
    if experiment.config['just_eval']:
        dynomo.eval()
    else:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))
        dynomo.track()