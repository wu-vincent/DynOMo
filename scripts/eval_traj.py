
import argparse
import os
import shutil
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

from utils.common_utils import seed_everything
from utils.eval_traj import eval_traj, vis_grid_trajs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )

    metrics = eval_traj(experiment.config, results_dir=results_dir, vis_trajs=experiment.config['viz']['vis_tracked'])
    print(metrics)
    if experiment.config['viz']['vis_grid']:
        vis_grid_trajs(experiment.config, params=None, cam=None, results_dir=results_dir)
        print(f"Stored visualizations to {results_dir}...")