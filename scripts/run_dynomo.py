import os
import sys
sys.path.append(os.getcwd())
from src.model.dynomo import DynOMo
from importlib.machinery import SourceFileLoader
from src.utils.common_utils import seed_everything
import shutil
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--just_eval", default=0, type=int, help="if only eval")
    parser.add_argument("--eval_renderings", default=1, type=int, help="if eval renderings")
    parser.add_argument("--eval_traj", default=1, type=int, help="if eval traj")
    parser.add_argument("--vis_trajs", default=0, type=int, help="if eval traj")
    parser.add_argument("--vis_grid", default=1, type=int, help="if eval traj")
    parser.add_argument("--novel_view_mode", default=None, help="if eval novel view")
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
        dynomo.eval(
            novel_view_mode=args.novel_view_mode,
            eval_renderings=args.eval_renderings,
            vis_trajs=args.vis_trajs,
            eval_traj=args.eval_traj,
            vis_grid=args.vis_grid
            )
    else:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))
        dynomo.track()