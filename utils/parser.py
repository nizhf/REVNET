import os
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cfgs/default.yaml", help="YAML config file")
    parser.add_argument("--exp_name", type=str, default="default", help="experiment name")
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none", help="job launcher")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_freq", type=int, default=1, help="test freq")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume an interrupted training")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to the model ckpt")
    parser.add_argument("--save_path", type=str, default="./", help="Where to save the experiments")
    # seed
    parser.add_argument("--seed", type=int, default=114514, help="random seed")
    parser.add_argument("--deterministic", action="store_true", help="deterministic options for CUDNN backend.")
    # bn
    parser.add_argument("--sync_bn", action="store_true", default=False, help="whether to use sync bn")
    # select specific classes
    parser.add_argument("--classes", default=[], type=str, nargs="*", help="only keep the given classes in the dataset")
    # finetuning pretrained models with new hyperparameters
    parser.add_argument("--finetune_config", type=str, default=None, help="config for finetuning model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed steps in the model inference")

    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(0)
        args.local_rank = 0
    else:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    args.config = Path(args.config)
    if args.finetune_config is not None:
        args.finetune_config = Path(args.finetune_config)
        args.finetune_ckpt_path = Path(args.save_path) / "runs" / args.exp_name / "ckpt"
        args.exp_name = f"{args.exp_name}_finetune"
    args.experiment_path = Path(args.save_path) / "runs" / args.exp_name
    args.ckpt_path = args.experiment_path / "ckpt"
    # args.tfboard_path = Path(args.save_path) / "TFBoard" / args.exp_name
    args.tfboard_path = args.experiment_path  # set to the same folder
    args.experiment_path.mkdir(parents=True, exist_ok=True)
    args.ckpt_path.mkdir(exist_ok=True)
    args.tfboard_path.mkdir(exist_ok=True, parents=True)
    return args
