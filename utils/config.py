import yaml
import shutil
from easydict import EasyDict
from pathlib import Path
import logging
from .logger import print_log


def log_args_to_file(args, pre="args", logger=None, level=logging.DEBUG):
    for key, val in args.__dict__.items():
        print_log(f"{pre}.{key} : {val}", logger=logger, level=level)


def log_config_to_file(cfg, pre="cfg", logger=None, level=logging.DEBUG):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f"{pre}.{key} = edict()", logger=logger, level=level)
            log_config_to_file(cfg[key], pre=pre + "." + key, logger=logger, level=level)
            continue
        print_log(f"{pre}.{key} : {val}", logger=logger, level=level)


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == "dataset_cfg":
                with open(new_config["dataset_cfg"], "r") as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file: Path):
    config = EasyDict()
    with cfg_file.open("r") as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config


def get_config(args, logger=None):
    if args.resume:
        cfg_path = args.experiment_path / "config.yaml"
        if not cfg_path.exists():
            print_log("Failed to resume", logger=logger)
            raise FileNotFoundError()
        print_log(f"Resume yaml from {cfg_path}", logger=logger)
        args.config = cfg_path
    elif args.finetune_config is not None:
        args.config = args.finetune_config
        print_log(f"Finetune yaml from {args.config}", logger=logger)
    elif args.ckpt:
        cfg_path = args.experiment_path / "config.yaml"
        if cfg_path.exists():
            print_log(f"Load saved yaml from {cfg_path}", logger=logger)
            args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    if args.finetune_config or (not args.resume and not args.ckpt and args.local_rank == 0):
        save_experiment_config(args, config, logger)
    return config


def save_experiment_config(args, config, logger=None):
    config_path = Path(args.experiment_path) / "config.yaml"
    shutil.copy2(args.config, config_path)
    print_log(f"Copy the Config file from {args.config} to {config_path}", logger=logger)
