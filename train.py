import time
from pathlib import Path
import numpy as np

import torch
from tensorboardX import SummaryWriter

from utils.parser import get_args
from utils.config import get_config, log_args_to_file, log_config_to_file
from utils.logger import setup_logger
from utils.misc import set_random_seed
from utils.runner import run_train
from utils import dist_utils


def main():
    ## configs and environment
    args = get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    else:
        raise NotImplementedError("CPU not supported")
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        args.distributed = False
        world_size = 1
        args.world_size = 1
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # init logger
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logger_path = args.experiment_path / f"train_{timestamp}.log"
    logger = setup_logger(f"Train", filename=logger_path)
    # config
    cfgs = get_config(args, logger=logger)
    # Tensorboard
    if not args.distributed or args.local_rank == 0:
        train_writer = SummaryWriter(args.tfboard_path / "train")
        val_writer = SummaryWriter(args.tfboard_path / "val")
    else:
        train_writer = None
        val_writer = None
    # batch size
    if args.distributed:
        assert cfgs.total_bs % world_size == 0
    cfgs.data.train.batch_size = cfgs.total_bs_train // world_size
    cfgs.data.val.batch_size = cfgs.total_bs_val // world_size
    cfgs.data.test.batch_size = cfgs.total_bs_val // world_size
    cfgs.data.vis.batch_size = 1
    # log
    log_args_to_file(args, "args", logger=logger)
    log_config_to_file(cfgs, "config", logger=logger)
    # set random seeds
    if args.distributed:
        set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic)  # seed + rank, for augmentation
    else:
        set_random_seed(args.seed, deterministic=args.deterministic)  # seed + rank, for augmentation

    if args.distributed:
        assert args.local_rank == dist_utils.get_rank()

    run_train(args, cfgs, train_writer, val_writer, logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
    logger.info("Finish")


if __name__ == "__main__":
    main()
