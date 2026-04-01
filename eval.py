import time
from pathlib import Path
import numpy as np

import torch

from utils.parser import get_args
from utils.config import get_config, log_args_to_file, log_config_to_file
from utils.logger import setup_logger
from utils.misc import set_random_seed
from utils.runner import run_test
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
    if args.launcher == 'none':
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
    logger_path = args.experiment_path / f"eval_{timestamp}.log"
    logger = setup_logger(f"Eval", filename=logger_path)
    # config
    config = get_config(args, logger=logger)
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
    config.data.train.batch_size = config.total_bs_train // world_size
    config.data.val.batch_size = config.total_bs_val // world_size
    config.data.test.batch_size = config.total_bs_test // world_size
    config.data.vis.batch_size = 1
    # log
    log_args_to_file(args, "args", logger=logger)
    log_config_to_file(config, "config", logger=logger)
    # set random seeds
    if args.distributed:
        set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic)  # seed + rank, for augmentation
    else:
        set_random_seed(args.seed, deterministic=args.deterministic)  # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == dist_utils.get_rank() 

    run_test(args, config, logger=logger)
    logger.info("Finish")


if __name__ == "__main__":
    main()
