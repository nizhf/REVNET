import numpy as np
import time
from tqdm import tqdm
from collections import defaultdict
import torch

from utils.parser import get_args
from utils.config import get_config, log_args_to_file, log_config_to_file
from utils.builder import build_dataset_dataloader_from_cfg, build_model_from_cfg, load_model
from utils.logger import setup_logger, print_log
from utils.misc import set_random_seed
from utils.augment import Compose
from utils.metrics import cd_fscore_emd, MetricLogger


def main():
    args = get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        raise NotImplementedError("CPU not supported")
    # init logger
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logger_path = args.experiment_path / f"consistency_{timestamp}.log"
    logger = setup_logger(f"Consistency", filename=logger_path)
    # config
    cfgs = get_config(args, logger=logger)
    # log
    log_args_to_file(args, "args", logger=logger)
    log_config_to_file(cfgs, "config", logger=logger)
    # set random seeds
    set_random_seed(args.seed, deterministic=args.deterministic)
    args.distributed = False
    cfgs.data.test.batch_size = 1
    dataset_test, _, _ = build_dataset_dataloader_from_cfg(
        cfgs.data.test, args, cfgs.model.num_fine, train=False, logger=logger
    )

    rotation_augment = Compose({"RandomRotation": {"axes": "xyz"}})

    model = build_model_from_cfg(cfgs.model, None, None, verbose=args.verbose)
    ckpt_file = args.ckpt_path / args.ckpt
    load_model(model, ckpt_file, logger=logger)
    model.to(args.local_rank)
    model.eval()

    loop = 30
    metric_names = ["cd_p_consistency", "cd_t_consistency", "f1_0.01_consistency", "f1_0.02_consistency"]
    great_is_better = [False, False, False, False]
    consistency_metrics = MetricLogger(metric_names, great_is_better)
    per_class_metrics = defaultdict(lambda: MetricLogger(metric_names, great_is_better))
    # set random seeds again
    set_random_seed(args.seed, deterministic=args.deterministic)
    with torch.no_grad():
        for idx in tqdm(range(len(dataset_test))):
            partials_batch = []
            completes_batch = []
            meta_data, data, _ = dataset_test[idx]
            label = int(meta_data["label"])
            for _ in range(loop):
                augmented, originals = rotation_augment(data)
                partials_batch.append(augmented[0].unsqueeze(0))
                completes_batch.append(augmented[1].unsqueeze(0))

            partials = torch.cat(partials_batch, dim=0).to(args.local_rank)
            completes = torch.cat(completes_batch, dim=0).to(args.local_rank)

            coarse, fine = model(partials)
            cd_p, cd_t, f1_list, _ = cd_fscore_emd(fine, completes, threshold=[0.01, 0.02], compute_emd=False)
            cd_p_consistency = torch.max(cd_p) - torch.min(cd_p)
            cd_t_consistency = torch.max(cd_t) - torch.min(cd_t)
            f1_001_consistency = torch.max(f1_list[0]) - torch.min(f1_list[0])
            f1_002_consistency = torch.max(f1_list[1]) - torch.min(f1_list[1])
            consistency_metrics.update(
                {
                    "cd_p_consistency": cd_p_consistency,
                    "cd_t_consistency": cd_t_consistency,
                    "f1_0.01_consistency": f1_001_consistency,
                    "f1_0.02_consistency": f1_002_consistency,
                },
                count=1,
            )
            per_class_metrics[label].update(
                {
                    "cd_p_consistency": cd_p_consistency,
                    "cd_t_consistency": cd_t_consistency,
                    "f1_0.01_consistency": f1_001_consistency,
                    "f1_0.02_consistency": f1_002_consistency,
                },
                count=1,
            )

    print_log(f"Total Consistency: {consistency_metrics}", logger=logger)
    for label, metrics in per_class_metrics.items():
        print_log(f"Class Consistency {label}: {metrics}", logger=logger)


if __name__ == "__main__":
    main()
