from pathlib import Path
import numpy as np
import open3d as o3d

import torch

from utils.parser import get_args
from utils.config import get_config, log_args_to_file, log_config_to_file
from utils.builder import build_dataset_dataloader_from_cfg, build_model_from_cfg, load_model
from utils.logger import setup_logger
from utils.misc import set_random_seed
from utils.augment import Compose
from utils.vis_utils import save_pcd_tensor, paint_coarse_missing_fine, paint_coarse_missing, COLORS


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
    logger = setup_logger("Vis")
    # config
    cfgs = get_config(args, logger=logger)
    # log
    log_args_to_file(args, "args", logger=logger)
    log_config_to_file(cfgs, "config", logger=logger)
    # set random seeds
    set_random_seed(args.seed, deterministic=args.deterministic)
    args.distributed = False
    cfgs.data.vis.batch_size = 1
    dataset_test, dataloader_test, _ = build_dataset_dataloader_from_cfg(
        cfgs.data.vis, args, cfgs.model.num_fine, train=False, logger=logger
    )

    augment = Compose(
        {
            # "RandomScaling": {"low": 0.3, "high": 2.0},
            # "RandomMirrorPoints": {},
            "RandomRotation": {"axes": "xyz"},
            # "RandomTranslation": {"translation_range": 0.5},
            # "RandomJittering": {"std": 0.01, "maximum": 0.01},
            # "RandomCropping": {"max_cropping_ratio": 0.2},
        }
    )

    model = build_model_from_cfg(cfgs.model, None, None, verbose=args.verbose)
    ckpt_file = args.ckpt_path / args.ckpt
    load_model(model, ckpt_file, logger=logger)
    model.to(args.local_rank)
    model.eval()

    vis_dir = Path(args.save_path) / "vis" / args.exp_name
    vis_dir.mkdir(exist_ok=True)
    model_ids = [278, 1488, 6139, 7983, 12777, 15685, 16423, 20050, 20563, 25848, 26446, 30872]
    model_ids += [179, 474, 16818, 25629, 29543]
    with torch.no_grad():
        for idx in model_ids:
            # idx = np.random.choice(len(dataset_test))
            # idx = 26446
            # set random seeds again
            set_random_seed(args.seed + idx, deterministic=args.deterministic)
            meta_data, [partials, completes], _ = dataset_test[idx]
            label = meta_data["label"]
            partial_id = meta_data["partial_id"]
            print("==========", partial_id, label, "==========")

            augmented, originals = augment([partials, completes])
            partials = augmented[0].unsqueeze(0).to(device)
            completes = augmented[1].unsqueeze(0).to(device)
            # print(f"p{idx}", partials[0].max(0)[0], partials[0].min(0)[0])
            # print(f"c{idx}", completes[0].max(0)[0], completes[0].min(0)[0])

            if "PCN" in cfgs.model.NAME:
                coarse, fine = model(partials)
                num_missing_anchors = 0
            elif "PCN" in cfgs.data.vis.dataset_cfg.dataset.NAME:
                coarse, fine = model(partials)
                num_missing_anchors = getattr(model, "num_missing_anchors", 0)
            else:
                coarse, fine = model(partials)
                num_missing_anchors = getattr(model, "num_missing_anchors", 0)

            save_pcd_tensor(partials[0], None, str(vis_dir / f"{label}_{partial_id}_partial.ply"))
            save_pcd_tensor(completes[0], None, str(vis_dir / f"{label}_{partial_id}_gt.ply"))
            save_pcd_tensor(fine[0], None, str(vis_dir / f"{label}_{partial_id}_fine.ply"))

            coarse_fine_missing, fine_color = paint_coarse_missing_fine(
                coarse[0],
                fine[0],
                num_missing_anchors,
                COLORS["yellow"],
                COLORS["gray"],
                COLORS["red"],
            )
            save_pcd_tensor(coarse_fine_missing, fine_color, str(vis_dir / f"{label}_{partial_id}_all.ply"))


if __name__ == "__main__":
    main()
