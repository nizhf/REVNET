from pathlib import Path
import time
import argparse
from utils.logger import print_log, setup_logger
from utils.kitti_metrics import get_metrics_kitti


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", type=str, help="experiment path")
    parser.add_argument("--batch_size", type=int, default=32, help="experiment path")
    parser.add_argument("--num_workers", type=int, default=1, help="experiment path")
    parser.add_argument("--dataset", type=str, default="PCN", help="reference dataset name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    experiment_path = Path(args.experiment_path)
    vis_path = experiment_path / "vis_KITTI"
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logger_path = experiment_path / f"kitti_{timestamp}.log"
    kitti_logger = setup_logger("KITTI", logger_path)
    fidelity, mmd_t, mmd_p = get_metrics_kitti(
        vis_path, dataset=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers, logger=kitti_logger
    )

    print_log(f"Test on KITTI: Fidelity {fidelity}, MMD CD-T {mmd_t}, CD-P {mmd_p}", logger=kitti_logger)
