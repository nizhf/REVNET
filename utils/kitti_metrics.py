from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import math

from datasets import PCNCarKITTITestDataset, MVPCarKITTITestDataset
from extensions.chamfer3D import dist_chamfer_3D
from .io import read_point_cloud
from .logger import print_log


chamfer_distance_ext = dist_chamfer_3D.chamfer_3DDist()


def get_Fidelity(samples, logger=None):
    # Fidelity Error, CD-T (L2 distance)
    metric = []
    for frame_name in tqdm(samples):
        if (Path(frame_name) / "input.ply").exists():
            input_data = read_point_cloud(frame_name / "input.ply").unsqueeze(0).cuda()
            pred_data = read_point_cloud(frame_name / "pred.ply").unsqueeze(0).cuda()
        elif (Path(frame_name) / "input.npy").exists():
            input_data = torch.from_numpy(np.load(frame_name / "input.npy")).unsqueeze(0).cuda()
            pred_data = torch.from_numpy(np.load(frame_name / "pred.npy")).unsqueeze(0).cuda()
        else:
            print_log(
                f"Cannot find a valid input .ply or .npy file in {frame_name}. Skip.",
                logger=logger,
            )
            continue
        cd_t_in_to_pred = chamfer_distance_ext(input_data, pred_data)[0].mean().cpu().item()
        if math.isnan(cd_t_in_to_pred):
            print_log(
                f"{frame_name.name} nan Fidelity. # Valid input points: {((input_data != 0).sum() // 3).item()}",
                logger=logger,
            )
            continue
        metric.append(cd_t_in_to_pred)
    fidelity = np.mean(metric)
    print_log(f"Total valid #samples {len(metric)}/{len(samples)}, Fidelity {fidelity}", logger=logger)
    return fidelity


def get_Consistency(samples):
    # Consistency
    cars_dict = {}
    dataroot = samples[0].parent
    for folder_name in samples:
        frame_name = folder_name.name
        all_elements = frame_name.split("_")  # example sample = 'frame_1_car_3_647'
        frame_id = int(all_elements[1])
        car_id = int(all_elements[-2])
        sample_id = int(all_elements[-1])

        if cars_dict.get(car_id) is None:
            cars_dict[car_id] = [f"frame_{frame_id:03d}_car_{car_id:02d}_{sample_id:03d}"]
        else:
            # example sample = 'frame_001_car_003_647'
            cars_dict[car_id].append(f"frame_{frame_id:03d}_car_{car_id:02d}_{sample_id:03d}")

    consistency = []
    for key, car_list in cars_dict.items():
        car_list = sorted(car_list)
        each_car_consistency = []
        for i, this_car in enumerate(car_list):
            if i == len(car_list) - 1:
                break
            this_elements = this_car.split("_")
            this_frame = int(this_elements[1])

            next_car = car_list[i + 1]
            next_elements = next_car.split("_")
            next_frame = int(next_elements[1])

            if next_frame - 1 != this_frame:
                continue

            this_car_path = (
                dataroot / f"frame_{this_frame}_car_{int(this_elements[3])}_{int(this_elements[4]):03d}" / "pred.ply"
            )
            this_car = read_point_cloud(this_car_path).unsqueeze(0).cuda()
            next_car_path = (
                dataroot / f"frame_{next_frame}_car_{int(next_elements[3])}_{int(next_elements[4]):03d}" / "pred.ply"
            )
            next_car = read_point_cloud(next_car_path).unsqueeze(0).cuda()

            cd_p2g, cd_g2p, _, _ = chamfer_distance_ext(this_car, next_car)
            cd_t = torch.mean(cd_p2g, dim=1) + torch.mean(cd_g2p, dim=1)
            each_car_consistency.append(cd_t.mean())

        consistency.append(np.mean(each_car_consistency))
    consistency = np.mean(consistency)
    return consistency


def get_MMD(car_dataloader, samples, logger=None):
    # MMD
    metric_cd_t = []
    metric_cd_p = []
    for i, frame_name in enumerate(samples):
        if (Path(frame_name) / "input.ply").exists():
            part_data = read_point_cloud(frame_name / "input.ply").unsqueeze(0).cuda()
            pred_data = read_point_cloud(frame_name / "pred.ply").unsqueeze(0).cuda()
        elif (Path(frame_name) / "input.npy").exists():
            part_data = torch.from_numpy(np.load(frame_name / "input.npy")).unsqueeze(0).cuda()
            pred_data = torch.from_numpy(np.load(frame_name / "pred.npy")).unsqueeze(0).cuda()
        else:
            print_log(
                f"Cannot find a valid input .ply or .npy file in {frame_name}. Skip.",
                logger=logger,
            )
            continue
        if torch.any(pred_data.isnan()):
            print_log(f"{i + 1}/{len(samples)}: {frame_name.name} has nan coordinates. Skip", logger=logger)
            continue
        if (torch.sum((part_data**2).sum(dim=-1) > 1e-9)).item() <= 5:
            print_log(f"{i + 1}/{len(samples)}: {frame_name.name} has less than 5 points. Skip", logger=logger)
            continue
        batch_cd_t = []
        batch_cd_p = []
        for meta_data, gt in car_dataloader:
            gt = gt.cuda()
            batch_pred_data = pred_data.expand(gt.shape[0], -1, -1)
            cd_p2g, cd_g2p, _, _ = chamfer_distance_ext(batch_pred_data, gt)
            cd_t = torch.mean(cd_p2g, dim=1) + torch.mean(cd_g2p, dim=1)
            cd_p = (torch.sqrt(cd_p2g).mean(dim=1) + torch.sqrt(cd_g2p).mean(dim=1)) / 2
            batch_cd_t.append(cd_t)
            batch_cd_p.append(cd_p)
        batch_cd_t = torch.cat(batch_cd_t)
        batch_cd_p = torch.cat(batch_cd_p)
        min_cd_t = batch_cd_t.min().cpu().item()
        min_cd_p = batch_cd_p.min().cpu().item()
        metric_cd_t.append(min_cd_t)
        metric_cd_p.append(min_cd_p)
        print_log(
            f"{i + 1}/{len(samples)}: {frame_name.name} MMD CD-T {min_cd_t}, CD-P {min_cd_p}. "
            f"Total CD-T {np.mean(metric_cd_t)}, CD-P {np.mean(metric_cd_p)}",
            logger=logger,
        )
    mmd_t = np.mean(metric_cd_t)
    mmd_p = np.mean(metric_cd_p)
    print_log(f"Total valid #samples {len(metric_cd_t)}/{len(samples)}, MMD CD-T {mmd_t}, CD-P {mmd_p}", logger=logger)
    return mmd_t, mmd_p


def get_metrics_kitti(vis_path: Path, dataset="PCN", batch_size=32, num_workers=1, logger=None):
    if dataset == "PCN":
        car_dataset = PCNCarKITTITestDataset("data/PCN", logger=logger)
    elif dataset == "MVP":
        car_dataset = MVPCarKITTITestDataset("data/MVP", logger=logger)
    car_dataloader = DataLoader(car_dataset, batch_size=batch_size, num_workers=num_workers)
    samples = [d for d in vis_path.iterdir() if d.is_dir()]
    samples = sorted(samples)
    fidelity = get_Fidelity(samples, logger=logger)
    mmd_t, mmd_p = get_MMD(car_dataloader, samples, logger=logger)
    return fidelity, mmd_t, mmd_p
