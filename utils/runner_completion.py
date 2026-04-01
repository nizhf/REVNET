from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import logging

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from .metrics import cd_fscore_emd, MetricLogger
from .logger import print_log
from .vis_utils import visualize_KITTI

COMPLETION_METRIC_NAMES = ["cd_p", "cd_t", "f1_0.01", "f1_0.02"]
COMPLETION_GREAT_IS_BETTER = [False, False, True, True]
COMPLETION_BEST_METRIC_NAMES = ["cd_p", "f1_0.02"]


def train_one_batch_completion(model, data, optimizer, scaler, clip_grad, device, logger):
    meta_data, augmented, original = data
    partials = augmented[0].to(device)
    completes = augmented[1].to(device)
    # step loss weight scheduler
    for _, sche in model.loss_scheduler.items():
        sche.step()
    # forward and compute loss
    loss_dict = model(partials, gt=completes)
    train_loss = loss_dict["loss"]
    if torch.isnan(train_loss):
        print_log(f"NaN encountered in loss, stop.", logger=logger, level=logging.ERROR)
        print_log(f"{meta_data}", logger=logger, level=logging.ERROR)
        raise ValueError(f"NaN in loss")
    # backward
    optimizer.zero_grad()
    scaler.scale(train_loss).backward()
    if clip_grad:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    scaler.step(optimizer)
    scaler.update()
    return loss_dict


def run_val_completion(
    model,
    dataloader,
    writer: SummaryWriter,
    epoch,
    distributed,
    device,
    per_class=False,
    per_model=False,
):
    model.eval()
    # just log CD for coarse, not used for best decision
    metric_names = [*COMPLETION_METRIC_NAMES, "cd_p_coarse", "cd_t_coarse"]
    metric_names_per_class = COMPLETION_METRIC_NAMES
    great_is_better = [*COMPLETION_GREAT_IS_BETTER, False, False]
    val_metrics = MetricLogger(metric_names, great_is_better)
    per_class_metrics = defaultdict(lambda: MetricLogger(metric_names_per_class, great_is_better))
    per_model_metrics = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Val: ", unit="batch"):
            meta_data, augmented, original = batch
            partials = augmented[0].to(device)  # (B, N, 3)
            completes = augmented[1].to(device)  # (B, N, 3)

            preds = model(partials)
            coarse = preds[0]
            fine = preds[1]
            cd_p, cd_t, f1_list, _ = cd_fscore_emd(fine, completes, threshold=[0.01, 0.02], compute_emd=False)
            cd_p_coarse, cd_t_coarse, _, _ = cd_fscore_emd(coarse, completes, threshold=[0.01, 0.02], compute_emd=False)
            if distributed:
                torch.cuda.synchronize()
            val_metrics.update(
                {
                    "cd_p": cd_p.sum().item() * 1e3,
                    "cd_t": cd_t.sum().item() * 1e4,
                    "f1_0.01": f1_list[0].sum().item(),
                    "f1_0.02": f1_list[1].sum().item(),
                    "cd_p_coarse": cd_p_coarse.sum().item() * 1e3,
                    "cd_t_coarse": cd_t_coarse.sum().item() * 1e4,
                },
                count=cd_p.shape[0],
            )
            if per_class:
                labels = meta_data["label"]
                if isinstance(labels, torch.Tensor):
                    labels = labels.flatten()
                for i in range(len(labels)):
                    label = labels[i]
                    if isinstance(label, torch.Tensor):
                        label = int(label.item())
                    label = str(label)
                    per_class_metrics[label].update(
                        {
                            "cd_p": cd_p[i].item() * 1e3,
                            "cd_t": cd_t[i].item() * 1e4,
                            "f1_0.01": f1_list[0][i].sum().item(),
                            "f1_0.02": f1_list[1][i].sum().item(),
                        },
                        count=1,
                    )
            if per_model:
                labels = meta_data["label"]
                if isinstance(labels, torch.Tensor):
                    labels = labels.flatten()
                partial_ids = meta_data["partial_id"]
                if isinstance(partial_ids, torch.Tensor):
                    partial_ids = partial_ids.flatten()

                for i in range(len(labels)):
                    label = labels[i]
                    if isinstance(label, torch.Tensor):
                        label = int(label.item())
                    label = str(label)

                    partial_id = partial_ids[i]
                    if isinstance(partial_id, torch.Tensor):
                        partial_id = int(partial_id.item())
                    partial_id = str(partial_id)

                    per_model_metrics.append(
                        {
                            "id": partial_id,
                            "label": label,
                            "cd_p": cd_p[i].item() * 1e3,
                            "cd_t": cd_t[i].item() * 1e4,
                            "f1_0.01": f1_list[0][i].sum().item(),
                            "f1_0.02": f1_list[1][i].sum().item(),
                        }
                    )
    # gather the stats from all processes
    val_metrics.synchronize_between_processes()
    # log metrics
    if writer is not None:
        writer.add_scalar("Val/Epoch/CD_P", val_metrics.avg("cd_p"), epoch)
        writer.add_scalar("Val/Epoch/CD_T", val_metrics.avg("cd_t"), epoch)
        writer.add_scalar(f"Val/Epoch/FScore_0.01", val_metrics.avg(f"f1_0.01"), epoch)
        writer.add_scalar(f"Val/Epoch/FScore_0.02", val_metrics.avg(f"f1_0.02"), epoch)
    return val_metrics, per_class_metrics, per_model_metrics


def run_test_kitti_completion(model, dataloader, distributed, device, vis_path, logger=None):
    model.eval()
    outbox_prediction = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Test: ", unit="batch"):
            meta_data, partials = batch
            model_ids = meta_data["partial_id"]
            partials = partials.to(device)
            fine = model(partials)[1]
            if torch.any(torch.isnan(fine)).item():
                for i, (model_id, data) in enumerate(zip(model_ids, fine)):
                    if torch.any(torch.isnan(data)):
                        print_log(f"Model {model_id} has nan coordinates", logger=logger, level=logging.WARNING)
            if distributed:
                torch.cuda.synchronize()
            for i in range(len(model_ids)):
                if torch.any(torch.abs(fine[i]) > 1.0):
                    outbox_prediction += 1
                # discard invalid points
                discard = torch.any(torch.abs(fine[i]) >= 1.0, dim=1)
                fine[i][discard] = 0
                save_path = vis_path / f"{model_ids[i]}"
                if not save_path.exists():
                    save_path.mkdir()
                visualize_KITTI(save_path, [partials[i].cpu(), fine[i].cpu()])
    print_log(f"{outbox_prediction} out of 1.0 box")
    print_log("All KITTI predicted models saved. Use run_kitti_metrics.py to calculate the metrics.", logger=logger)
