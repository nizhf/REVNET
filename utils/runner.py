from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import logging
import pandas as pd

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from .builder import (
    build_model_from_cfg,
    build_criterions_and_schedulers_from_cfg,
    build_dataset_dataloader_from_cfg,
    build_lr_scheduler_from_cfg,
    build_optimizer_from_cfg,
    build_bnm_scheduler_from_cfg,
    save_ckpt,
    resume_model,
    resume_optimizer,
    load_model,
    load_components,
)
from .metrics import MetricLogger
from .logger import print_log
from .dist_utils import has_batchnorms
from .runner_completion import (
    COMPLETION_METRIC_NAMES,
    COMPLETION_GREAT_IS_BETTER,
    COMPLETION_BEST_METRIC_NAMES,
    train_one_batch_completion,
    run_val_completion,
    run_test_kitti_completion,
)


def run_train(args, cfgs, train_writer: SummaryWriter, val_writer: SummaryWriter, logger):
    # task
    task = cfgs.task
    # dataset and model
    dataset_train, dataloader_train, sampler_train = build_dataset_dataloader_from_cfg(
        cfg=cfgs.data.train, args=args, npoints=cfgs.model.num_fine, train=True, logger=logger
    )
    dataset_val, dataloader_val, sampler_val = build_dataset_dataloader_from_cfg(
        cfg=cfgs.data.val, args=args, npoints=cfgs.model.num_fine, train=False, logger=logger
    )
    nbatches = len(dataloader_train)
    device = args.local_rank
    finetune = args.finetune_config is not None
    # metrics
    best_metrics = None
    best_epoch = -1
    if "completion" in task:
        all_metric_names = COMPLETION_METRIC_NAMES
        great_is_better = COMPLETION_GREAT_IS_BETTER
        best_metric_names = COMPLETION_BEST_METRIC_NAMES
        best_metric_dict = {f"best_{name}": None for name in best_metric_names}
        best_epoch_dict = {f"best_{name}": -1 for name in best_metric_names}
    else:
        raise NotImplementedError(f"{task} not supported.")

    # loss functions and weight schedulers
    loss_fn, loss_scheduler = build_criterions_and_schedulers_from_cfg(cfgs.loss, cfgs.loss_weight_scheduler)
    # model
    start_epoch = 0
    model = build_model_from_cfg(cfgs.model, loss_fn=loss_fn, loss_scheduler=loss_scheduler, verbose=args.verbose)
    # load pretrained components
    load_components(model, cfgs, logger=logger)
    # resume model
    if args.resume:
        best_metrics = MetricLogger(all_metric_names, great_is_better)
        start_epoch, best_metrics = resume_model(model, args, best_metrics, logger=logger)
    elif args.ckpt is not None:
        if finetune:
            ckpt_file = args.finetune_ckpt_path / args.ckpt
        else:
            ckpt_file = args.ckpt_path / args.ckpt
        load_model(model, ckpt_file, logger=logger)

    # model to GPUs
    if args.distributed:
        # synchronize batch norms (if any)
        if has_batchnorms(model):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % torch.cuda.device_count()])
    else:
        model.to(device)

    # clip gradient by norm
    if (clip_grad := cfgs.get("clip_grad")) is not None:
        if clip_grad <= 0:
            clip_grad = None
    # optimizer
    optimizer = build_optimizer_from_cfg(model, cfgs.optimizer)
    if args.resume:
        resume_optimizer(optimizer, args, logger=logger)
    scaler = torch.GradScaler(device=device)
    # scheduler
    scheduler = build_lr_scheduler_from_cfg(cfgs.scheduler, optimizer, last_epoch=start_epoch - 1)
    # batchnorm scheduler for PoinTr
    if cfgs.get("bnmscheduler") is not None:
        bn_scheduler = build_bnm_scheduler_from_cfg(model, cfgs.bnmscheduler)
    else:
        bn_scheduler = None
    # finetuning mode, do an initial evaluation
    if finetune:
        epoch = -1
        metrics = run_val(
            model=model,
            dataloader=dataloader_val,
            task=task,
            writer=val_writer,
            epoch=epoch,
            distributed=args.distributed,
            device=device,
        )
        print_log(f"Finetune Initial Val: {metrics}", logger=logger)
        best_metrics = metrics
        save_ckpt(model, optimizer, epoch, metrics, metrics, "best", args, logger=logger)
        save_ckpt(model, optimizer, epoch, metrics, metrics, "last", args, logger=logger)
        for name in best_metric_names:
            best_metric_dict[f"best_{name}"] = metrics
            save_ckpt(
                model,
                optimizer,
                epoch,
                metrics,
                metrics,
                f"best_{name}",
                args,
                logger=logger,
            )
    # run
    for epoch in range(start_epoch, cfgs.max_epoch):
        # Adjust KITTI random dropout rate
        if "KITTI" in cfgs.data.test.dataset_cfg.dataset.NAME:
            kitti_transforms = dataset_train.transform
            for t in kitti_transforms.transforms:
                if hasattr(t, "min_num_points"):
                    t.min_num_points = max(int(200 / (epoch // 5 + 1)), 10)
                    print_log(f"Adjust random point dropping to min {t.min_num_points}", logger=logger)
        train_metrics = MetricLogger(model.loss_names, False)
        model.train()
        # reshuffle multiview split
        if getattr(dataset_train, "is_multiview", False):
            dataset_train.reshuffle()
        # set distributed data sampler
        if args.distributed:
            sampler_train.set_epoch(epoch)
        for it, batch in enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch} Train: ", unit="batch")):
            it = nbatches * epoch + it
            # use individual batch training
            if "completion" in task:
                loss_dict = train_one_batch_completion(
                    model=model,
                    data=batch,
                    optimizer=optimizer,
                    scaler=scaler,
                    clip_grad=clip_grad,
                    device=device,
                    logger=logger,
                )
            else:
                raise NotImplementedError(f"{task} not supported.")
            # log step
            if args.distributed:
                torch.cuda.synchronize()
            train_metrics.update(
                {name: loss.detach().cpu().item() for name, loss in loss_dict.items() if "loss" in name}
            )
            if train_writer is not None:
                for name in train_metrics.metric_names:
                    train_writer.add_scalar(f"Loss/Batch/{name}", train_metrics.val(name), it)
        # log epoch
        epoch_lr = optimizer.param_groups[0]["lr"]
        # gather the stats from all processes
        train_metrics.synchronize_between_processes()
        print_log(f"Epoch {epoch} Train: lr {epoch_lr}, {train_metrics}", logger=logger)
        # update train tensorboard
        if train_writer is not None:
            for name, loss in train_metrics.state_dict().items():
                train_writer.add_scalar(f"Loss/Epoch/{name}", loss, epoch)
            train_writer.add_scalar("Loss/Epoch/lr", epoch_lr, epoch)
        # lr scheduler step
        scheduler.step()
        # for PoinTr
        if bn_scheduler is not None:
            bn_scheduler.step()
        # validation
        if epoch % args.val_freq == 0 or epoch == cfgs.max_epoch - 1:
            metrics = run_val(
                model=model,
                dataloader=dataloader_val,
                task=task,
                writer=val_writer,
                epoch=epoch,
                distributed=args.distributed,
                device=device,
            )
            print_log(f"Epoch {epoch} Val: {metrics}", logger=logger)
            # save overall best ckpt
            if metrics.better_than(best_metrics, all_metric_names):
                best_metrics = metrics
                best_epoch = epoch
                save_ckpt(model, optimizer, epoch, metrics, best_metrics, "best", args, logger=logger)
            # save best single metric ckpt
            for name in best_metric_names:
                if metrics.better_than(best_metric_dict[f"best_{name}"], [name]):
                    best_metric_dict[f"best_{name}"] = metrics
                    best_epoch_dict[f"best_{name}"] = epoch
                    save_ckpt(model, optimizer, epoch, metrics, metrics, f"best_{name}", args, logger=logger)

        # save as the last ckpt, no need to log
        save_ckpt(model, optimizer, epoch, metrics, best_metrics, "last", args, logger=None)

    # log best epoch
    print_log(f"Best all @ {best_epoch} epoch {best_metrics}", logger=logger)
    for name in best_metric_names:
        temp_best_metric = best_metric_dict[f"best_{name}"]
        temp_best_epoch = best_epoch_dict[f"best_{name}"]
        print_log(f"Best {name} @ {temp_best_epoch} epoch {temp_best_metric}", logger=logger)


def run_val(model, dataloader, task, writer, epoch, distributed, device):
    model.eval()
    if "completion" in task:
        metrics = run_val_completion(
            model=model,
            dataloader=dataloader,
            writer=writer,
            epoch=epoch,
            distributed=distributed,
            device=device,
        )[0]
    else:
        raise NotImplementedError(f"{task} not supported.")
    return metrics


def run_test(args, cfgs, logger: logging.Logger):
    # task
    task = cfgs.task
    # dataset and model
    dataset_test, dataloader_test, _ = build_dataset_dataloader_from_cfg(
        cfgs.data.test, args, cfgs.model.num_fine, train=False, logger=logger
    )
    device = args.local_rank
    model = build_model_from_cfg(cfgs.model, None, None, verbose=args.verbose)
    ckpt_file = args.ckpt_path / args.ckpt
    load_model(model, ckpt_file, logger=logger)
    model.to(device)

    if "completion" in task:
        if "KITTI" in dataset_test.__class__.__name__:
            vis_path = args.experiment_path / "vis_KITTI"
            if not vis_path.exists():
                vis_path.mkdir()
            run_test_kitti_completion(
                model=model,
                dataloader=dataloader_test,
                distributed=args.distributed,
                device=device,
                vis_path=vis_path,
                logger=logger,
            )
            metrics, per_class_metrics, per_model_metrics = None, None, None
        else:
            metrics, per_class_metrics, per_model_metrics = run_val_completion(
                model=model,
                dataloader=dataloader_test,
                writer=None,
                epoch=-1,
                distributed=args.distributed,
                device=device,
                per_class=True,
                per_model=True,
            )
    else:
        raise NotImplementedError(f"{task} not supported.")

    if metrics is not None:
        print_log(f"Test: {metrics}", logger=logger)
    if per_class_metrics is not None:
        for label, metrics in per_class_metrics.items():
            print_log(f"Class {label}: {metrics}", logger=logger)
    if per_model_metrics is not None:
        per_model_metrics = pd.DataFrame(per_model_metrics)
        per_model_metrics.to_csv(args.experiment_path / "per_model_metrics.csv")
