from pathlib import Path
import logging
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import models
from .logger import print_log
from .dist_utils import is_main_process
from models import build_model_from_cfg
from loss import (
    build_criterion_from_cfg,
    build_lr_scheduler_from_cfg,
    build_weight_scheduler_from_cfg,
    BNMomentumScheduler,
)
from datasets import build_dataset_dataloader_from_cfg


def build_criterions_and_schedulers_from_cfg(criterion_cfg, scheduler_cfg):
    criterions = {}
    schedulers = {}
    for name, cfg in criterion_cfg.items():
        criterions[name] = build_criterion_from_cfg(cfg)
    for name, cfg in scheduler_cfg.items():
        schedulers[name] = build_weight_scheduler_from_cfg(cfg)
    return criterions, schedulers


def build_optimizer_from_cfg(model, optim_cfg):
    optim_cfg = copy.deepcopy(optim_cfg)
    optim_name = optim_cfg.NAME
    optim_cfg.pop("NAME")
    if optim_name == "AdamW":

        def add_weight_decay(model, weight_decay=1e-5, skip_list=(), initial_lr=0.001):
            decay = []
            no_decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                # we do not regularize biases nor Norm parameters
                if len(param.shape) == 1 or name.endswith(".bias") or "token" in name or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {"params": no_decay, "weight_decay": 0.0, "initial_lr": initial_lr},
                {"params": decay, "weight_decay": weight_decay, "initial_lr": initial_lr},
            ]

        param_groups = add_weight_decay(model, weight_decay=optim_cfg.weight_decay, initial_lr=optim_cfg.lr)
        optimizer = optim.AdamW(param_groups, **optim_cfg)
    elif optim_name == "Adam":
        optimizer = optim.Adam([{"params": model.parameters(), "initial_lr": optim_cfg["lr"]}], **optim_cfg)
    elif optim_name == "SGD":
        optimizer = optim.SGD(
            [{"params": filter(lambda p: p.requires_grad, model.parameters()), "initial_lr": optim_cfg["lr"]}],
            **optim_cfg,
        )
    else:
        raise NotImplementedError(f"{optim_name} Optimizer unknown.")

    return optimizer


def build_bnm_scheduler_from_cfg(model, bnm_cfg, last_epoch=-1):
    if bnm_cfg.get("decay_step") is not None:
        bnm_lmbd = lambda e: max(
            bnm_cfg.bn_momentum * bnm_cfg.bn_decay ** (e / bnm_cfg.decay_step), bnm_cfg.lowest_decay
        )
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    else:
        raise NotImplementedError()
    return bnm_scheduler


def save_ckpt(model, optimizer, epoch, metrics, best_metrics, filename, args, logger=None):
    if is_main_process():
        ckpt_file = args.ckpt_path / f"{filename}.pth"
        torch.save(
            {
                "model": model.module.state_dict() if args.distributed else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics.state_dict() if metrics is not None else dict(),
                "best_metrics": best_metrics.state_dict() if best_metrics is not None else dict(),
            },
            ckpt_file,
        )
        print_log(f"Save checkpoint at {ckpt_file}", logger=logger)


def save_ckpt_common(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def load_model(model, ckpt_file, logger=None):
    if not ckpt_file.exists():
        raise FileNotFoundError(f"no checkpoint file from path {ckpt_file}...")
    print_log(f"Loading weights from {ckpt_file}...", logger=logger)

    # load state dict
    state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    # parameter resume of base model
    if state_dict.get("model") is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict["model"].items()}
    elif state_dict.get("base_model") is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict["base_model"].items()}
    else:
        raise RuntimeError("mismatch of ckpt weight")
    model.load_state_dict(base_ckpt)

    epoch = -1
    if state_dict.get("epoch") is not None:
        epoch = state_dict["epoch"]
    if state_dict.get("metrics") is not None:
        metrics = state_dict["metrics"]
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = "No Metrics"
    print_log(f"Load ckpts @ {epoch} epoch (performance = {str(metrics):s})", logger=logger)
    return


def resume_model(model, args, best_metrics, logger=None):
    ckpt_file = args.ckpt_path / "last.pth"
    if not ckpt_file.exists():
        print_log(f"[RESUME] no checkpoint file from path {ckpt_file}, start from epoch 0", logger=logger)
        return 0, None
    print_log(f"[RESUME] Loading model weights from {ckpt_file}...", logger=logger)

    # load state dict
    state_dict = torch.load(ckpt_file, map_location="cpu")
    # parameter resume of base model
    if state_dict.get("model") is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict["model"].items()}
    else:
        raise RuntimeError("mismatch of ckpt weight")
    model.load_state_dict(base_ckpt)

    # metrics
    start_epoch = state_dict["epoch"]
    best_metrics_state_dict = state_dict["best_metrics"]
    best_metrics.load_state_dict(best_metrics_state_dict)

    print_log(f"[RESUME] resume ckpts @ {start_epoch - 1} epoch (best_metrics: {str(best_metrics):s})", logger=logger)
    return start_epoch, best_metrics


def resume_optimizer(optimizer, args, logger=None):
    ckpt_file = ckpt_file = args.ckpt_path / "last.pth"
    if not ckpt_file.exists():
        print_log(f"[RESUME] no checkpoint file from path {ckpt_file}...", logger=logger)
        return
    print_log(f"[RESUME] Loading optimizer from {ckpt_file}...", logger=logger)
    # load state dict
    state_dict = torch.load(ckpt_file, map_location="cpu")
    # optimizer
    optimizer.load_state_dict(state_dict["optimizer"])


def load_components(model, cfg, logger=None):
    if (weight_maps := cfg.model.get("pretrained_weight_maps")) is not None:
        if not isinstance(weight_maps, list):
            weight_maps = [weight_maps]
        pretrained_ckpts = cfg.model.pretrained_ckpts
        if not isinstance(pretrained_ckpts, list):
            pretrained_ckpts = [pretrained_ckpts]

        loaded_components = []
        state_dict = {}
        for idx, ckpt in enumerate(pretrained_ckpts):
            # map pretrained components to model components
            weight_map = weight_maps[idx]
            pretrained_state_dict = torch.load(Path(ckpt), map_location="cpu")["model"]
            for k, w in pretrained_state_dict.items():
                pretrained_component_name = k.split(".")[0]
                model_component_name = weight_map.get(pretrained_component_name)
                if (model_component_name := weight_map.get(pretrained_component_name)) is not None:
                    model_weight_name = k.replace(pretrained_component_name, model_component_name)
                    state_dict[model_weight_name] = w
                    loaded_components.append(model_weight_name)

        unmatched = model.load_state_dict(state_dict, strict=False)
        if unmatched.unexpected_keys:
            print_log(f"Keys not matched {unmatched.unexpected_keys}", logger=logger, level=logging.WARNING)
        print_log(f"Load {weight_maps} from {pretrained_ckpts}", logger=logger)
        # Freeze the loaded components
        freeze_exception = cfg.model.get("pretrained_freeze_exception", [])
        log_msg = "Following loaded weights are not frozen: "
        for k, w in model.named_parameters():
            no_freeze = False
            for no_freeze_name in freeze_exception:
                if k.startswith(no_freeze_name):
                    no_freeze = True
                    break
            if k in loaded_components and not no_freeze:
                w.requires_grad = False
                log_msg += f"{k} "
        print_log(log_msg, logger=logger, level=logging.DEBUG)
