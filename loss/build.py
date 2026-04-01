from utils import registry

LOSS = registry.Registry("loss")
LR_SCHEDULERS = registry.Registry("lr_schedulers")
WEIGHT_SCHEDULERS = registry.Registry("weight_schedulers")


def build_criterion_from_cfg(cfg, **kwargs):
    """
    Build a criterion (loss function), defined by cfg.NAME.
    Args:
        cfg (eDICT):
    Returns:
        criterion: a constructed loss function specified by cfg.NAME
    """
    return LOSS.build(cfg, **kwargs)


def build_lr_scheduler_from_cfg(cfg, optimizer, last_epoch=-1, **kwargs):
    """
    Build a learning rate scheduler, defined by cfg.NAME.
    Args:
        cfg (eDICT):
    Returns:
        lr_scheduler: a constructed learning rate scheduler specified by cfg.NAME
    """
    default_args = {
        "optimizer": optimizer,
        "last_epoch": last_epoch
    }
    return LR_SCHEDULERS.build(cfg, default_args=default_args, **kwargs)


def build_weight_scheduler_from_cfg(cfg, **kwargs):
    """
    Build a weight scheduler, defined by cfg.NAME.
    Args:
        cfg (eDICT):
    Returns:
        weight scheduler: a constructed weight scheduler specified by cfg.NAME
    """
    return WEIGHT_SCHEDULERS.build(cfg, **kwargs)
