from utils import registry

MODELS = registry.Registry("models")


def build_model_from_cfg(cfg, loss_fn, loss_scheduler, verbose=False):
    """
    Build a model, defined by `NAME`.
    Args:
        cfg (eDICT):
    Returns:
        Model: a constructed model specified by NAME.
    """
    default_args = {"loss_fn": loss_fn, "loss_scheduler": loss_scheduler, "verbose": verbose}
    return MODELS.build(cfg, default_args=default_args)
