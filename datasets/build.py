from torch.utils.data import DistributedSampler, DataLoader
from utils import registry
from utils.logger import print_log

DATASETS = registry.Registry("datasets")


def build_dataset_from_cfg(cfg, args):
    """
    Build a model, defined by `NAME`.
    Args:
        cfg (eDICT):
    Returns:
        Model: a constructed model specified by NAME.
    """
    return DATASETS.build(cfg, default_args=args)


def build_dataset_dataloader_from_cfg(cfg, args, npoints, train=True, logger=None):
    dataset_cfg = cfg.dataset_cfg
    split = cfg.get("split", "train" if train else "test")
    extra_args = {
        "split": split,
        "npoints": npoints,
        "subset": cfg.get("subset", None),
        "logger": logger,
    }
    if len(args.classes) > 0:
        extra_args["classes"] = args.classes
    if cfg.get("augment") is not None:
        if len(cfg.augment) > 0:
            extra_args["augment"] = cfg.augment
    dataset = build_dataset_from_cfg(dataset_cfg.dataset, extra_args)

    if dataset_cfg.batch_sampler is None:
        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=train)
            bsampler = None
            bs = cfg.batch_size
            shuffle = None
            print_log(f"Use default distributed sampler.", logger=logger)
        else:
            sampler = None
            bsampler = None
            bs = cfg.batch_size
            shuffle = train
        sampler_return = sampler
    else:
        sampler = None
        bsampler_class = DATASETS.get(dataset_cfg.batch_sampler)
        bsampler = bsampler_class(
            dataset,
            shuffle=train,
            batch_size=cfg.batch_size,
            drop_last=False,
            distributed=args.distributed,
            logger=logger,
        )
        bs = 1
        shuffle = None
        sampler_return = bsampler

    if dataset_cfg.collate_fn is None:
        collate_fn = None
    else:
        collate_fn = DATASETS.get(dataset_cfg.collate_fn)

    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=bsampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return dataset, dataloader, sampler_return
