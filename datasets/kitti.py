from pathlib import Path
import json
import torch
import numpy as np
import open3d as o3d
from utils.logger import print_log
from utils.augment import Compose
from utils.io import read_point_cloud
from .build import DATASETS


# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py
@DATASETS.register_module()
class KITTIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path="data/KITTI",
        split="test",
        npoints=16384,
        npoints_input=2048,
        augment={},
        subset=None,
        logger=None,
        **kwargs,
    ):
        assert split == "test", "KITTI only supports test subset"
        self.logger = logger
        if len(kwargs) > 0:
            print_log(f"Following arguments are not used: {kwargs}", self.logger)
        self.root = Path(path)
        self.npoints = npoints
        self.is_multiview = False
        self.split = split
        self.pre_transform = Compose(
            {
                "NormalizeObjectPose": {"apply_on": 0},
                "RandomSamplePoints": {"npoints": npoints_input, "apply_on": 0},
            }
        )
        self.transform = Compose(augment)

        dataset_file = self.root / "KITTI.json"
        with dataset_file.open("r") as f:
            self.category_lists = json.loads(f.read())
        self.data_list = self._load_data(self.split, subset)

        print_log(
            f"KITTI Dataset {split} loading finish. {len(self.data_list)} models available. Length {len(self)}. "
            f"Using pre-transformation: {self.pre_transform} and augmentation: {self.transform}",
            logger=self.logger,
        )

    def _load_data(self, split, subset=None):
        """Prepare file list for the dataset"""
        if subset is not None:
            assert subset <= 1 and subset > 0

        data_list = []
        for cl in self.category_lists:
            samples = cl[split]
            if subset is not None:
                num_keep = int(len(samples) * subset)
                samples = np.random.choice(samples, num_keep, replace=False)
            for model_id in samples:
                data_list.append(
                    {
                        "taxonomy_id": cl["taxonomy_id"],
                        "model_id": model_id,
                        "partial_path": self.root / "cars" / f"{model_id}.pcd",
                        "bbox_path": self.root / "bboxes" / f"{model_id}.txt",
                    }
                )
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        partial_path = sample["partial_path"]
        partial = read_point_cloud(partial_path)
        bbox = np.loadtxt(sample["bbox_path"]).astype(np.float32)
        meta_data = {"label": sample["taxonomy_id"], "partial_id": sample["model_id"]}
        [partial], _ = self.pre_transform([partial], bbox=bbox)
        [partial], _ = self.transform([partial], bbox=bbox)
        return meta_data, partial
