from pathlib import Path
import json
import torch
import numpy as np
import open3d as o3d
from utils.logger import print_log
from utils.augment import Compose
from utils.io import read_point_cloud
from .build import DATASETS

CAT2ID = {
    # seen categories
    "airplane": "02691156",  # plane
    "cabinet": "02933112",  # dresser
    "car": "02958343",
    "chair": "03001627",
    "lamp": "03636649",
    "sofa": "04256520",
    "table": "04379243",
    "vessel": "04530566",  # boat
    # alis for some seen categories
    "boat": "04530566",  # vessel
    "couch": "04256520",  # sofa
    "dresser": "02933112",  # cabinet
    "airplane": "02691156",  # airplane
    "watercraft": "04530566",  # boat
    # unseen categories
    "bus": "02924116",
    "bed": "02818832",
    "bookshelf": "02871439",
    "bench": "02828884",
    "guitar": "03467517",
    "motorbike": "03790512",
    "skateboard": "04225987",
    "pistol": "03948459",
}


# Adapted from https://github.com/yuxumin/PoinTr/blob/master/datasets/PCNDataset.py
@DATASETS.register_module()
class PCNDataset(torch.utils.data.Dataset):
    """
    ShapeNet dataset in "PCN: Point Completion Network". It contains 28974 training
    samples while each complete samples corresponds to 8 viewpoint partial scans, 800
    validation samples and 1200 testing samples.
    """

    def __init__(
        self,
        path="data/PCN",
        split="train",
        npoints_input=2048,
        npoints=16384,
        augment={},
        classes=[],
        subset=None,
        logger=None,
        **kwargs,
    ):
        super().__init__()
        assert split in ["train", "val", "test"], "split error value!"
        self.logger = logger
        if len(kwargs) > 0:
            print_log(f"Following arguments are not used: {kwargs}", self.logger)
        self.root = Path(path)
        self.npoints = npoints
        self.is_multiview = False
        self.split = split
        if len(classes) == 0:
            classes = ["airplane", "cabinet", "car", "chair", "lamp", "sofa", "table", "vessel"]
        self.classes = classes
        self.category_ids = [CAT2ID[cls] for cls in self.classes]
        self.pre_transform = Compose({"RandomSamplePoints": {"npoints": self.npoints, "apply_on": 1}})
        self.transform = Compose({**augment, "RandomSamplePoints": {"npoints": npoints_input, "apply_on": 0}})
        self.partial_ori_post_transform = Compose({"RandomSamplePoints": {"npoints": npoints_input, "apply_on": 0}})
        # load the filenames of required categories
        dataset_file = self.root / "PCN.json"
        with dataset_file.open("r") as f:
            category_lists = json.loads(f.read())
            self.category_lists = [cl for cl in category_lists if cl["taxonomy_id"] in self.category_ids]

        self.n_renderings = 8 if self.split == "train" else 1
        self.data_list = self._load_data(self.split, subset, self.n_renderings)

        print_log(
            f"PCN Dataset {split} loading finish. {len(self.data_list)} models "
            f"with {self.n_renderings} views available. Length {len(self)}. "
            f"Using pre-augmentation: {self.pre_transform}. "
            f"Using augmentation: {self.transform}.",
            logger=self.logger,
        )

    def __len__(self):
        return len(self.data_list)

    def _load_data(self, split, subset=None, n_renderings=1):
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
                        "partial_paths": [
                            self.root / split / "partial" / cl["taxonomy_id"] / model_id / f"{i:02d}.pcd"
                            for i in range(n_renderings)
                        ],
                        "gt_path": self.root / split / "complete" / cl["taxonomy_id"] / f"{model_id}.pcd",
                    }
                )
        return data_list

    def __getitem__(self, index):
        sample = self.data_list[index]
        partial_idx = np.random.randint(self.n_renderings) if self.split == "train" else 0
        partial_path = sample["partial_paths"][partial_idx]
        complete_path = sample["gt_path"]
        partial = read_point_cloud(partial_path)
        complete = read_point_cloud(complete_path)
        meta_data = {
            "label": sample["taxonomy_id"],
            "partial_id": f"{sample['model_id']}_{partial_idx:02d}",
            "complete_id": sample["model_id"],
        }
        transformed, _ = self.pre_transform([partial, complete])
        transformed, original = self.transform(transformed)
        [partial_original], _ = self.partial_ori_post_transform([original[0]])
        original = [partial_original, *original[1:]]
        return meta_data, transformed, original


class PCNCarKITTITestDataset(torch.utils.data.Dataset):
    """
    This dataset is only for KITTI Test purpose, contains all car models in PCN dataset, including train, test and val
    """

    def __init__(
        self,
        path="data/PCN",
        logger=None,
        **kwargs,
    ):
        self.logger = logger
        if len(kwargs) > 0:
            print_log(f"Following arguments are not used: {kwargs}", self.logger)
        self.root = Path(path)
        self.npoints = 16384
        self.classes = ["car"]
        self.category_ids = [CAT2ID[cls] for cls in self.classes]
        # load the filenames of required categories
        dataset_file = self.root / "PCN.json"
        with dataset_file.open("r") as f:
            category_lists = json.loads(f.read())
            self.category_lists = [cl for cl in category_lists if cl["taxonomy_id"] in self.category_ids]
        self.data_list = self._load_data()

        print_log(
            f"PCNCar Dataset for KITTI Test loading finish. {len(self.data_list)} models available.",
            logger=self.logger,
        )

    def __len__(self):
        return len(self.data_list)

    def _load_data(self):
        """Prepare file list for the dataset"""
        data_list = []
        splits = ["train", "val", "test"]
        for cl in self.category_lists:
            for split in splits:
                samples = cl[split]
                for model_id in samples:
                    data_list.append(
                        {
                            "taxonomy_id": cl["taxonomy_id"],
                            "model_id": model_id,
                            "gt_path": self.root / split / "complete" / cl["taxonomy_id"] / f"{model_id}.pcd",
                        }
                    )
        return data_list

    def __getitem__(self, index):
        sample = self.data_list[index]
        complete_path = sample["gt_path"]
        complete = read_point_cloud(complete_path)
        meta_data = {
            "label": sample["taxonomy_id"],
            "complete_id": sample["model_id"],
        }
        return meta_data, complete
