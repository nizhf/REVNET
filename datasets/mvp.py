import h5py
import numpy as np
import torch
from utils.logger import print_log
from utils.augment import Compose
from .build import DATASETS


@DATASETS.register_module()
class MVPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path="data/MVP",
        split="train",
        npoints=8192,
        novel_input=True,
        novel_input_only=False,
        augment={},
        classes=[],
        subset=None,
        logger=None,
    ):
        super().__init__()
        self.logger = logger
        self.input_path = f"{path}/mvp_{split}_input.h5"
        self.gt_path = f"{path}/mvp_{split}_gt_{npoints}pts.h5"
        self.gt_missing_path = f"{path}/mvp_{split}_gt_missing.h5"
        self.npoints = npoints
        self.is_multiview = False
        self.classes = [int(cls) for cls in classes]
        self.transform = Compose(augment)

        input_file = h5py.File(self.input_path, "r")
        self.input_data = np.array(input_file["incomplete_pcds"][:])
        self.labels = np.array(input_file["labels"][:])
        self.novel_input_data = np.array(input_file["novel_incomplete_pcds"][:])
        self.novel_labels = np.array(input_file["novel_labels"][:])
        input_file.close()

        gt_file = h5py.File(self.gt_path, "r")
        self.gt_data = np.array(gt_file["complete_pcds"][:])
        self.novel_gt_data = np.array(gt_file["novel_complete_pcds"][:])
        gt_file.close()


        if novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif novel_input:
            self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)
        self.labels = self.labels.astype(int)
        # only keep needed classes
        if len(self.classes) > 0:
            keep_idxes = []
            for idx in range(len(self.labels)):
                if self.labels[idx] in self.classes:
                    keep_idxes.append(idx)
            keep_idxes = np.array(keep_idxes)
        else:
            keep_idxes = np.arange(self.input_data.shape[0])

        if subset is not None:
            assert subset <= 1 and subset > 0
            num_keep = int(len(keep_idxes) * subset)
            self.valid_idxes = np.random.choice(keep_idxes, num_keep, replace=False)
        else:
            self.valid_idxes = keep_idxes

        self.len = len(self.valid_idxes)

        print_log(
            f"MVP Dataset {split} loading finish. {len(self.gt_data)} models "
            f"with {len(self.input_data) / len(self.gt_data)} views available. "
            f"Length {self.len}, input_data {self.input_data.shape}, gt_data {self.gt_data.shape}, "
            f"Using augmentation: {self.transform}",
            logger=self.logger,
        )

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = self.valid_idxes[index]
        partial = torch.from_numpy(self.input_data[index])
        complete = torch.from_numpy(self.gt_data[index // 26])
        meta_data = {
            "label": self.labels[index],
            "partial_id": index,
            "complete_id": index // 26,
        }
        transformed, original = self.transform([partial, complete])
        return meta_data, transformed, original


class MVPCarKITTITestDataset(torch.utils.data.Dataset):
    """
    This dataset is only for KITTI Test purpose, contains all car models in MVP dataset, including train, test and val
    """

    def __init__(
        self,
        path="data/MVP",
        npoints=8192,
        logger=None,
        **kwargs,
    ):
        super().__init__()
        self.logger = logger
        self.npoints = npoints

        train_gt_path = f"{path}/mvp_train_gt_{npoints}pts.h5"
        train_gt_file = h5py.File(train_gt_path, "r")
        train_labels = np.array(train_gt_file["labels"][:])
        train_gt_data = np.array(train_gt_file["complete_pcds"][:])
        train_gt_file.close()

        test_gt_path = f"{path}/mvp_test_gt_{npoints}pts.h5"
        test_gt_file = h5py.File(test_gt_path, "r")
        test_labels = np.array(test_gt_file["labels"][:])
        test_gt_data = np.array(test_gt_file["complete_pcds"][:])
        test_gt_file.close()

        labels = np.concatenate([train_labels, test_labels], axis=0)
        gt_data = np.concatenate([train_gt_data, test_gt_data], axis=0)

        # only keep needed classes (car=2)
        keep_idxes = labels == 2
        self.labels = labels[keep_idxes]
        self.gt_data = gt_data[keep_idxes]

        self.len = len(self.labels)

        print_log(
            f"MVPCar Dataset loading finish. Length {self.len}. "
            f"Labels {self.labels.shape}, gt_data {self.gt_data.shape}",
            logger=self.logger,
        )

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        complete = torch.from_numpy(self.gt_data[index])
        meta_data = {
            "label": self.labels[index],
            "complete_id": index,
        }
        return meta_data, complete
