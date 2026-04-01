from typing import Any
import numpy as np
import scipy.spatial.transform
import torch
from typing import List
from pointnet2_ops import pointnet2_utils


class Compose:
    def __init__(self, transforms: dict):
        self.transforms = []
        for name, param in transforms.items():
            self.transforms.append(eval(name)(**param))

    def __call__(self, data: List[torch.Tensor], **kwargs):
        original = [torch.clone(d) for d in data]
        for t in self.transforms:
            data = t(data, **kwargs)
        return data, original

    def __repr__(self) -> str:
        format_string = type(self).__name__ + "("
        for t in self.transforms:
            format_string += f"{t}, "
        format_string += ")"
        return format_string


class Augmentation:
    def __init__(self, apply_on="all", batch_process=False):
        if isinstance(apply_on, int):
            apply_on = [apply_on]
        self.apply_on = apply_on  # normally, 0 for partial, 1 for complete, 2 for missing (if available), "all" for all
        self.batch_process = batch_process

    def __call__(self, data: List[torch.Tensor], **kwargs):
        # batch process
        if len(data[0].shape) == 3:
            if len(data) > 1:
                assert len(data[0].shape) == len(data[1].shape)
            if self.batch_process:
                augmented = self.apply(data, **kwargs)
            else:
                bs = data[0].shape[0]
                augmented = [torch.empty_like(d) for d in data]
                for i in range(bs):
                    single_batch_result = self([d[i, ...] for d in data], **kwargs)
                    for i_d in range(len(data)):
                        augmented[i_d][i, ...] = single_batch_result[i_d]
            return augmented
        else:
            augmented = self.apply(data, **kwargs)
            return augmented

    def apply(self, **kwargs):
        raise NotImplementedError()


class RandomSamplePoints(Augmentation):
    def __init__(self, npoints=2048, mode="zero", apply_on=[0]):
        super().__init__(apply_on)
        self.npoints = npoints
        self.mode = mode

    def apply(self, data: List[torch.Tensor], **kwargs):
        # (N, 3)
        augmented = []
        if self.apply_on == "all":
            apply_on = np.arange(len(data))
        else:
            apply_on = self.apply_on
        # only apply on the selected data
        for i_pcd, pcd in enumerate(data):
            if i_pcd in apply_on:
                # upsample
                if pcd.shape[0] < self.npoints:
                    # non-existing points as zero
                    if self.mode == "zero":
                        redundant = torch.zeros((self.npoints - pcd.shape[0], pcd.shape[1]))
                    elif self.mode == "first":
                        redundant = pcd[0, :].repeat(self.npoints - pcd.shape[0], 1)
                    else:
                        choice = np.random.choice(pcd.shape[0], self.npoints - pcd.shape[0], replace=True)
                        redundant = pcd[choice, :]
                    new_pcd = torch.cat([pcd, redundant], dim=0)
                # same
                elif pcd.shape[0] == self.npoints:
                    new_pcd = pcd
                # downsample
                else:
                    choice = np.random.choice(pcd.shape[0], self.npoints, replace=False)
                    new_pcd = pcd[choice, :]
                augmented.append(new_pcd)
            else:
                augmented.append(pcd)
        return augmented

    def __repr__(self) -> str:
        format_string = f"{type(self).__name__} with npoints={self.npoints} and upsample mode={self.mode}"
        return format_string


class PointShuffle(Augmentation):
    def __init__(self, apply_on=[0]):
        super().__init__(apply_on)

    def apply(self, data: List[torch.Tensor], **kwargs):
        # (N, 3)
        augmented = []
        if self.apply_on == "all":
            apply_on = np.arange(len(data))
        else:
            apply_on = self.apply_on
        # only apply on the selected data
        for i_pcd, pcd in enumerate(data):
            if i_pcd in apply_on:
                idxes = np.arange(pcd.shape[0])
                np.random.shuffle(idxes)
                new_pcd = pcd[idxes, :]
                augmented.append(new_pcd)
            else:
                augmented.append(pcd)
        return augmented

    def __repr__(self) -> str:
        format_string = f"{type(self).__name__}"
        return format_string


class RandomRotation(Augmentation):
    def __init__(self, axes="y", apply_on="all"):
        super().__init__(apply_on)
        self.axes = axes

    def apply(self, data: List[torch.Tensor], **kwargs):
        # (N, 3)
        augmented = []
        if self.apply_on == "all":
            apply_on = np.arange(len(data))
        else:
            apply_on = self.apply_on

        if len(self.axes) == 1:
            rotation_angle = np.random.uniform(0, 1) * 2 * np.pi
        else:
            rotation_angle = np.random.uniform(0, 1, (len(self.axes))) * 2 * np.pi
        rotation_matrix = scipy.spatial.transform.Rotation.from_euler(self.axes, rotation_angle).as_matrix()
        rotation_matrix = torch.from_numpy(rotation_matrix.astype(np.float32)).to(data[0].device)

        # only apply on the selected data
        for i_pcd, pcd in enumerate(data):
            if i_pcd in apply_on:
                new_pcd = pcd @ rotation_matrix
                augmented.append(new_pcd)
            else:
                augmented.append(pcd)
        return augmented

    def __repr__(self) -> str:
        format_string = f"{type(self).__name__} with axes={self.axes}"
        return format_string


class RandomTranslation(Augmentation):
    def __init__(self, translation_range=0.5, apply_on="all"):
        super().__init__(apply_on)
        self.translation_range = translation_range

    def apply(self, data: List[torch.Tensor], **kwargs):
        # (N, 3)
        augmented = []
        if self.apply_on == "all":
            apply_on = np.arange(len(data))
        else:
            apply_on = self.apply_on

        translation = np.random.uniform(-self.translation_range, self.translation_range, (3))
        translation = torch.from_numpy(translation.astype(np.float32)).to(data[0].device)

        # only apply on the selected data
        for i_pcd, pcd in enumerate(data):
            if i_pcd in apply_on:
                new_pcd = pcd + translation
                augmented.append(new_pcd)
            else:
                augmented.append(pcd)
        return augmented

    def __repr__(self) -> str:
        format_string = f"{type(self).__name__} with translation_range={self.translation_range}"
        return format_string


class RandomScaling(Augmentation):
    def __init__(self, low=0.5, high=1.5, apply_on="all"):
        super().__init__(apply_on)
        self.low = low
        self.high = high

    def apply(self, data: List[torch.Tensor], **kwargs):
        # (N, 3)
        augmented = []
        if self.apply_on == "all":
            apply_on = np.arange(len(data))
        else:
            apply_on = self.apply_on

        scale = np.random.uniform(low=self.low, high=self.high)

        # only apply on the selected data
        for i_pcd, pcd in enumerate(data):
            if i_pcd in apply_on:
                new_pcd = pcd * scale
                augmented.append(new_pcd)
            else:
                augmented.append(pcd)
        return augmented

    def __repr__(self) -> str:
        format_string = f"{type(self).__name__} with low={self.low} high={self.high}"
        return format_string


class RandomJittering(Augmentation):
    def __init__(self, std=0.01, maximum=0.001, apply_on="all"):
        super().__init__(apply_on)
        self.std = std
        self.maximum = torch.tensor(maximum)

    def apply(self, data: List[torch.Tensor], **kwargs):
        # (N, 3)
        augmented = []
        if self.apply_on == "all":
            apply_on = np.arange(len(data))
        else:
            apply_on = self.apply_on
        # only apply on the selected data
        for i_pcd, pcd in enumerate(data):
            if i_pcd in apply_on:
                noise = torch.randn_like(pcd) * self.std
                noise = torch.minimum(torch.maximum(noise, -self.maximum), self.maximum)
                new_pcd = pcd + noise
                augmented.append(new_pcd)
            else:
                augmented.append(pcd)
        return augmented

    def __repr__(self) -> str:
        format_string = f"{type(self).__name__} with std={self.std} maximum={self.maximum}"
        return format_string


def discard_points_sample(pcd, n_discard):
    # pc: (N, C)
    n_points = pcd.shape[0]
    n_keep = n_points - n_discard
    idx_keep = torch.multinomial(torch.ones(n_points), n_keep)
    pcd = pcd[idx_keep, :]
    return pcd


def discard_points_batch(pcds, n_discard):
    # pcs: (B, N, C)
    bs, n_points, _ = pcds.shape
    n_keep = n_points - n_discard
    idx_keep = torch.multinomial(torch.ones(bs, n_points), n_keep)
    idx_batch = torch.arange(0, bs).view(-1, 1)
    pcds = pcds[idx_batch, idx_keep, :]
    return pcds


class RandomPointDropout(Augmentation):
    r"""
    Randomly drop some points

    Args:
        max_dropout_ratio (float, optional): maximum ratio of points to be droppped. Defaults to 0.5.
        min_num_points (int, optional): minimum number of points to be kept. Defaults to 50.
        mode (str, optional): first = set dropped points coordinates to the first point, discard = remove points. Defaults to "first".
        apply_on (str, optional): all data or selected indexes as a list. Defaults to "all".
    """

    def __init__(self, max_dropout_ratio=0.5, min_num_points=50, mode="first", apply_on="all"):
        super().__init__(apply_on, True)
        self.max_dropout_ratio = max_dropout_ratio
        self.min_num_points = min_num_points
        self.mode = mode

    def apply(self, data: List[torch.Tensor], **kwargs):
        # (N, 3)
        augmented = []
        if self.apply_on == "all":
            apply_on = np.arange(len(data))
        else:
            apply_on = self.apply_on
        # only apply on the selected data
        for i_pcd, pcd in enumerate(data):
            if i_pcd in apply_on:
                new_pcd = self._apply_random_dropout(pcd)
                augmented.append(new_pcd)
            else:
                augmented.append(pcd)
        return augmented

    def _apply_random_dropout(self, pcds: torch.Tensor):
        if self.mode == "first":
            if len(pcds.shape) == 2:
                pcds = self._dropout_mode_first(pcds)
            else:
                bs = pcds.shape[0]
                for b in range(bs):
                    pcds[b, :, :] = self._dropout_mode_first(pcds[b, :, :])
        elif self.mode == "zero":
            if len(pcds.shape) == 2:
                pcds = self._dropout_mode_zero(pcds)
            else:
                bs = pcds.shape[0]
                for b in range(bs):
                    pcds[b, :, :] = self._dropout_mode_zero(pcds[b, :, :])
        else:
            if len(pcds.shape) == 2:
                # (N, 3)
                n_dropout = max(int(np.random.uniform(0, self.max_dropout_ratio) * pcds.shape[0]), self.min_num_points)
                pcds = discard_points_sample(pcds, n_dropout)
            else:
                # (B, N, 3)
                n_dropout = max(int(np.random.uniform(0, self.max_dropout_ratio) * pcds.shape[1]), self.min_num_points)
                pcds = discard_points_batch(pcds, n_dropout)
        return pcds

    def _dropout_mode_first(self, pcd):
        dropout_ratio = np.random.uniform(0, self.max_dropout_ratio)
        drop_idxes = torch.where(torch.rand(pcd.shape[0]) <= dropout_ratio)[0]
        # Too many points are dropped, keep at least min_num_points
        if (pcd.shape[0] - len(drop_idxes)) < self.min_num_points:
            num_dropout = pcd.shape[0] - self.min_num_points
            drop_idxes = torch.randperm(pcd.shape[0])[:num_dropout]
        if len(drop_idxes) > 0:
            pcd[drop_idxes, :] = pcd[0, :].clone()  # set to the first point
        return pcd

    def _dropout_mode_zero(self, pcd):
        dropout_ratio = np.random.uniform(0, self.max_dropout_ratio)
        drop_idxes = torch.where(torch.rand(pcd.shape[0]) <= dropout_ratio)[0]
        # Too many points are dropped, keep at least min_num_points
        if (pcd.shape[0] - len(drop_idxes)) < self.min_num_points:
            num_dropout = pcd.shape[0] - self.min_num_points
            drop_idxes = torch.randperm(pcd.shape[0])[:num_dropout]
        if len(drop_idxes) > 0:
            pcd[drop_idxes, :] = 0  # set to zero
        return pcd

    def __repr__(self) -> str:
        format_string = f"{type(self).__name__} with max_dropout_ratio={self.max_dropout_ratio} mode={self.mode}"
        return format_string


class RandomMirrorPoints(Augmentation):
    def __init__(self, apply_on="all"):
        super().__init__(apply_on)

    def _get_transformation_matrix(self):
        rnd_value = np.random.uniform()
        trfm_mat = torch.eye(3)
        flip_x = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
        flip_z = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
        if rnd_value <= 0.25:
            trfm_mat = flip_x @ trfm_mat
            trfm_mat = flip_z @ trfm_mat
        elif rnd_value <= 0.5:
            trfm_mat = flip_x @ trfm_mat
        elif rnd_value <= 0.75:
            trfm_mat = flip_z @ trfm_mat
        return trfm_mat

    def apply(self, data: List[torch.Tensor], **kwargs):
        # (N, 3)
        augmented = []
        if self.apply_on == "all":
            apply_on = np.arange(len(data))
        else:
            apply_on = self.apply_on
        # only apply on the selected data
        for i_pcd, pcd in enumerate(data):
            transformation_matrix = self._get_transformation_matrix()
            if i_pcd in apply_on:
                new_pcd = pcd @ transformation_matrix
                augmented.append(new_pcd)
            else:
                augmented.append(pcd)
        return augmented

    def __repr__(self) -> str:
        format_string = f"{type(self).__name__}"
        return format_string


class RandomCropping(Augmentation):
    r"""
    RandomCropping

    Args:
        max_cropping_ratio (float, optional): maximum ratio of cropped region in each axis. Defaults to 0.5.
        mode (str, optional): first = set dropped points coordinates to the first point, discard = remove points. Defaults to "first".
        apply_on (str, optional): options [both, partial, completes]. Defaults to "both".
    """

    def __init__(self, max_cropping_ratio=0.5, mode="first", min_num_point=512, apply_on="all"):
        super().__init__(apply_on, True)
        self.max_cropping_ratio = max_cropping_ratio
        self.mode = mode
        self.min_num_point = min_num_point

    def apply(self, data: List[torch.Tensor], **kwargs):
        # (N, 3)
        augmented = []
        if self.apply_on == "all":
            apply_on = np.arange(len(data))
        else:
            apply_on = self.apply_on
        # only apply on the selected data
        for i_pcd, pcd in enumerate(data):
            if i_pcd in apply_on:
                new_pcd = self._apply_random_cropping(pcd)
                augmented.append(new_pcd)
            else:
                augmented.append(pcd)
        return augmented

    def _apply_random_cropping(self, pcds: torch.Tensor):
        if self.mode == "first":
            if len(pcds.shape) == 2:
                pcds = self._crop_mode_first(pcds)
            else:
                bs = pcds.shape[0]
                for b in range(bs):
                    pcds[b, :, :] = self._crop_mode_first(pcds[b, :, :])
        else:
            if len(pcds.shape) == 2:
                keep = self._get_crop_idx(pcds)
                pcds = pcds[keep, :]
            else:
                # Crop, then discard points according to the minimum number of points in cropped pcs
                pcds = self._crop_mode_discard_batch(pcds)
        return pcds

    def _get_crop_idx(self, pcd):
        while True:
            crop_ratio = np.random.uniform(low=0.0, high=self.max_cropping_ratio, size=(1, 3))  # for each axis
            keep_ratio = 1 - crop_ratio
            ratio_min = np.random.uniform(low=0, high=crop_ratio)
            ratio_max = ratio_min + keep_ratio
            coord_min = pcd.min(dim=0)[0]
            coord_max = pcd.max(dim=0)[0]
            interval = coord_max - coord_min
            range_min = coord_min + interval * ratio_min
            range_max = coord_min + interval * ratio_max
            keep = torch.logical_and(torch.all(pcd >= range_min, dim=1), torch.all(pcd <= range_max, dim=1))
            if torch.sum(keep) >= self.min_num_point:
                break
            else:
                # print(f"Too less ({torch.sum(keep)}) valid points after cropping, redo.")
                pass
        return keep

    def _crop_mode_first(self, pcd):
        keep = self._get_crop_idx(pcd)
        not_keep = torch.logical_not(keep)
        coord_repeat = pcd[keep, :][0]
        pc_cropped = pcd.clone()
        pc_cropped[not_keep, :] = coord_repeat
        return pc_cropped

    def _crop_mode_discard_batch(self, pcds):
        bs = pcds.shape[0]
        pcs_cropped = []
        for b in range(bs):
            keep = self._get_crop_idx(pcds[b, :, :])
            pcs_cropped.append(pcds[b, keep, :])
        n_points_remaining = [pc.shape[0] for pc in pcs_cropped]
        min_n_points = np.min(n_points_remaining)
        n_discard = [n - min_n_points for n in n_points_remaining]
        pcs_cropped = [discard_points_sample(pc, n) for pc, n in zip(pcs_cropped, n_discard)]
        pcs_cropped = torch.stack(pcs_cropped)
        return pcs_cropped

    def __repr__(self) -> str:
        format_string = f"{type(self).__name__} with max_cropping_ratio={self.max_cropping_ratio} mode={self.mode}"
        return format_string


# Adapted from https://github.com/yuxumin/PoinTr/blob/master/datasets/data_transforms.py
class NormalizeObjectPose(Augmentation):
    def __init__(self, apply_on="all"):
        super().__init__(apply_on)

    def apply(self, data: List[torch.Tensor], bbox):
        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        center = torch.from_numpy(center).view(1, 3).to(torch.float32)
        rotation = torch.from_numpy(rotation).to(torch.float32)

        augmented = []
        if self.apply_on == "all":
            apply_on = np.arange(len(data))
        else:
            apply_on = self.apply_on
        # only apply on the selected data
        for i_pcd, pcd in enumerate(data):
            if i_pcd in apply_on:
                new_pcd = torch.matmul(pcd - center, rotation) / scale
                new_pcd = torch.matmul(new_pcd, torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32))
                augmented.append(new_pcd)
            else:
                augmented.append(pcd)
        return augmented

    def __repr__(self) -> str:
        format_string = f"{type(self).__name__} for KITTI dataset"
        return format_string
