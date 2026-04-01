from collections import defaultdict, deque

import numpy as np
import torch
import open3d as o3d

import torch.distributed as dist
from .dist_utils import is_dist_avail_and_initialized, get_world_size

from extensions.chamfer3D import dist_chamfer_3D
from extensions.PyTorchEMD.emd import earth_mover_distance


chamfer_distance_ext = dist_chamfer_3D.chamfer_3DDist()


def cd_fscore_emd(pred: torch.Tensor, gt: torch.Tensor, threshold=0.01, compute_emd=True):
    cd_p2g, cd_g2p, _, _ = chamfer_distance_ext(pred, gt)
    cd_t = torch.mean(cd_p2g, dim=1) + torch.mean(cd_g2p, dim=1)  # L2 distance
    cd_p = (torch.sqrt(cd_p2g).mean(dim=1) + torch.sqrt(cd_g2p).mean(dim=1)) / 2  # L1 distance
    # f1 = f_score(pred, gt, threshold)
    if isinstance(threshold, float):
        threshold = [threshold]
    f1_thres = []
    for thres in threshold:
        f1, _, _ = f_score_vrcnet(cd_p2g, cd_g2p, thres**2)
        f1_thres.append(f1)
    if compute_emd:
        emd = earth_mover_distance(pred, gt, transpose=False)
    else:
        emd = torch.tensor([0])
    return cd_p, cd_t, f1_thres, emd


def f_score(pred: torch.Tensor, gt: torch.Tensor, threshold=0.01):
    """
    References: https://github.com/lmb-freiburg/what3d/blob/master/util.py

    Args:
        pred (np.ndarray): (N1, 3)
        gt   (np.ndarray): (N2, 3)
        threshold   (float): a distance threshhold
    """
    if len(pred.shape) > 3:
        raise ValueError("Not support batch size > 1")
    elif len(pred.shape) == 3:
        bs = pred.shape[0]
        assert bs == gt.shape[0]
        fscore_list = []
        for i in range(bs):
            fs = f_score(pred[i], gt[i], threshold)
            fscore_list.append(fs)
        return torch.Tensor(fscore_list)
    else:
        pred = tensor_to_open3d_pcd(pred)
        gt = tensor_to_open3d_pcd(gt)
        dist1 = pred.compute_point_cloud_distance(gt)
        dist2 = gt.compute_point_cloud_distance(pred)
        precision_1 = np.mean((np.asarray(dist1) < threshold).astype(float))
        precision_2 = np.mean((np.asarray(dist2) < threshold).astype(float))
        fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2) if (precision_1 + precision_2) > 0 else 0.0
        return fscore


def tensor_to_open3d_pcd(tensor: torch.Tensor):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tensor.detach().cpu().numpy())
    return pcd


def f_score_vrcnet(dist1, dist2, threshold=0.0001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances
    # so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2

def per_pixel_acc(pred: torch.Tensor, gt: torch.Tensor, seg_classes, seg_label_to_class):
    # compute 
    pass

def mean_iou(pred: torch.Tensor, gt: torch.Tensor, seg_classes, seg_label_to_class):
    pass


def class_acc(pred: torch.Tensor, gt: torch.Tensor):
    pass


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def reset(self):
        self.deque.clear()
        self.count = 0
        self.total = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, metric_names: list, great_is_better=None):
        self.metric_names = metric_names
        self.n_metrics = len(self.metric_names)
        self.meters = {name: SmoothedValue() for name in metric_names}
        if great_is_better is None:
            self.great_is_better = {name: True for name in self.metric_names}
        elif isinstance(great_is_better, bool):
            self.great_is_better = {name: great_is_better for name in self.metric_names}
        elif isinstance(great_is_better, list):
            self.great_is_better = {name: item for name, item in zip(metric_names, great_is_better)}
        else:
            self.great_is_better = great_is_better

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def update(self, values, count=1):
        """
        Update metrics
        NOTE the input can have two options:
        1. values are mean, count=1 AND batch size is same for all batches
        2. values are sum, count=current batch size
        """
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, count)

    def val(self, metric_names=None):
        if metric_names is None:
            return {name: meter.value for name, meter in self.meters.items()}
        if not isinstance(metric_names, list):
            return self.meters[metric_names].value
        return {name: self.meters[name].value for name in metric_names}

    def count(self, metric_names=None):
        if metric_names is None:
            return {name: meter.count for name, meter in self.meters.items()}
        if not isinstance(metric_names, list):
            return self.meters[metric_names].count
        return {name: self.meters[name].count for name in metric_names}

    def avg(self, metric_names=None):
        # only consider global avg
        if metric_names is None:
            return {name: meter.global_avg for name, meter in self.meters.items()}
        if not isinstance(metric_names, list):
            return self.meters[metric_names].global_avg
        return {name: self.meters[name].global_avg for name in metric_names}

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def state_dict(self, metric_names=None):
        if metric_names is None:
            return {name: meter.global_avg for name, meter in self.meters.items()}
        if not isinstance(metric_names, list):
            metric_names = [metric_names]
        return {name: self.meters[name].global_avg for name in metric_names}

    def load_state_dict(self, state_dict: dict):
        for name, global_avg in state_dict.items():
            self.meters[name].total = global_avg
            self.meters[name].count = 1

    def __str__(self):
        out = ""
        for name, value in self.state_dict().items():
            out += f"{name}: {value}, "
        out = out[:-2]
        return out

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def better_than(self, to_compare, metric_names=None):
        if to_compare is None:
            return True
        if metric_names is None:
            return self.better_than(to_compare, self.metric_names)
        if isinstance(metric_names, list):
            better = []
            for name in metric_names:
                better.append(self.better_than(to_compare, name))
            return np.all(better)
        current = self.avg(metric_names)
        if np.isnan(current):
            return False
        great = current > to_compare.avg(metric_names)
        return great == self.great_is_better[metric_names]

    def add_meter(self, name, meter):
        self.meters[name] = meter
