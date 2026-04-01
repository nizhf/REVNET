import torch
import torch.nn as nn
import torch.nn.functional as F
from extensions.chamfer3D import dist_chamfer_3D
from extensions.PyTorchEMD.emd import earth_mover_distance
from components.common_utils import knn_point, index_points, square_distance

from .build import LOSS


@LOSS.register_module()
class EMDLoss(nn.Module):
    def __init__(self, transpose=False):
        super().__init__()
        self.transpose = transpose

    def forward(self, pred, gt):
        return earth_mover_distance(pred, gt, transpose=self.transpose).mean()


chamfer_distance_ext = dist_chamfer_3D.chamfer_3DDist()


@LOSS.register_module()
class ChamferDistanceLoss(nn.Module):
    def __init__(self, mode="cd_p", single_directional=False):
        super().__init__()
        self.mode = mode
        self.single_directional = single_directional

    def forward(self, pred, gt):
        cd_p2g, cd_g2p, _, _ = chamfer_distance_ext(pred, gt)
        if self.single_directional:
            return cd_p2g.mean(), cd_g2p.mean()
        else:
            cd_p = (torch.sqrt(cd_p2g).mean(dim=1) + torch.sqrt(cd_g2p).mean(dim=1)) / 2
            cd_t = cd_p2g.mean(dim=1) + cd_g2p.mean(dim=1)
            if self.mode == "cd_p":
                return cd_p.mean()
            elif self.mode == "cd_t":
                return cd_t.mean()
            else:
                return cd_p.mean(), cd_t.mean()


def chamfer_one_direction_any_dim_chunked(src, dst, chunk_size=512):
    B, N, C = src.shape
    _, M, _ = dst.shape

    min_dist = torch.full((B, N), float("inf"), device=src.device, dtype=src.dtype)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        dst_chunk = dst[:, start:end, :]  # [B, m, C]
        # [B, N, m]
        dist_chunk = square_distance(src, dst_chunk)
        # [B, N]
        min_dist_chunk, _ = torch.min(dist_chunk, dim=-1)
        min_dist = torch.minimum(min_dist, min_dist_chunk)

    return min_dist  # [B, N]


@LOSS.register_module()
class ChamferDistanceAnyDim(nn.Module):
    def __init__(self, mode="cd_p", single_directional=False, chunk_size=512):
        super().__init__()
        self.mode = mode
        self.single_directional = single_directional
        self.chunk_size = chunk_size

    def forward(self, pred, gt):
        """
        pred, gt: [B, Np, C], [B, Ng, C]
        """
        # pred -> gt
        cd_p2g = chamfer_one_direction_any_dim_chunked(pred, gt, self.chunk_size).clamp_min(1e-9)  # [B, Np]
        # gt -> pred
        cd_g2p = chamfer_one_direction_any_dim_chunked(gt, pred, self.chunk_size).clamp_min(1e-9)  # [B, Ng]

        if self.single_directional:
            return cd_p2g.mean(), cd_g2p.mean()
        else:
            cd_p = (torch.sqrt(cd_p2g).mean(dim=1) + torch.sqrt(cd_g2p).mean(dim=1)) / 2
            cd_t = cd_p2g.mean(dim=1) + cd_g2p.mean(dim=1)
            if self.mode == "cd_p":
                return cd_p.mean()
            elif self.mode == "cd_t":
                return cd_t.mean()
            else:
                return cd_p.mean(), cd_t.mean()
