import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pointnet2_ops import pointnet2_utils


def pc_normalize(pc):
    if isinstance(pc, np.ndarray):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
    else:
        centroid = torch.mean(pc, axis=0)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc**2, axis=1)))
        pc = pc / m
    return pc


def knn(x, k, x_q=None):
    """
    knn for DGCNN-style input (B, C, N), compute nearest neighbors in the given point set
    distance x_q -> x: ||x_q - x||^2 = x_q^T x_q - 2x_q^T x + x^T x, nearest -> negative

    Args:
        x (Tensor): (B, C, N), original point set
        k (int): k
        x_q (Tensor): (B, C, N_q), query point set

    Returns:
        Tensor: (B, N_q, k) index of k-nearest neighbors for each point in query point set
    """
    if x_q is None:
        x_q = x
    inner = -2 * torch.matmul(x_q.transpose(2, 1), x)  # -2x_q^T x (B, N_q, N)
    xqxq = torch.sum(x_q**2, dim=1, keepdim=True)  # x_q^2 (B, 1, N_q)
    xx = torch.sum(x**2, dim=1, keepdim=True)  # x^2 (B, 1, N)
    pairwise_distance = -xqxq.transpose(2, 1) - inner - xx
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N_q, k)
    return idx


def knn_grouper(x, q, k):
    """
    Args:
        x: (B, C, N)
        q: (B, C, N_q)
        k: k in kNN

    Returns:
        idx: (B * N_q * k)
    """
    batch_size, _, num_points = x.shape
    idx = knn(x=x, k=k, x_q=q)  # (B, N_q, k)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points  # (B, 1, 1)
    idx = idx + idx_base
    idx = idx.view(-1)  # (B*N_q*k)
    return idx


def grouping_operation(x, idx, k):
    """
    Args:
        x: (B, C, N)
        idx: (B * N_q * k)
        k: k

    Returns:
        x_group: (B, C, N_q, k)
    """
    batch_size, num_channels, num_points = x.shape
    x = x.transpose(1, 2).contiguous().view(batch_size * num_points, -1)
    x_group = x[idx.flatten(), :].view(batch_size, -1, k, num_channels).permute(0, 3, 1, 2).contiguous()
    return x_group


def get_graph_feature_all_in_one(coor_q, x_q, coor, x, k, query="fts", sgm=False, relative_feature=True, idx=None):
    """
    Args:
        coor_q (Tensor): Query coordinates (B, 3, N_q)
        x_q (Tensor): Query features (B, C, N_q)
        coor (Tensor): Original coordinates (B, 3, N)
        x (Tensor): Original features (B, C, N)
        k (int): k for kNN
        query (str): kNN in feature space (fts) or geometry space (xyz)
        sgm (bool): use sorted gram matrix
        relative_feature (bool): final feature using torch.cat([feature - x_q, x_q]) or torch.cat([feature, x_q])
        idx: using given grouping index

    Returns:
        Tensor: edge features (B, 2*C, N_q, k)
        Tensor: group coordinates (B, 3, N_q, k)
        Tensor: group index (B*N_q*k)

    """
    batch_size, num_channels, num_points = x.shape  # (B, C, N)
    num_points_q = x_q.shape[2]
    if idx is None:
        with torch.no_grad():
            if query == "xyz":
                idx = knn_grouper(coor, coor_q, k)
            else:
                idx = knn_grouper(x, x_q, k)

    if coor is None:
        # sometimes we don't care about the coordinates
        coor_group = None
    else:
        coor_group = grouping_operation(coor, idx, k)  # (B, 3, N_q, k)

    feature = grouping_operation(x, idx, k)  # (B, C, N_q, k)
    x_q_k = x_q.view(batch_size, num_channels, num_points_q, 1).expand(-1, -1, -1, k)  # (B, C, N_q, k)

    if relative_feature:
        offset = feature - x_q_k
    else:
        offset = feature

    if sgm:
        x_q_k_gram = x_q_k.permute(0, 2, 3, 1) @ x_q_k.permute(0, 2, 1, 3)  # (B, N_q, k, k)
        x_q_k_sorted_gram = torch.sort(x_q_k_gram, dim=-1)[0]  # last dim is dim_feature
        offset_gram = offset.permute(0, 2, 3, 1) @ offset.permute(0, 2, 1, 3)  # (B, N_q, k, k)
        offset_sorted_gram = torch.sort(offset_gram, dim=-1)[0]  # last dim is dim_feature
        # (B, 2k, N_q, k)
        feature_group = torch.cat([offset_sorted_gram, x_q_k_sorted_gram], dim=-1).permute(0, 3, 1, 2).contiguous()
    else:
        feature_group = torch.cat([offset, x_q_k], dim=1)  # (B, 2C, N_q, k)

    return feature_group, coor_group, idx


def get_graph_feature(x: torch.Tensor, k, relative_feature=True):
    """
    Args:
        x (torch.Tensor): Features (batch_size, num_channels, num_points)
        sgm (bool): apply sorted gram matrix

    Returns:
        torch.Tensor: edge features = Concat([neighbor_x - x, x]) (B, C', N, k)
    """
    feature = get_graph_feature_all_in_one(x, x, x, x, k, relative_feature=relative_feature)[0]
    return feature


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    knn from pointnet2, find nsample points in xyz for each point in new_xyz
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    dist, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return dist, group_idx


def index_points(points, idx):
    """

    Input:
        points: input points data, (B, N, C)
        idx: sample index data, (B, *)
    Return:
        new_points: indexed points data, (B, *, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, ...]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # farthest = torch.zeros((B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, -1)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def fps_downsample(coor, x, num_group, mode="xyz", use_ext=False):
    # coor: (B, 3, N), x: (B, C, N) or (B, C, 3, N)
    coor_transpose = coor.transpose(1, 2).contiguous()
    if x is not None:
        is_vn = len(x.shape) == 4
        batch_size, npoints = x.shape[0], x.shape[-1]
        x = x.view(batch_size, -1, npoints)
        x_transpose = x.transpose(1, 2).contiguous()

    if mode == "xyz":
        if use_ext:
            fps_idx = pointnet2_utils.furthest_point_sample(coor_transpose, num_group)
        else:
            fps_idx = farthest_point_sample(coor_transpose, num_group).to(torch.int32)
    else:
        assert x is not None
        if use_ext:
            fps_idx = pointnet2_utils.furthest_point_sample(x_transpose, num_group)
        else:
            fps_idx = farthest_point_sample(x_transpose, num_group).to(torch.int32)

    if use_ext:
        new_coor = pointnet2_utils.gather_operation(coor, fps_idx)
    else:
        new_coor = index_points(coor_transpose, fps_idx).transpose(1, 2).contiguous()
    if x is not None:
        new_x = pointnet2_utils.gather_operation(x, fps_idx)
        if is_vn:
            new_x = new_x.reshape(batch_size, -1, 3, num_group)
        else:
            new_x = new_x.contiguous()
        return new_coor, new_x, fps_idx
    else:
        return new_coor, fps_idx


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = square_distance(new_xyz, xyz)
    # FIXME might have logical error, sorting group_idx makes no sense
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def three_nn(tgt, src):
    """
    Find the three nearest neighbors of tgt in src

    Args:
        tgt: (B, N, 3)
        src: (B, M, 3)

    Return:
        dist: (B, N, 3) l2 distance to the three nearest neighbors
        idx: (B, N, 3) index of the three nearest neighbors
    """
    dist, idx = knn_point(3, src, tgt)
    return dist, idx


def three_interpolate(features, idx, weight):
    """
    Performs weight linear interpolation on 3 features

    Args:
        features: (B, C, N)
        idx: (B, M, 3)
        weight: (B, M, 3)

    Returns:
        interpolated_features: (B, C, M)
    """
    B, M, _ = weight.shape
    features = features.transpose(1, 2)  # (B, N, C)
    # (B, M, 3, C) * (B, M, 3, 1) -> (B, M, 3, C), then sum at dim=2 -> (B, M, C)
    interpolated_features = torch.sum(index_points(features, idx) * weight.view(B, M, 3, 1), dim=2)
    interpolated_features = interpolated_features.transpose(1, 2).contiguous()  # (B, C, M)
    return interpolated_features


class Linear_ResBlock(nn.Module):
    def __init__(self, input_size=1024, output_size=256):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, output_size),
        )
        self.conv_res = nn.Linear(input_size, output_size)

    def forward(self, feature):
        return self.linear_block(feature) + self.conv_res(feature)
