import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops import pointnet2_utils
from .common_utils import knn, knn_grouper, fps_downsample, index_points

EPS = 1e-6


class VNGrouper(nn.Module):
    def __init__(
        self,
        *,
        grouper,
        grouper_cfg,
        k,
        verbose=False,
    ):
        super().__init__()
        self.grouper = grouper
        self.grouper_cfg = grouper_cfg
        self.k = k
        self.verbose = verbose

    def forward(self, coor_q: torch.Tensor, x_q: torch.Tensor, coor: torch.Tensor, x: torch.Tensor, mode=None):
        """
        Args:
            coor_q (torch.Tensor): (B, 3, N_q)
            x_q (torch.Tensor): (B, C, 3, N_q)
            coor (torch.Tensor): (B, 3, N)
            x (torch.Tensor): (B, C, 3, N)

        Returns:
            coor_group (B, 3, N_q, k), x_group (B, C, 3, N_q, k)
        """
        batch_size, num_channels, _, num_points = x.shape
        nsamples = x_q.shape[-1]
        # Grouping
        x = x.view(batch_size, num_channels * 3, num_points)  # (B, C*3, N)
        x_q = x_q.view(batch_size, num_channels * 3, nsamples)  # (B, C*3, N_q)
        with torch.no_grad():
            if self.grouper == "knn":
                mode = self.grouper_cfg["mode"] if mode is None else mode
                if mode == "xyz":
                    idx = knn(coor, self.k, coor_q)  # (B, N_q, k)
                else:
                    idx = knn(x, self.k, x_q)  # (B, N_q, k)
                coor_group = pointnet2_utils.grouping_operation(coor, idx.to(torch.int32))  # (B, 3, N_q, k)
                x_group = pointnet2_utils.grouping_operation(x, idx.to(torch.int32))  # (B, C*3, N_q, k)
            else:
                idx = pointnet2_utils.ball_query(
                    self.grouper_cfg["radius"], self.k, coor.transpose(1, 2), coor_q.transpose(1, 2)
                )
                coor_group = pointnet2_utils.gather_operation(coor, idx.to(torch.int32))
                x_group = pointnet2_utils.gather_operation(x, idx.to(torch.int32))

        x_group = x_group.reshape(batch_size, num_channels, 3, nsamples, self.k)
        return coor_group, x_group


def get_graph_feature_vn_all_in_one(coor_q, x_q, coor, x, k, query="fts", relative_feature=True, cross=False, idx=None):
    """
    Args:
        coor_q (Tensor): Query coordinates (B, 3, N_q)
        x_q (Tensor): Query features (B, C, 3, N_q)
        coor (Tensor): Original coordinates (B, 3, N)
        x (Tensor): Original features (B, C, 3, N)
        k (int): k for kNN
        query (str): kNN in feature space (fts) or geometry space (xyz)
        relative_feature (bool): final feature using torch.cat([feature - x_q, x_q]) or torch.cat([feature, x_q])
        cross (bool): additionally concat a cross product
        idx: using given grouping index

    Returns:
        Tensor: edge features (B, 2*C, 3, N_q, k)
        Tensor: group coordinates (B, 3, N_q, k)
        Tensor: group index (B*N_q*k)

    """
    batch_size, num_channels, _, num_points = x.shape  # (B, C, 3, N)
    num_points_q = x_q.shape[3]
    x = x.view(batch_size, num_channels * 3, num_points)  # (B, C*3, N)
    x_q = x_q.view(batch_size, num_channels * 3, num_points_q)  # (B, C*3, N_q)
    if idx is None:
        with torch.no_grad():
            if query == "xyz":
                idx = knn(coor, k, coor_q)  # (B, N_q, k)
            else:
                idx = knn(x, k, x_q)  # (B, N_q, k)
    if coor is None:
        # sometimes we don't care about the coordinates
        coor_group = None
    else:
        coor_group = pointnet2_utils.grouping_operation(coor, idx.to(torch.int32))  # (B, 3, N_q, k)

    feature = pointnet2_utils.grouping_operation(x, idx.to(torch.int32))  # (B, C*3, N_q, k)
    feature = feature.view(batch_size, num_channels, 3, num_points_q, k)  # (B, C, 3, N_q, k)

    x_q_k = x_q.view(batch_size, num_channels, 3, num_points_q, 1).expand(-1, -1, -1, -1, k)  # (B, C, 3, N_q, k)

    if relative_feature:
        offset = feature - x_q_k
    else:
        offset = feature

    if cross:
        feature_cross = torch.cross(feature, x_q_k, dim=2)  # (B, C, 3, N_q, k)
        feature_group = torch.cat([offset, x_q_k, feature_cross], dim=1)  # (B, 3C, 3, N_q, k)
    else:
        feature_group = torch.cat([offset, x_q_k], dim=1)  # (B, 2C, 3, N_q, k)

    return feature_group, coor_group, idx


class VNBlock(nn.Module):
    def __init__(
        self,
        layers,
        layers_type,
        data_dim,
        bias=False,
        share_nonlinearity=False,
        negative_slope=0.2,
        relu_in=False,
        pooling=None,
    ):
        super().__init__()
        sequence = []
        if isinstance(bias, (bool, str)):
            bias = [bias] * len(layers_type)
        if isinstance(share_nonlinearity, bool):
            share_nonlinearity = [share_nonlinearity] * len(layers_type)
        if isinstance(negative_slope, float):
            negative_slope = [negative_slope] * len(layers_type)
        if isinstance(relu_in, bool):
            relu_in = [relu_in] * len(layers_type)

        for i in range(len(layers) - 1):
            in_channels = layers[i]
            out_channels = layers[i + 1]
            match layers_type[i]:
                case "linear":
                    layer = VNLinear(in_channels, out_channels, bias[i])
                case "relu":
                    layer = VNLinearLeakyReLU(
                        in_channels,
                        out_channels,
                        dim=data_dim[i],
                        norm=None,
                        bias=bias[i],
                        share_nonlinearity=share_nonlinearity[i],
                        negative_slope=negative_slope[i],
                        relu_in=relu_in[i],
                    )
                case "bn_relu":
                    layer = VNLinearLeakyReLU(
                        in_channels,
                        out_channels,
                        dim=data_dim[i],
                        norm="batchnorm",
                        bias=bias[i],
                        share_nonlinearity=share_nonlinearity[i],
                        negative_slope=negative_slope[i],
                        relu_in=relu_in[i],
                    )
                case _ if "zca" in layers_type[i] or "vec" in layers_type[i]:
                    layer = VNLinearLeakyReLU(
                        in_channels,
                        out_channels,
                        dim=data_dim[i],
                        norm=layers_type[i],
                        bias=bias[i],
                        share_nonlinearity=share_nonlinearity[i],
                        negative_slope=negative_slope[i],
                        relu_in=relu_in[i],
                    )
                case _:
                    raise NotImplementedError(f"{layers_type[i]} unknown")
            sequence.append(layer)

        match pooling:
            case None:
                self.pool = None
            case "mean":
                self.pool = VNAvgPool(dim=-1)
            case "max":
                self.pool = VNMaxPool(layers[-1])
            case _:
                raise NotImplementedError(f"Pooling {pooling}")

        if self.pool is not None:
            sequence.append(self.pool)

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """
        x: (B, C, 3, N, ...)
        """
        x = self.model(x)
        return x


class VNDropout(nn.Module):
    def __init__(self, p, inplace=False) -> None:
        super().__init__()
        self.dropout = nn.Dropout1d(p, inplace)

    def forward(self, x):
        """
        Only support (B, C, 3, ...)
        Args:
            x: (B, C, 3, ...)
        """
        if len(x.shape) == 3:
            bs, c, d = x.shape
            assert d == 3
            x = self.dropout(x)
        elif len(x.shape) == 4:
            bs, c, d, n = x.shape
            assert d == 3
            x = x.transpose(2, -1).reshape(bs, -1, 3)  # (B, C*N, 3)
            x = self.dropout(x)
            x = x.view(bs, c, n, 3).transpose(2, -1).contiguous()
        elif len(x.shape) == 5:
            bs, c, d, n, m = x.shape
            assert d == 3
            x = x.transpose(2, -1).reshape(bs, -1, 3)  # (B, C*M*N, 3)
            x = self.dropout(x)
            x = x.view(bs, c, m, n, 3).transpose(2, -1).contiguous()
        else:
            raise NotImplementedError(f"VNDropout shape {x.shape} not supported.")
        return x


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.map_to_feat = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        if bias:
            if isinstance(bias, bool):
                self.bias_normalized = False
            elif bias == "normalize":
                self.bias_normalized = True
            else:
                raise NotImplementedError(bias)
            self.bias = nn.Parameter(torch.empty(out_channels, 3))
            self.map_to_bias = nn.Conv1d(in_channels, 3, kernel_size=1, bias=False)
            fan_in = in_channels
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: (B, C, 3, N, ...)
        """
        bs, in_channels = x.shape[:2]
        rest_shape = x.shape[2:]  # (3, N, ...)
        # flatten for Conv1D
        x_flat = x.view(bs, in_channels, -1)  # (B, C_in, T)
        x_feat = self.map_to_feat(x_flat)  # (B, C_out, T)
        x_out = x_feat.view(bs, -1, *rest_shape)  # (B, C_out, 3, N, ...)

        if self.bias is not None:
            bias_mat_flat = self.map_to_bias(x_flat)  # (B, 3 (C), T)
            bias_mat = bias_mat_flat.view(bs, 3, 3, -1)  # (B, 3 (C), 3, N*...)
            if self.bias_normalized:
                fnorm = torch.linalg.matrix_norm(bias_mat, ord="fro", dim=(1, 2), keepdim=True).clamp_min(EPS)
                bias_mat = bias_mat / fnorm

            bias = self.bias.view(1, 1, -1, 3)  # (1, 1, C_out, 3)
            bias_mat = bias_mat.permute(0, 3, 1, 2)  # (B, N*..., 3, 3)
            bias = torch.matmul(bias, bias_mat)  # (B, N*..., C_out, 3)
            bias = bias.permute(0, 2, 3, 1)  # (B, C_out, 3, N*...)
            bias = bias.view(bs, -1, *rest_shape)
            x_out = x_out + bias

        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        out_channels = 1 if share_nonlinearity else in_channels
        self.map_to_dir = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        """
        x: (B, C, 3, N, ...)
        """
        bs, in_channels = x.shape[:2]
        rest_shape = x.shape[2:]  # (3, N, ...)
        # flatten for Conv1D
        x_flat = x.view(bs, in_channels, -1)  # (B, C, T)
        d_flat = self.map_to_dir(x_flat)  # (B, Cout, T)
        d = d_flat.view(bs, -1, *rest_shape)  # (B, Cout, 3, N, ...)

        dotprod = (x * d).sum(dim=2, keepdim=True)  # <x,d>, (B, Cout, 1, ...)
        neg = torch.relu(-dotprod)  # max(0, -<x,d>)
        inv_d_norm_sq = (d * d).sum(dim=2, keepdim=True).clamp_min(EPS**2).reciprocal()
        alpha = 1.0 - self.negative_slope
        scale = alpha * neg * inv_d_norm_sq  # (B, Cout, 1, ...)
        x_out = x + scale * d

        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dim=5,
        norm="batchnorm",
        bias=False,
        share_nonlinearity=False,
        negative_slope=0.2,
        relu_in=False,
    ):
        super().__init__()
        self.dim = dim
        self.negative_slope = negative_slope

        self.linear = VNLinear(in_channels, out_channels, bias=bias)

        match norm:
            case None:
                self.norm = nn.Identity()
            case "bn" | "batchnorm" | "batch":
                self.norm = VNBatchNorm(out_channels, dim=self.dim)
            case "layer_zca" | "layer_vec":
                self.norm = VNLayerNorm(out_channels, affine=True, bias=False, mode=norm)
            case _:
                raise NotImplementedError(f"Norm {norm} not supported.")

        self.relu_in = relu_in
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity, negative_slope)

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        # Linear
        p = self.linear(x)
        # Apply Norm
        p = self.norm(p)
        # LeakyReLU
        x_out = self.leaky_relu(p)
        return x_out.contiguous()


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super().__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2).clamp_min(EPS)
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn

        return x


def vn_layer_zca(x, correction=0):
    # x: (B, C, 3, N, ...)
    bs, c, d, *npoints = x.shape
    reduce_dim = list(range(3, len(x.shape)))
    reduce_dim.append(1)
    x_centered = x - x.mean(dim=reduce_dim, keepdim=True)  # (B, C, 3, N, ...)
    x_centered = x_centered.view(bs, c, d, -1).permute(0, 3, 1, 2).reshape(bs, -1, d)  # (B, N*...*C, 3)
    var = x_centered.transpose(-1, -2) @ x_centered / (x_centered.shape[1] - correction)  # (B, 3, 3)

    eigval, eigvec = torch.linalg.eigh(var)
    eigval = torch.clamp_min(eigval, EPS)
    sigma_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(eigval))
    w_zca = eigvec @ sigma_inv_sqrt @ eigvec.transpose(1, 2)  # (B, 3, 3)
    x_zca = x_centered @ w_zca  # (B, N*...*C, 3)
    x_zca = x_zca.view(bs, -1, c, d).permute(0, 2, 3, 1).reshape(bs, c, d, *npoints)  # (B, C, 3, N, ...)
    return x_zca, eigvec.transpose(1, 2)


class VNLayerNorm(nn.Module):
    def __init__(self, num_channels, affine=True, bias=False, mode="zca"):
        super().__init__()
        self.mode = mode  # using covariance, or vector norm
        if "zca" in self.mode:  # layer_zca
            self.num_channels = num_channels
            self.affine = affine
            # elementwise affine
            if self.affine:
                self.weight = nn.Parameter(torch.empty(self.num_channels))
                if bias:
                    raise NotImplementedError("Bias in VNZCALayerNorm not supported")
                else:
                    self.register_parameter("bias", None)
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)
            self.reset_parameters()
        else:
            self.layernorm = nn.LayerNorm(num_channels, elementwise_affine=affine)

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, 3, N, ...)
        """
        bs, c, d, *npoints = x.shape
        if "zca" in self.mode:
            assert c == self.num_channels
            x, _ = vn_layer_zca(x, correction=0)  # same X shape
            if self.affine:
                num_extra_dims = len(npoints)
                w_shape = [1, c, 1] + [1] * num_extra_dims
                weight = self.weight.view(*w_shape)
                x = x * weight
        else:
            norm = torch.norm(x, dim=2).clamp_min(EPS)  # (B, C, N, ...)
            norm_ln = self.layernorm(norm.transpose(1, -1)).transpose(1, -1).reshape(bs, c, 1, *npoints)
            norm = norm.view(bs, c, 1, *npoints)
            x = x / norm * norm_ln
        return x.contiguous()


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdims=True)
        # FIXME see https://github.com/FlyingGiraffe/vnn/issues/11, the linear layer cannot be updated
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]], indexing="ij") + (idx,)
        x_max = x[index_tuple]
        return x_max


class VNAvgPool(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x, keepdim=False):
        x = torch.mean(x, dim=self.dim, keepdim=keepdim)
        return x


class VNStdTransform(nn.Module):
    def __init__(
        self,
        in_channels,
        dim=4,
        normalize_frame=False,
        norm=None,
        share_nonlinearity=False,
        negative_slope=0.2,
        relu_in=False,
    ):
        super().__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(
            in_channels,
            in_channels // 4,
            dim=dim,
            norm=norm,
            bias=False,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            relu_in=relu_in,
        )
        mid_channels = in_channels // 4
        out_channels = 2 if normalize_frame else 3

        self.vn_lin = nn.Conv1d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: point features of shape (B, C, 3, N, ...)
        """
        z0 = self.vn1(x)  # (B, C_mid, 3, N_samples, ...)
        bs, mid_channels = z0.shape[:2]
        rest_shape = z0.shape[2:]  # (3, N_samples, ...)

        # Flatten dimensions for Conv1D
        z_flat = z0.view(bs, mid_channels, -1)  # (B, C_mid, T)
        z0 = self.vn_lin(z_flat)  # (B, 2 or 3, T)
        z0 = z0.view(bs, -1, *rest_shape)  # (B, 2 or 3, 3, N_samples, ...)

        if not self.normalize_frame:
            return z0

        v1 = z0[:, 0]  # (B, 3, N_samples, ...)
        v2 = z0[:, 1]  # (B, 3, N_samples, ...)

        # u1 = v1 / ||v1||
        norm1_squared = (v1 * v1).sum(dim=1, keepdim=True).clamp_min(EPS**2)
        u1 = v1 * torch.rsqrt(norm1_squared)
        # projection + normalization --> u2
        v2 = v2 - (v2 * u1).sum(dim=1, keepdim=True) * u1
        norm2_squared = (v2 * v2).sum(dim=1, keepdim=True).clamp_min(EPS**2)
        u2 = v2 * torch.rsqrt(norm2_squared)
        # u3 = u1 × u2
        u3 = torch.cross(u1, u2, dim=1)
        z0 = torch.stack([u1, u2, u3], dim=1)  # (B, 3, 3, *S)

        # NOTE need transpose for rotation invariant feature
        # z0 = z0.transpose(1, 2)
        return z0


class VNStdFeature(nn.Module):
    def __init__(
        self,
        in_channels,
        dim=4,
        normalize_frame=False,
        norm=None,
        share_nonlinearity=False,
        negative_slope=0.2,
        relu_in=False,
    ):
        super().__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        self.transform = VNStdTransform(
            in_channels=in_channels,
            dim=dim,
            normalize_frame=normalize_frame,
            norm=norm,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            relu_in=relu_in,
        )

    def forward(self, x: torch.Tensor):
        """
        x: point features (B, C, 3, N, ...)
        """
        z0 = self.transform(x).transpose(1, 2)  # (B, 3, 3, N, ...)
        bs, in_channels = x.shape[:2]
        if self.dim == 3:
            # x: (B, C, 3), z0: (B, 3, 3)
            x_std = x @ z0
        elif self.dim > 3:
            # flatten other dims
            sample_shape = x.shape[3:]
            nsamples = int(torch.tensor(sample_shape).prod().item())
            x_f = x.permute(0, 3, 1, 2).reshape(bs * nsamples, in_channels, 3)  # (B*N*..., C, 3)
            z_f = z0.permute(0, 3, 1, 2).reshape(bs * nsamples, 3, 3)  # (B*N*..., 3, 3)
            y_f = x_f @ z_f  # (B*N*..., C, 3)
            x_std = y_f.view(bs, *sample_shape, in_channels, 3).permute(0, -2, -1, *range(1, 1 + len(sample_shape)))
        else:
            raise ValueError(f"Unsupported dim={self.dim}")
        return x_std.contiguous(), z0


class STNkd(nn.Module):
    def __init__(self, pooling, d=64):
        super().__init__()
        self.conv1 = VNLinearLeakyReLU(d, 64 // 3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 128 // 3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128 // 3, 1024 // 3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU(1024 // 3, 512 // 3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512 // 3, 256 // 3, dim=3, negative_slope=0.0)

        if pooling == "max":
            self.pool = VNMaxPool(1024 // 3)
        elif pooling == "mean":
            self.pool = VNAvgPool()

        self.fc3 = VNLinear(256 // 3, d)
        self.d = d

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class VNPointNet(nn.Module):
    def __init__(self, n_knn, global_feat=True, feature_transform=False, pooling="max", channel=3):
        super().__init__()
        self.n_knn = n_knn

        self.conv_pos = VNLinearLeakyReLU(3, 64 // 3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64 // 3, 64 // 3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64 // 3 * 2, 128 // 3, dim=4, negative_slope=0.0)

        self.conv3 = VNLinear(128 // 3, 1024 // 3)
        self.bn3 = VNBatchNorm(1024 // 3, dim=4)

        self.std_feature = VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=False, negative_slope=0.0)

        if pooling == "max":
            self.pool = VNMaxPool(64 // 3)
        elif pooling == "mean":
            self.pool = VNAvgPool()

        self.global_feat = global_feat
        self.feature_transform = feature_transform

        if self.feature_transform:
            self.fstn = STNkd(pooling, d=64 // 3)

    def forward(self, x):
        B, D, N = x.size()

        x = x.unsqueeze(1)
        feat = get_graph_feature_vn_all_in_one(x, x, x, x, self.n_knn, cross=True)
        x = self.conv_pos(feat)
        x = self.pool(x)

        x = self.conv1(x)

        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1, 1, 1, N)
            x = torch.cat((x, x_global), 1)

        pointfeat = x
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))

        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(B, -1, N)

        x = torch.max(x, -1, keepdim=False)[0]

        trans_feat = None
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class VNDGCNN(nn.Module):
    def __init__(self, out_channels=341, k=20, layers=[21, 21, 42, 84], pooling="max", normal_channel=False):
        super().__init__()
        self.n_knn = k

        self.conv1 = VNLinearLeakyReLU(2, layers[0])
        self.conv2 = VNLinearLeakyReLU(layers[0] * 2, layers[1])
        self.conv3 = VNLinearLeakyReLU(layers[1] * 2, layers[2])
        self.conv4 = VNLinearLeakyReLU(layers[2] * 2, layers[3])

        self.conv5 = VNLinearLeakyReLU(
            layers[0] + layers[1] + layers[2] + layers[3],
            out_channels,
            dim=4,
            share_nonlinearity=True,
        )

        self.std_feature = VNStdFeature(out_channels * 2, dim=4, normalize_frame=False)

        if pooling == "max":
            self.pool1 = VNMaxPool(layers[0])
            self.pool2 = VNMaxPool(layers[1])
            self.pool3 = VNMaxPool(layers[2])
            self.pool4 = VNMaxPool(layers[3])
        elif pooling == "mean":
            self.pool1 = VNAvgPool()
            self.pool2 = VNAvgPool()
            self.pool3 = VNAvgPool()
            self.pool4 = VNAvgPool()

    def forward(self, x):
        # x: (B, N, C)
        batch_size = x.size(0)
        x = x.transpose(1, 2).contiguous()
        x = x.unsqueeze(1)  # (B, 1, C, N)
        x = get_graph_feature_vn_all_in_one(None, x, None, x, k=self.n_knn, query="fts")
        x = self.conv1(x)
        x1 = self.pool1(x)

        x = get_graph_feature_vn_all_in_one(None, x1, None, x1, k=self.n_knn, query="fts")
        x = self.conv2(x)
        x2 = self.pool2(x)

        x = get_graph_feature_vn_all_in_one(None, x2, None, x2, k=self.n_knn, query="fts")
        x = self.conv3(x)
        x3 = self.pool3(x)

        x = get_graph_feature_vn_all_in_one(None, x3, None, x3, k=self.n_knn, query="fts")
        x = self.conv4(x)
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(batch_size, -1, num_points)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        trans_feat = None
        return x, trans_feat
