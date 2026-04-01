import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops import pointnet2_utils
from .vnn_layers import (
    VNGrouper,
    VNLinear,
    VNBlock,
    VNAvgPool,
    VNMaxPool,
    VNLayerNorm,
    VNLinearLeakyReLU,
    VNDropout,
    VNStdFeature,
    get_graph_feature_vn_all_in_one,
)
from .vnn_attn import VNMultiheadAttention
from .common_utils import fps_downsample
from components.vnn_attn import VNMultiheadAttention


class VNMetaGrouper(nn.Module):
    def __init__(
        self,
        *,
        num_anchors,
        layers=[16, 32, 64, 128, 128],
        blocks=[2, 3, 2],
        k=[16, 32, 16, 8],
        pooling="mean",
        downsample_mode="xyz",
        fps_ext=False,
        grouper="knn",
        grouper_cfg={"mode": "xyz"},
        invres_expansion=2,
        bias=False,
        norm="batchnorm",
        pos_enc_ff_cfg={"layers": [], "layers_type": ["relu"], "data_dim": [5]},
        vn_cfg={"share_nonlinearity": False, "negative_slope": 0.2, "relu_in": False},
        verbose=False,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.verbose = verbose

        # layer 0, local patch as encoder input feature
        self.k_init = k[0]
        if norm is None:
            type_mlp_in = "relu"
        elif norm in ("bn", "batchnorm", "batch"):
            type_mlp_in = "bn_relu"
        elif "layer" in norm or "vec" in norm:  # layer_zca, layer_vec
            type_mlp_in = norm
        else:
            raise NotImplementedError(f"MetaGrouper {norm} unknown")

        self.mlp_in = VNBlock(
            layers=[3, layers[0]],
            layers_type=[type_mlp_in],
            data_dim=[5],
            bias=bias,
            pooling=pooling,
            **vn_cfg,
        )

        encoder = []
        # encoder layers, Downsample->SetAbstraction->InvResMLP
        for i in range(len(blocks)):
            block = []
            mlp_pe = VNPositionEncodingSub(
                d_model=layers[i + 1],
                ff_xyz_cfg=copy.deepcopy(pos_enc_ff_cfg),
                diff_input=True,
                vn_cfg=vn_cfg,
            )
            for b in range(blocks[i]):
                if b == 0:
                    layer = VNSetAbstraction(
                        in_channels=layers[i],
                        out_channels=layers[i + 1],
                        layers=1,
                        nsamples=self.num_anchors * 2 ** (len(blocks) - 1 - i),
                        mlp_pe=mlp_pe,
                        downsample_mode=downsample_mode,
                        fps_ext=fps_ext,
                        grouper=grouper,
                        grouper_cfg=grouper_cfg,
                        grouper_k=k[i + 1],
                        norm=norm,
                        pooling=pooling,
                        use_residue=False,
                        vn_cfg=vn_cfg,
                        verbose=self.verbose,
                    )
                else:
                    layer = VNInvResMLP(
                        in_channels=layers[i + 1],
                        layers=2,
                        mlp_pe=mlp_pe,
                        expansion=invres_expansion,
                        grouper=grouper,
                        grouper_cfg=grouper_cfg,
                        grouper_k=k[i + 1],
                        norm=norm,
                        pooling=pooling,
                        use_residue=True,
                        vn_cfg=vn_cfg,
                        verbose=self.verbose,
                    )
                block.append(layer)
            block = nn.Sequential(*block)
            encoder.append(block)

        self.encoder = nn.Sequential(*encoder)

        self.mlp_final = VNLinearLeakyReLU(
            in_channels=sum(layers[:-1]),
            out_channels=layers[-1],
            dim=4,
            norm=norm,
            bias=bias,
            **vn_cfg,
        )

    def forward(self, coor: torch.Tensor, num_anchors=None):
        """
        coor: (B, N, 3) point cloud
        """
        if num_anchors is None:
            num_anchors = self.num_anchors
        if coor.shape[-1] == 3 and coor.shape[-2] != 3:
            coor = coor.transpose(1, 2).contiguous()  # (B, 3, N)
        bs = coor.shape[0]
        x = coor.unsqueeze(1)  # (B, 1, 3, N)

        # initial MLP
        x_group, _, _ = get_graph_feature_vn_all_in_one(coor, x, coor, x, self.k_init, "xyz", True, True)
        x = self.mlp_in(x_group)

        x_hierarchical = []
        x_hierarchical.append(x)

        # anchor encoder
        for i in range(len(self.encoder)):
            block_out = self.encoder[i]({"coor": coor, "x": x})
            coor, x, fps_idx = block_out["coor"], block_out["x"], block_out["fps_idx"]
            for j in range(len(x_hierarchical)):
                x_temp = x_hierarchical[j]
                x_temp = x_temp.view(bs, -1, x_temp.shape[-1])
                x_temp = pointnet2_utils.gather_operation(x_temp, fps_idx).reshape(bs, -1, 3, x.shape[-1])
                x_hierarchical[j] = x_temp

            x_hierarchical.append(x)

        # MLP for hierarchical data
        x = self.mlp_final(torch.cat(x_hierarchical, dim=1))

        return coor, x


class VNSetAbstraction(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        layers,
        nsamples,
        mlp_pe,
        downsample_mode="xyz",
        fps_ext=False,
        grouper="knn",
        grouper_cfg={"mode": "xyz"},
        grouper_k=16,
        bias=False,
        norm="batchnorm",
        pooling="mean",  # mean or max or attn
        use_residue=False,
        vn_cfg={"share_nonlinearity": False, "negative_slope": 0.2, "relu_in": False},
        verbose=False,
    ):
        super().__init__()
        self.nsamples = nsamples
        self.downsample_mode = downsample_mode
        self.fps_ext = fps_ext
        self.use_residue = use_residue
        self.verbose = verbose

        self.grouper = VNGrouper(
            grouper=grouper,
            grouper_cfg=grouper_cfg,
            k=grouper_k,
            verbose=self.verbose,
        )

        mid_channels = out_channels // 2
        layers_mlp_in = [in_channels] + [mid_channels] * (layers - 1) + [out_channels]
        if norm is None:
            types_mlp_in = ["relu"] * (len(layers_mlp_in) - 1)
        elif norm in ("bn", "batchnorm", "batch"):
            types_mlp_in = ["bn_relu"] * (len(layers_mlp_in) - 1)
        elif "zca" in norm or "vec" in norm:  # layer_zca, layer_vec
            types_mlp_in = [norm] * (len(layers_mlp_in) - 1)
        else:
            raise NotImplementedError(f"VNSetAbstraction {norm} unknown")

        if self.use_residue:
            self.skip_conv = (
                VNLinear(in_channels, out_channels, bias=bias) if in_channels != out_channels else nn.Identity()
            )

        self.mlp_in = VNBlock(
            layers=layers_mlp_in,
            layers_type=types_mlp_in,
            data_dim=[4] * len(types_mlp_in),
            bias=bias,
            **vn_cfg,
            pooling=None,
        )

        self.mlp_pe = mlp_pe  # unified mlp_pe for all MLP blocks at a stage

        match pooling:
            case "mean":
                self.pool = VNAvgPool(dim=-1)
            case "max":
                self.pool = VNMaxPool(in_channels=out_channels)
            case _:
                raise NotImplementedError(f"pooling {pooling}")

    def forward(self, dict_in: dict):
        """
        Args:
            coor: (B, 3, N)
            x : (B, C_in, 3, N)
        """
        coor, x = dict_in["coor"], dict_in["x"]
        batch_size, _, _, num_points = x.shape
        identity = x
        # in MLP
        x = self.mlp_in(x)
        num_channels = x.shape[1]
        # Subsampling, (B, 3, N_q), (B, C, 3, N_q)
        coor_q, x_q, fps_idx = fps_downsample(
            coor, x, num_group=self.nsamples, mode=self.downsample_mode, use_ext=self.fps_ext
        )
        # Grouping
        coor_group, x_group = self.grouper(coor_q, x_q, coor, x)
        coor_diff = coor_group - coor_q.view(batch_size, 3, self.nsamples, 1)
        coor_diff = coor_diff.view(batch_size, 1, 3, self.nsamples, -1)  # (B, 1, 3, N_q, k)
        # Position encoding
        pos_enc = self.mlp_pe(coor_diff)  # (B, C_out, 3, N_q, k)
        x_group = x_group + pos_enc
        # Pooling
        x = self.pool(x_group)
        # Residual path
        if self.use_residue:
            # identity = index_points(identity.view(batch_size, -1, num_points).transpose(1, 2), fps_idx).transpose(1, 2)
            identity = pointnet2_utils.gather_operation(identity.view(batch_size, -1, num_points), fps_idx)
            identity = identity.reshape(batch_size, num_channels, 3, num_points)
            identity = self.skip_conv(identity)
            x = x + identity

        out = {"coor": coor_q, "x": x, "fps_idx": fps_idx}
        return out


class VNInvResMLP(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        layers,
        mlp_pe: nn.Module,
        expansion=1,
        grouper="knn",
        grouper_cfg={"k": 16, "mode": "xyz"},
        grouper_k=16,
        bias=False,
        norm="batchnorm",
        pooling="mean",  # mean or max or attn
        use_residue=True,
        vn_cfg={"share_nonlinearity": False, "negative_slope": 0.2, "relu_in": False},
        verbose=False,
    ):
        super().__init__()
        self.use_residue = use_residue
        self.verbose = verbose

        self.grouper = VNGrouper(
            grouper=grouper,
            grouper_cfg=grouper_cfg,
            k=grouper_k,
            verbose=self.verbose,
        )

        mid_channels = in_channels * expansion
        out_channels = in_channels

        layers_mlp_in = [in_channels, in_channels]
        layers_mlp_update = [in_channels] + [mid_channels] * (layers - 1) + [out_channels]
        if norm is None:
            types_mlp_in = ["relu"] * (len(layers_mlp_in) - 1)
            types_mlp_update = ["relu"] * (len(layers_mlp_update) - 1)
        elif norm in ("bn", "batchnorm", "batch"):
            types_mlp_in = ["bn_relu"] * (len(layers_mlp_in) - 1)
            types_mlp_update = ["bn_relu"] * (len(layers_mlp_update) - 1)
        elif "zca" in norm or "vec" in norm:  # layer_zca, layer_vec
            types_mlp_in = [norm] * (len(layers_mlp_in) - 1)
            types_mlp_update = [norm] * (len(layers_mlp_update) - 1)
        else:
            raise NotImplementedError(f"VNInvResMLP {norm} unknown")

        self.mlp_in = VNBlock(
            layers=layers_mlp_in,
            layers_type=types_mlp_in,
            data_dim=[4] * len(types_mlp_in),
            bias=bias,
            **vn_cfg,
            pooling=None,
        )
        self.mlp_pe = mlp_pe  # unified mlp_pe for all MLP blocks at a stage

        match pooling:
            case "mean":
                self.pool = VNAvgPool(dim=-1)
            case "max":
                self.pool = VNMaxPool(in_channels=out_channels)
            case _:
                raise NotImplementedError(f"pooling {pooling}")

        self.mlp_update = VNBlock(
            layers=layers_mlp_update,
            layers_type=types_mlp_update,
            data_dim=[4] * len(types_mlp_update),
            bias=bias,
            **vn_cfg,
            pooling=None,
        )

    def forward(self, dict_in: dict):
        """
        Args:
            coor (torch.Tensor): (B, 3, N)
            x (torch.Tensor): (B, C_in, 3, N)
        """
        coor, x = dict_in["coor"], dict_in["x"]
        fps_idx = dict_in.get("fps_idx", None)

        batch_size, _, _, num_points = x.shape
        identity = x
        # in MLP
        x = self.mlp_in(x)
        # Grouping
        coor_group, x_group = self.grouper(coor, x, coor, x)
        coor_diff = coor_group - coor.view(batch_size, 3, num_points, 1)
        coor_diff = coor_diff.view(batch_size, 1, 3, num_points, -1)  # (B, 1, 3, N, k)
        # Position encoding
        pos_enc = self.mlp_pe(coor_diff)  # (B, C_out, 3, N_q, k)
        x_group = x_group + pos_enc
        # Pooling
        x = self.pool(x_group)
        # MLP
        x = self.mlp_update(x)
        # Residue
        if self.use_residue:
            x = x + identity

        out = {"coor": coor, "x": x, "fps_idx": fps_idx}
        return out


class VNCoarseDecoderRotInv(nn.Module):
    def __init__(
        self,
        *,
        up_dim,
        num_coarse,
        coarse_dim1,
        coarse_dim2,
        normalize_frame,
        norm,
        vn_cfg,
        verbose=False,
        rotvar_cfg=None,
    ):
        super().__init__()
        self.verbose = verbose
        self.normalize_frame = normalize_frame
        self.global_rotinv_transform = VNStdFeature(
            in_channels=up_dim, dim=3, normalize_frame=normalize_frame, norm=norm, **vn_cfg
        )
        self.coarse_mlp = nn.Sequential(
            nn.Linear(3 * up_dim, coarse_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(coarse_dim1, coarse_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(coarse_dim2, 3 * num_coarse),
        )
        if not self.normalize_frame:
            rotvar_cfg["layers"] = [up_dim, *rotvar_cfg["layers"], 3]
            self.equiv_transform = VNBlock(**rotvar_cfg, **vn_cfg)

    def forward(self, x_global_up):
        """
        Args:
            x_global_up: (B, C_up, 3)

        Returns:
            Point coordinates: (B, 3, N)
        """
        # transform global feature to rotation-invariant
        bs = x_global_up.shape[0]
        x_global_up_rotinv, transform_matrix = self.global_rotinv_transform(x_global_up)
        coarse_rotinv = self.coarse_mlp(x_global_up_rotinv.view(bs, -1)).view(bs, -1, 3)  # (B, N_coarse, 3)
        if self.normalize_frame:
            equiv_matrix = transform_matrix.transpose(1, 2)
        else:
            equiv_matrix = self.equiv_transform(x_global_up)  # (B, 3, 3)
        coarse_equiv = coarse_rotinv @ equiv_matrix
        coarse_equiv = coarse_equiv.transpose(1, -1).contiguous()
        return coarse_equiv


class VNPositionEncodingSub(nn.Module):
    def __init__(self, d_model, ff_xyz_cfg, diff_input, vn_cfg):
        super().__init__()
        self.d_model = d_model
        self.diff_input = diff_input
        ff_xyz_cfg["layers"] = [1, *ff_xyz_cfg["layers"], d_model]
        self.ff_xyz = VNBlock(**ff_xyz_cfg, **vn_cfg)

    def forward(self, coor_q, coor_k=None):
        """
        Args:
            coor_q: (B, 3, N_q) cross-attn, (B, 3, N_q, N_k) group self-attn, (B, 3, N_q, N_k) diff_input
            coor_k: (B, 3, N_k) cross-attn, (B, 3, N_q) group self-attn

        Returns:
            position encoding: (B, d_model, 3, N_q, N_k)
        """
        if self.diff_input:
            view_shape = list(coor_q.shape)
            if view_shape[1] != 1:
                view_shape.insert(1, 1)
            coor_diff = coor_q.view(view_shape)  # (bs, 1, 3, ...)
        else:
            if coor_k is None:
                coor_k = coor_q
            if len(coor_k.shape) == 3:
                bs, _, n_q = coor_q.shape
                n_k = coor_k.shape[-1]
                coor_diff = coor_q.view(bs, 1, 3, n_q, 1) - coor_k.view(bs, 1, 3, 1, n_k)
            elif len(coor_k.shape) == 4:
                bs, _, n_q, n_k = coor_k.shape
                coor_diff = coor_q.view(bs, 1, 3, n_q, -1) - coor_k.view(bs, 1, 3, n_q, n_k)
            else:
                raise NotImplementedError(f"Unsupported dim {len(coor_q.shape)}")
        x = self.ff_xyz(coor_diff)
        return x


class VNPositionEncodingKNN(nn.Module):
    def __init__(self, d_model, k, ff_cfg, pooling, vn_cfg):
        super().__init__()
        ff_cfg["layers"] = [3, *ff_cfg["layers"], d_model]
        self.k = k
        self.pos_ff = VNBlock(**ff_cfg, **vn_cfg)
        match pooling:
            case "mean":
                self.pool = VNAvgPool(dim=-1)
            case "max":
                self.pool = VNMaxPool(in_channels=d_model)
            case _:
                raise NotImplementedError(f"pooling {pooling}")

    def forward(self, coor_q, coor=None):
        """
        Args:
            coor_q: (B, 3, N_q)

        Returns:
            position encoding: (B, d_model, 3, N_a)
        """
        if len(coor_q.shape) == 3:
            x_q = coor_q.unsqueeze(1)
        if coor is None:
            coor = coor_q
        if len(coor.shape) == 3:
            x = coor.unsqueeze(1)
        x, _, _ = get_graph_feature_vn_all_in_one(coor_q, x_q, coor, x, self.k, "xyz", True, True)
        x = self.pos_ff(x)
        x = self.pool(x)
        return x


class VNSelfAttnBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model,
        in_dim,
        pos_enc_dim,
        nhead,
        dim_feedforward,
        bias_feedforward=False,
        use_qkproj=True,
        attn_mode="sub",
        head_mode="separate",
        fwd_mode="channel",
        norm_mode="layer_zca",
        norm_affine=True,
        dropout=0.0,
        norm_first=False,
        share_nonlinearity=False,
        negative_slope=0.2,
        relu_in=False,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        # SA Block
        self.self_attn = VNMultiheadAttention(
            d_model=d_model,
            nhead=nhead,
            qdim=in_dim,
            kdim=in_dim + pos_enc_dim,
            vdim=in_dim,
            use_qkproj=use_qkproj,
            use_vproj=True,
            attn_mode=attn_mode,
            head_mode=head_mode,
            fwd_mode=fwd_mode,
            negative_slope=negative_slope,
            verbose=self.verbose,
        )
        # feed-forward, l1 -> leakyrelu -> l2
        self.sa_ff = VNBlock(
            layers=[d_model, dim_feedforward, d_model],
            layers_type=["relu", "linear"],
            data_dim=[4, 4],
            bias=bias_feedforward,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            relu_in=relu_in,
            pooling=None,
        )
        self.norm_first = norm_first
        norm_dim = in_dim if self.norm_first else d_model
        if norm_mode is not None:
            self.sa_norm1 = VNLayerNorm(num_channels=norm_dim, affine=norm_affine, bias=False, mode=norm_mode)
            self.sa_norm2 = VNLayerNorm(num_channels=norm_dim, affine=norm_affine, bias=False, mode=norm_mode)
        else:
            self.sa_norm1 = nn.Identity()
            self.sa_norm2 = nn.Identity()

        if in_dim != d_model:
            self.residual_path = VNBlock(
                layers=[in_dim, d_model],
                layers_type=["relu"],
                data_dim=[4, 4],  # don't care
                bias=False,
                share_nonlinearity=share_nonlinearity,
                negative_slope=negative_slope,
                relu_in=relu_in,
                pooling=None,
            )
        else:
            self.residual_path = nn.Identity()

        if dropout > 0.0:
            self.dropout = VNDropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x, pos_enc_xx_add=0, pos_enc_x_add=0):
        """
        Args:
            x: (B, C, 3, N_x)

        """
        if self.norm_first:
            x_norm = self.sa_norm1(x)
            sa_out, sa_attn = self.self_attn(x_norm, x_norm, x_norm, pos_enc_xx_add, pos_enc_x_add)
            x = self.residual_path(x) + self.dropout(sa_out)
            x = x + self.dropout(self.sa_ff(self.sa_norm2(x)))
        else:
            sa_out, sa_attn = self.self_attn(x, x, x, pos_enc_xx_add, pos_enc_x_add)
            x = self.residual_path(x) + self.dropout(sa_out)
            x = self.sa_norm1(x)
            x = x + self.dropout(self.sa_ff(x))
            x = self.sa_norm2(x)

        return x, sa_attn


class VNCrossAttnBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model,
        in_dim,
        pos_enc_dim,
        nhead,
        dim_feedforward,
        bias_feedforward=False,
        use_qkproj=True,
        attn_mode="sub",
        head_mode="separate",
        fwd_mode="channel",
        norm_mode="layer_zca",
        norm_affine=True,
        dropout=0.0,
        norm_first=False,
        share_nonlinearity=False,
        negative_slope=0.2,
        relu_in=False,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        # CA Block
        self.cross_attn = VNMultiheadAttention(
            d_model=d_model,
            nhead=nhead,
            qdim=in_dim,
            kdim=in_dim + pos_enc_dim,
            vdim=in_dim,
            use_qkproj=use_qkproj,
            use_vproj=True,
            attn_mode=attn_mode,
            head_mode=head_mode,
            fwd_mode=fwd_mode,
            negative_slope=negative_slope,
            verbose=self.verbose,
        )
        # feed-forward, l1 -> leakyrelu -> l2
        self.ca_ff = VNBlock(
            layers=[d_model, dim_feedforward, d_model],
            layers_type=["relu", "linear"],
            data_dim=[4, 4],
            bias=bias_feedforward,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            relu_in=relu_in,
            pooling=None,
        )
        self.norm_first = norm_first
        norm_dim = in_dim if self.norm_first else d_model
        if norm_mode is not None:
            self.ca_norm1 = VNLayerNorm(num_channels=norm_dim, affine=norm_affine, bias=False, mode=norm_mode)
            self.ca_norm2 = VNLayerNorm(num_channels=norm_dim, affine=norm_affine, bias=False, mode=norm_mode)
        else:
            self.ca_norm1 = nn.Identity()
            self.ca_norm2 = nn.Identity()

        if in_dim != d_model:
            self.residual_path = VNBlock(
                layers=[in_dim, d_model],
                layers_type=["relu"],
                data_dim=[4, 4],  # don't care
                bias=False,
                share_nonlinearity=share_nonlinearity,
                negative_slope=negative_slope,
                relu_in=relu_in,
                pooling=None,
            )
        else:
            self.residual_path = nn.Identity()

        if dropout > 0.0:
            self.dropout = VNDropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, q, k, v, pos_enc_qk_concat=None, pos_enc_v_concat=None, pos_enc_qk_add=0, pos_enc_v_add=0):
        """
        Args:
            q: (B, C, 3, N_q)
            k, v: (B, C, 3, N_k)
            pos_enc_concat: (B, P, 3, N_q, N_k)
            pos_enc_add: (B, C, 3, N_q, N_k)

        """
        bs, c, _, n_q = q.shape
        k_ca = k
        if pos_enc_qk_concat is not None:
            k_ca = k_ca.view(bs, c, 3, 1, -1).expand(-1, -1, -1, n_q, -1)
            k_ca = torch.cat([k_ca, pos_enc_qk_concat], dim=1)  # (B, C+P, 3, N_q, N_k)
        v_ca = v

        if self.norm_first:
            q_norm = self.ca_norm1(q)
            q_ca = q_norm
            ca_out, ca_attn = self.cross_attn(q_ca, k_ca, v_ca, pos_enc_qk_add, pos_enc_v_add)
            q = self.residual_path(q) + self.dropout(ca_out)
            q = q + self.dropout(self.ca_ff(self.ca_norm2(q)))
        else:
            q_ca = q
            ca_out, ca_attn = self.cross_attn(q_ca, k_ca, v_ca, pos_enc_qk_add, pos_enc_v_add)
            q = self.residual_path(q) + self.dropout(ca_out)
            q = self.ca_norm1(q)
            q = q + self.dropout(self.ca_ff(q))
            q = self.ca_norm2(q)

        return q, ca_attn


class VNSimpleRebuildFCRotInv(nn.Module):
    def __init__(
        self,
        *,
        local_dim,
        global_dim,
        up_ratio,
        norm,
        refine_cfg,
        vn_cfg,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.mlp_std = VNStdFeature(
            in_channels=local_dim + global_dim,
            dim=4,
            normalize_frame=True,
            norm=norm,
            **vn_cfg,
        )
        self.rebuild_fc = nn.Sequential(
            nn.Conv1d((local_dim + global_dim) * 3, refine_cfg["hidden_dim"], kernel_size=1),
            nn.GELU(),
            nn.Conv1d(refine_cfg["hidden_dim"], up_ratio * 3, kernel_size=1),
        )

    def forward(self, xyz, x_global, x_local, need_dense_xyz=False):
        """
        Args:
            xyz: (B, 3, N)
            x_global: (B, C_g, 3)
            x_local: (B, C_l, 3, N)

        """
        bs, _, _, num_coarse = x_local.shape
        xyz = xyz.view(bs, 3, num_coarse)  # (B, 3, N)
        # transform to rot-inv anchor feature
        x_global = x_global.view(bs, -1, 3, 1).expand(-1, -1, -1, num_coarse)
        x_feat = torch.cat([x_global, x_local], dim=1)  # (B, C_g+C_l, 3, N)
        x_feat_std, z0 = self.mlp_std(x_feat)  # (B, C_g+C_l, 3, N) rot-inv, (B, 3, 3, N)
        equiv_matrix = z0.transpose(1, 2)

        x_feat_std = x_feat_std.view(bs, -1, num_coarse)  # (B, (C_g+C_l)*3, N)
        offset_std = self.rebuild_fc(x_feat_std)  # (B, up*3, N)
        offset_std = offset_std.view(bs, self.up_ratio, 3, -1).permute(0, 2, 3, 1)  # (B, 3, N, up)
        offset = torch.einsum("bink,bijn->bjnk", offset_std, equiv_matrix)  # (B, 3, N, up)
        return offset
