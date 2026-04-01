import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from components.vnn_layers import (
    VNBlock,
    VNLinearLeakyReLU,
    VNMaxPool,
    VNAvgPool,
    VNLayerNorm,
)
from components.revnet_blocks import (
    VNMetaGrouper,
    VNCoarseDecoderRotInv,
    VNPositionEncodingSub,
    VNPositionEncodingKNN,
    VNSelfAttnBlock,
    VNCrossAttnBlock,
    VNSimpleRebuildFCRotInv,
)
from .build import MODELS


@MODELS.register_module()
class REVNET(nn.Module):
    def __init__(
        self,
        *,
        loss_fn,
        loss_scheduler,
        num_fine,
        num_coarse,
        num_anchors,
        d_model,
        pos_enc_cfg,
        anchor_mode,
        anchor_cfg,
        missing_gen_cfg,
        transformer_in_proj_cfg,
        transformer_cfg,
        fine_cfg,
        vn_cfg,
        verbose=False,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_scheduler = loss_scheduler
        self.loss_names = ["loss", "loss_coarse", "loss_fine"]
        self.num_fine = num_fine
        self.num_coarse = num_coarse
        self.num_anchors = num_anchors
        self.num_missing_anchors = num_coarse - num_anchors
        self.up_ratio = self.num_fine // self.num_coarse
        self.d_model = d_model

        # Feature backbone and grouper
        if anchor_mode == "resmlp":
            self.grouper = VNMetaGrouper(
                num_anchors=self.num_anchors,
                **anchor_cfg,
                pos_enc_ff_cfg=copy.deepcopy(pos_enc_cfg["ff_cfg"]),
                vn_cfg=vn_cfg,
                verbose=verbose,
            )
        else:
            raise NotImplementedError(f"Feature backbone {anchor_mode}")
        # Positional encoding for anchors
        self.pos_enc_knn = VNPositionEncodingKNN(
            d_model=d_model,
            **copy.deepcopy(pos_enc_cfg),
            vn_cfg=vn_cfg,
        )
        # Coarse point cloud generation
        self.missing_gen = VNMissingGen(
            d_model=self.d_model,
            in_dim=anchor_cfg["layers"][-1],
            num_missing_anchors=self.num_missing_anchors,
            **missing_gen_cfg,
            pos_enc_knn=self.pos_enc_knn,
            vn_cfg=vn_cfg,
            verbose=verbose,
        )
        # map from backbone dim to d_model
        if transformer_cfg["num_sa_layer"] == 0:
            transformer_in_proj_cfg["layers"] = [
                anchor_cfg["layers"][-1],
                *transformer_in_proj_cfg["layers"],
                self.d_model,
            ]
            self.transformer_in_proj = VNBlock(**transformer_in_proj_cfg, **vn_cfg)
            if transformer_cfg["attn_cfg"]["norm_mode"] is not None:
                self.transformer_in_norm = VNLayerNorm(
                    num_channels=d_model,
                    affine=transformer_cfg["attn_cfg"]["norm_affine"],
                    bias=False,
                    mode=transformer_cfg["attn_cfg"]["norm_mode"],
                )
            else:
                self.transformer_in_norm = nn.Identity()
        else:
            self.transformer_in_proj = nn.Identity()
            self.transformer_in_norm = nn.Identity()
        # Missing anchor Transformer
        self.transformer = VNMissingTransformer(
            num_anchors=self.num_anchors,
            num_missing_anchors=self.num_missing_anchors,
            d_model=self.d_model,
            pos_enc_knn=self.pos_enc_knn,
            **transformer_cfg,
            vn_cfg=vn_cfg,
            verbose=verbose,
        )
        # Fine point cloud decoder
        self.fine_decoder = VNFineDecoder(
            d_model=self.d_model,
            up_ratio=self.up_ratio,
            **fine_cfg,
            vn_cfg=vn_cfg,
            verbose=verbose,
        )

    def forward(self, x, gt=None, **kwargs):
        """
        Args:
            x: (B, N, 3) point cloud
        """
        bs = x.shape[0]
        # backbone + Downsampling -> Anchor coor (B, 3, N_a) & feat (B, C128, 3, N_a), global (B, C128, 3)
        anchor_xyz, anchor_feat = self.grouper(x)
        # predict coarse missing coordinates and queries, from the existing global feature
        missing_xyz, missing_query, missing_global, all_xyz = self.missing_gen(anchor_xyz, anchor_feat)
        # map from backbone dim to d_model
        anchor_feat = self.transformer_in_norm(self.transformer_in_proj(anchor_feat))
        # use transformer to infer the missing features
        anchor_feat, missing_feat = self.transformer(anchor_feat, anchor_xyz, missing_query, missing_xyz)
        # concat for folding
        coarse = all_xyz  # (B, 3, N_a+N_m)
        feat = torch.cat([anchor_feat, missing_feat], dim=-1).contiguous()  # (B, C, 3, N_a+N_m)
        # fine decoder
        fine = self.fine_decoder(coarse, feat)
        missing_xyz = missing_xyz.transpose(1, 2).contiguous()
        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.reshape(bs, 3, -1).transpose(1, 2).contiguous()
        if gt is None:
            return coarse, fine
        else:
            out_dict = {
                "loss_coarse": self.loss_fn["loss_coarse"](missing_xyz, gt),
                "loss_fine": self.loss_fn["loss_fine"](fine, gt),
                "coarse": coarse,
                "fine": fine,
            }
            out_dict["loss"] = (
                self.loss_scheduler["loss_coarse"].get() * out_dict["loss_coarse"]
                + self.loss_scheduler["loss_fine"].get() * out_dict["loss_fine"]
            )
            return out_dict


class VNMissingGen(nn.Module):
    def __init__(
        self,
        *,
        d_model,
        in_dim,
        num_missing_anchors,
        up_dim,
        up_cfg,
        global_pooling,
        pos_enc_knn,
        coor_gen,
        coor_gen_cfg,
        query_mlp_cfg,
        vn_cfg,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        self.num_missing_anchors = num_missing_anchors
        # configure some settings
        up_cfg["layers"] = [in_dim, *up_cfg["layers"], up_dim]
        self.increase_dim = VNBlock(**up_cfg, **vn_cfg)
        # pooling layers
        pooling_layers = global_pooling.split(",")
        self.global_avg_pooling = None
        self.global_max_pooling = None
        for pooling in pooling_layers:
            match pooling:
                case "mean":
                    self.global_avg_pooling = VNAvgPool(dim=-1)
                case "max":
                    self.global_max_pooling = VNMaxPool(up_dim)
                case _:
                    raise NotImplementedError(f"Missing Anchor Generator pooling {pooling} unknown")
        if self.global_avg_pooling is not None and self.global_max_pooling is not None:
            self.reduce_global = VNLinearLeakyReLU(
                up_dim * 2,
                up_dim,
                dim=3,
                norm=None,
                bias=up_cfg["bias"],
                **vn_cfg,
            )
        elif self.global_avg_pooling is None and self.global_max_pooling is None:
            raise NotImplementedError("AvgPool and MaxPool should not be None both")
        else:
            self.reduce_global = VNLinearLeakyReLU(
                up_dim,
                up_dim,
                dim=3,
                norm=None,
                bias=up_cfg["bias"],
                **vn_cfg,
            )

        if coor_gen == "rotinv":
            self.missing_coor_gen = VNCoarseDecoderRotInv(
                up_dim=up_dim,
                num_coarse=num_missing_anchors,
                **coor_gen_cfg,
                vn_cfg=vn_cfg,
                verbose=self.verbose,
            )
        else:
            raise NotImplementedError(f"Coarse Generator: {coor_gen}")

        self.pos_enc = pos_enc_knn

        query_mlp_cfg["layers"] = [up_dim + d_model, *query_mlp_cfg["layers"], d_model]
        self.mlp_query = VNBlock(**query_mlp_cfg, **vn_cfg)

    def forward(self, anchor_xyz, anchor_feat):
        # anchor_feat: (B, C, 3, N)
        bs = anchor_feat.shape[0]
        # MLP for mapping exisiting anchors to missing global
        missing_up = self.increase_dim(anchor_feat)
        if self.global_max_pooling is not None and self.global_avg_pooling is not None:
            missing_global = torch.cat(
                [self.global_avg_pooling(missing_up), self.global_max_pooling(missing_up)], dim=1
            )
        elif self.global_avg_pooling is not None:
            missing_global = self.global_avg_pooling(missing_up)
        elif self.global_max_pooling is not None:
            missing_global = self.global_max_pooling(missing_up)
        missing_global = self.reduce_global(missing_global)
        # Generate missing coordinates based on the missing global feature
        missing_xyz = self.missing_coor_gen(missing_global)  # (B, 3, N_m), for coarse loss
        # detach coarse points
        # missing_xyz_d = missing_xyz.detach()
        all_xyz = torch.cat([anchor_xyz, missing_xyz], dim=-1)
        # use knn to get point coordinate positional encoding
        missing_pos_feat = self.pos_enc(missing_xyz, all_xyz)  # (B, d_model or up_dim, 3, N_m)
        missing_feat = torch.cat(
            [missing_global.view(bs, -1, 3, 1).expand(-1, -1, -1, self.num_missing_anchors), missing_pos_feat], dim=1
        )
        missing_query = self.mlp_query(missing_feat)
        return missing_xyz, missing_query, missing_global, all_xyz


class VNMissingTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_anchors,
        num_missing_anchors,
        d_model,
        pos_enc_knn,
        num_sa_layer,
        num_ca_layer,
        attn_cfg,
        vn_cfg,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        self.d_model = d_model
        self.num_anchors = num_anchors
        self.num_missing_anchors = num_missing_anchors
        # positional encoding for subtraction attention
        self.pos_enc_knn = pos_enc_knn
        # encoder for refining observed anchor features
        encoder = []
        for i in range(num_sa_layer):
            encoder.append(
                VNSelfAttnBlock(
                    d_model=d_model,
                    in_dim=d_model,
                    pos_enc_dim=0,
                    **attn_cfg,
                    **vn_cfg,
                    verbose=self.verbose,
                )
            )
        self.encoder = nn.ModuleList(encoder)
        # decoder for generating missing anchor features
        decoder = []
        for i in range(num_ca_layer):
            decoder.append(
                VNCrossAttnBlock(
                    d_model=d_model,
                    in_dim=d_model,
                    pos_enc_dim=0,
                    **attn_cfg,
                    **vn_cfg,
                    verbose=self.verbose,
                )
            )
        self.decoder = nn.ModuleList(decoder)

    def forward(self, x, coor, x_q, coor_q):
        """
        Args:
            x: existing anchor features (B, d_model, 3, N_a)
            coor: existing anchor xyz (B, 3, N_a)
            x_q: queries containing global info and predicted coordinates (B, d_model, 3, N_m). Defaults to None.
            coor_q: predicted coordinates (B, 3, N_m). Defaults to None.

        """
        # positional encoding
        all_coor = torch.cat([coor, coor_q], dim=-1)
        pos_enc_x = self.pos_enc_knn(coor, coor)
        pos_enc_q = self.pos_enc_knn(all_coor, all_coor)[..., -coor_q.shape[-1] :]
        # observed anchors self-attention
        x = x + pos_enc_x
        for i, block in enumerate(self.encoder):
            x, x_sa = block(x)
        # missing anchors cross-attention
        x_q = x_q + pos_enc_q
        for i, block in enumerate(self.decoder):
            x_q, qx_ca = block(x_q, x, x)

        return x, x_q


class VNFineDecoder(nn.Module):
    def __init__(
        self,
        *,
        d_model,
        up_ratio,
        decoder_dim,
        up_cfg,
        global_pooling,
        decoder_type,
        decoder_cfg,
        vn_cfg,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        up_dim = up_cfg["layers"][-1]
        up_cfg["layers"] = [d_model, *up_cfg["layers"]]
        self.increase_dim = VNBlock(**up_cfg, **vn_cfg)

        pooling_layers = global_pooling.split(",")
        self.global_avg_pooling = None
        self.global_max_pooling = None
        for pooling in pooling_layers:
            match pooling:
                case "mean":
                    self.global_avg_pooling = VNAvgPool(dim=-1)
                case "max":
                    self.global_max_pooling = VNMaxPool(up_dim)
                case _:
                    raise NotImplementedError(f"Missing Anchor Generator {pooling} unknown")

        if self.global_avg_pooling is not None and self.global_max_pooling is not None:
            self.reduce_global = VNLinearLeakyReLU(
                up_dim * 2, decoder_dim, dim=3, norm=None, bias=up_cfg["bias"], **vn_cfg
            )
        elif self.global_avg_pooling is None and self.global_max_pooling is None:
            raise NotImplementedError("AvgPool and MaxPool should not be None both")
        else:
            self.reduce_global = VNLinearLeakyReLU(
                up_dim,
                decoder_dim,
                dim=3,
                norm=None,
                bias=up_cfg["bias"],
                **vn_cfg,
            )
        if decoder_type == "simplefc":
            self.folding = VNSimpleRebuildFCRotInv(
                local_dim=d_model,
                global_dim=decoder_dim,
                up_ratio=up_ratio,
                **decoder_cfg,
                vn_cfg=vn_cfg,
            )
        else:
            raise NotImplementedError(f"Decoder {decoder_type}")

    def forward(self, coarse, x):
        """

        Args:
            coarse: (B, 3, N_coarse)
            x: (B, d_model, 3, N_coarse)
        """
        coarse_feat = x
        bs, _, num_coarse = coarse.shape
        # increase dim for global feature
        x_global = self.increase_dim(coarse_feat)
        if self.global_max_pooling is not None and self.global_avg_pooling is not None:
            x_global = torch.cat([self.global_avg_pooling(x_global), self.global_max_pooling(x_global)], dim=1)
        elif self.global_avg_pooling is not None:
            x_global = self.global_avg_pooling(x_global)
        elif self.global_max_pooling is not None:
            x_global = self.global_max_pooling(x_global)
        x_global = self.reduce_global(x_global)
        # Folding
        relative_xyz = self.folding(coarse, x_global, coarse_feat)  # (B, 3, N_coarse, up)
        # tile coarse
        fine = coarse.view(bs, 3, num_coarse, 1) + relative_xyz

        return fine
