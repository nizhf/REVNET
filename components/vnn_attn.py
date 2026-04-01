import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class VNMultiheadStdFeature(nn.Module):
    def __init__(
        self,
        *,
        head_dim,
        nhead,
        mode="separate",
    ):
        super().__init__()
        # Linear -> LeakyReLU -> Linear
        self.mode = mode
        self.head_dim = head_dim
        nhead = nhead if self.mode == "separate" else 1
        self.w1 = nn.Parameter(torch.empty(nhead, 3, head_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        """
        x: subtraction attention (B, nhead, emb_dim / nhead, 3, N_q, N_k)
        """
        # equivalent to:
        # z0 = torch.einsum("hji,bhidmn->bhjdmn", self.w1, x)  # (B, nhead, 2or3, 3, N_q, N_k)
        # x_std = torch.einsum("bhidmn,bhdjmn->bhijmn", x, z0)
        bs, n_h, d_h, n_d, n_q, n_k = x.shape
        assert n_d == 3
        w1 = self.w1 if self.mode == "separate" else self.w1.expand(n_h, -1, -1)  # (nhead, 3, head_dim)
        x_std_flat = x.new_zeros(bs * n_h, d_h, 3, n_q * n_k)  # (B*nhead, head_dim, 3, N_q*N_k)
        w1_repeat = w1.repeat(bs, 1, 1)  # (B*nhead, 3, head_dim)

        for d in range(3):
            xd = x[:, :, :, d, :, :].reshape(bs * n_h, d_h, n_q * n_k)  # (B*nhead, head_dim, N_q*N_k)
            ud = torch.bmm(w1_repeat, xd)  # w1 @ xd = (B*nhead, 3, N_q*N_k)
            # (B*nhead, 1, head_dim, N_q*N_k) @ (B*nhead, 3, 1, N_q*N_k) = (B*nhead, 3, head_dim, N_q*N_k)
            contrib = xd.unsqueeze(1) * ud.unsqueeze(2)
            x_std_flat += contrib.permute(0, 2, 1, 3)  # (B*nhead, head_dim, 3, N_q*N_k)

        x_std = x_std_flat.view(bs, n_h, d_h, 3, n_q, n_k)
        return x_std


class HeadwiseMLP(nn.Module):
    def __init__(self, head_dim, nhead, fwd_mode="channel", negative_slope=0.2):
        super().__init__()
        self.nhead = nhead
        self.head_dim = head_dim
        self.conv1 = nn.Conv1d(
            in_channels=nhead * head_dim * 3, out_channels=nhead * head_dim, kernel_size=1, groups=nhead, bias=True
        )
        self.act = nn.LeakyReLU(negative_slope, inplace=True)
        if fwd_mode == "channel":
            self.conv2 = nn.Conv1d(
                in_channels=nhead * head_dim, out_channels=nhead * head_dim, kernel_size=1, groups=nhead, bias=True
            )
        elif fwd_mode == "dim":
            self.conv2 = nn.Conv1d(
                in_channels=nhead * head_dim, out_channels=nhead * 3, kernel_size=1, groups=nhead, bias=True
            )
        else:  # "full"
            self.conv2 = nn.Conv1d(
                in_channels=nhead * head_dim, out_channels=nhead * 1, kernel_size=1, groups=nhead, bias=True
            )

    def forward(self, attn: torch.Tensor):
        bs, nh, d_h3, n_q, n_k = attn.shape  # (B, nhead, head_dim*3, N_q, N_k)
        attn = attn.view(bs, nh * d_h3, n_q * n_k)  # (B, nhead*head_dim*3, N_q*N_k)
        attn = self.conv1(attn)  # (B, nhead*head_dim, N_q*N_k)
        attn = self.act(attn)
        attn = self.conv2(attn)  # (B, nhead*C_out, N_q*N_k)
        attn = attn.view(bs, nh, -1, n_q, n_k).contiguous()  # (B, nhead, C_out, N_q, N_k)
        return attn


class VNMultiheadAttention(nn.Module):
    """Vector Neuron Multihead Attention

    Args:
        emb_dim (int): dimension of embbeding space
        nhead (int): number of head
        attn_mode (str, optional): attention type, sub (subtraction) or dot (dot product). Defaults to "sub".
        head_mode (str, optional): in subtraction mode, same std and MLP for each head (same), or (separate). Defaults to "separate".
        fwd_mode (str, optional): in subtraction mode, forward attention produces out_dim=head_dim (channel) or 3 (dim) or 1 (full). Defaults to "channel"
    """

    def __init__(
        self,
        *,
        d_model,
        nhead,
        qdim=None,
        kdim=None,
        vdim=None,
        use_qkproj=True,
        use_vproj=True,
        attn_mode="sub",
        head_mode="separate",
        fwd_mode="channel",
        negative_slope=0.2,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = self.d_model // self.nhead
        self.scale = math.sqrt(self.d_head)

        if qdim is None:
            qdim = d_model
        if kdim is None:
            kdim = d_model
        if vdim is None:
            vdim = d_model
        self.use_qkproj = use_qkproj
        if self.use_qkproj:
            self.q_proj_w = nn.Parameter(torch.empty(d_model, qdim))
            self.k_proj_w = nn.Parameter(torch.empty(d_model, kdim))
        self.use_vproj = use_vproj
        if self.use_vproj:
            self.v_proj_w = nn.Parameter(torch.empty(d_model, vdim))
            self.out_proj_w = nn.Parameter(torch.empty(d_model, d_model))
        self._reset_proj_parameters()

        self.attn_mode = attn_mode
        self.head_mode = head_mode
        self.fwd_mode = fwd_mode

        if self.head_mode == "separate":
            h = self.nhead
        else:
            h = 1

        if self.attn_mode == "sub":
            self.attn_std = VNMultiheadStdFeature(
                head_dim=self.d_head,
                nhead=h,
                mode=self.head_mode,
            )
            self.headwise_mlp = HeadwiseMLP(self.d_head, self.nhead, self.fwd_mode, negative_slope)
        else:
            self.fwd_mode = "full"

    def _reset_proj_parameters(self):
        if self.use_qkproj:
            nn.init.xavier_uniform_(self.q_proj_w)
            nn.init.xavier_uniform_(self.k_proj_w)
        if self.use_vproj:
            nn.init.xavier_uniform_(self.v_proj_w)
            nn.init.xavier_uniform_(self.out_proj_w)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos_enc_qk=0, pos_enc_v=0):
        """
        For subtraction global attention:
            q: (B, emb_dim, 3, N_q)
            k: (B, emb_dim, 3, N_k)
            v: (B, emb_dim, 3, N_k)
        For dot product global attention:
            q: (B, emb_dim, 3, N_q)
            k: (B, emb_dim, 3, N_k)
            v: (B, emb_dim, 3, N_k)
        Output: (B, emb_dim, 3, N_q)
        """
        bs, n_q, n_k = q.shape[0], q.shape[3], k.shape[-1]
        # QKV projection
        if self.use_qkproj:
            q = torch.einsum("ji,bid...->bjd...", self.q_proj_w, q)
            k = torch.einsum("ji,bid...->bjd...", self.k_proj_w, k)
        if self.use_vproj:
            v = torch.einsum("ji,bid...->bjd...", self.v_proj_w, v)
        # subtraction attention, must multiply with another rot-equiv matrix to make the weight rot-inv
        if self.attn_mode == "sub":
            with torch.autocast(device_type="cuda", enabled=True):
                if len(q.shape) == 4:
                    q = q.view(bs, -1, 3, n_q, 1)
                if len(k.shape) == 4:
                    k = k.view(bs, -1, 3, 1, n_k)
                # pos_encoding must be added after subtraction
                attn = q - k + pos_enc_qk  # (B, emb_dim, 3, N_q, N_k)
                # apply multihead rot-inv operation, shape same
                attn = attn.view(bs, self.nhead, self.d_head, 3, n_q, n_k)  # (B, nhead, head_dim, 3, N_q, N_k)
                attn = self.attn_std(attn)  # (B, nhead, head_dim, 3, N_q, N_k) rot-inv
                attn = attn.view(
                    bs, self.nhead, self.d_head * 3, n_q, n_k
                )  # (B, nhead, head_dim * 3, N_q, N_k) rot-inv
                # attention forward
                attn = self.headwise_mlp(attn)  # (B, nhead, C_out, N_q, N_k)
                # attn softmax
                attn = attn.float()
                attn = F.softmax(attn, dim=-1)
                v = v + pos_enc_v

                # weights apply on each channel (a channel is a 3d vector)
                if self.fwd_mode == "channel":  # attn (B, nhead, head_dim, N_q, N_k)
                    attn = attn.view(bs, self.d_model, n_q, n_k)  # (B, emb_dim, N_q, N_k)
                    result = torch.einsum("bcqk,bcdk->bcdq", attn, v)
                # weights apply on each feature vector dimension
                elif self.fwd_mode == "dim":  # attn (B, nhead, 3, N_q, N_k)
                    v = v.view(bs, self.nhead, self.d_head, 3, n_k)  # (B, nhead, head_dim, 3, N_k)
                    result = torch.einsum("bhdqk,bhcdk->bhcdq", attn, v)  # (B, nhead, head_dim, 3, N_q)
                    result = result.reshape(bs, self.d_model, 3, n_q)
                # weights apply on the full feature vector list
                else:  # attn (B, nhead, 1, N_q, N_k)
                    v = v.view(bs, self.nhead, self.d_head, 3, n_k)  # (B, nhead, head_dim, 3, N_k)
                    result = torch.einsum("bhqk,bhcdk->bhcdq", attn.view(bs, self.nhead, n_q, n_k), v)
                    result = result.reshape(bs, self.d_model, 3, n_q)
                # output projection
                result = torch.einsum("ji,bidm->bjdm", self.out_proj_w, result)

        # dot product attention, weight is already rot-inv
        else:
            # q: (B, nhead, N_q, head_dim*3), kv: (B, nhead, N_k, head_dim*3)
            q = q.reshape(bs, self.nhead, self.d_head * 3, n_q).transpose(2, 3).contiguous()
            k = k.reshape(bs, self.nhead, self.d_head * 3, n_k).transpose(2, 3).contiguous()
            v = v.reshape(bs, self.nhead, self.d_head * 3, n_k).transpose(2, 3).contiguous()
            # result: (B, nhead, N_q, head_dim*3)
            result = F.scaled_dot_product_attention(q, k, v)
            # (B, emb_dim, 3, N_q)
            result = result.transpose(2, 3).reshape(bs, self.d_model, 3, n_q)
            attn = None
            # output projection
            result = torch.einsum("ji,bidm->bjdm", self.out_proj_w, result)

        return result, attn
