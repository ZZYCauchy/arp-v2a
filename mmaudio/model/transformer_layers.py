from typing import Optional
import math
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from mmaudio.ext.rotary_embeddings import apply_rope
from mmaudio.model.low_level import MLP, ChannelLastConv1d, ConvMLP


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


# ============================================================
# P2P-style joint attention hook (shared by Methods A, C)
# ============================================================
# MMAudio's JointBlock concatenates `[latent || clip || text]` along the
# token-axis and runs ONE joint self-attention. To do Attend-and-Excite
# (loss on the latent->text attention sub-block) and Token Reweighting
# (additive bias on logits to text keys before softmax), we need a manual
# attention path. By default the context is None and SDPA is used (zero
# overhead). When set, JointBlock routes through `attention_joint_manual`.
class JointAttnContext:
    """Shared per-step state for joint attention hooks.

    Fields are intentionally simple so that they can be set from
    `guided_inference.py` without coupling to specific guidance modules.
        bias_text:         (text_len,) additive bias added to attn logits
                            before softmax, restricted to the latent->text
                            sub-block. Used by Method C (token reweight).
        bias_clip:         scalar or (clip_len,) additive bias added to attn
                            logits before softmax, restricted to the
                            latent->clip sub-block. Used by Cross-stream
                            Bias (Method CS).
        capture:           if True, save latent->text softmax map for the
                            layers in `capture_layers` into `captured_maps`.
        capture_layers:    set of layer indices for which to record maps.
        captured_maps:     dict[layer_idx -> tensor(B, H, latent_len, text_len)],
                            stored in fp32 (loss-side; small, ~0.5MB / layer).
        active_layers:     if not None, the hook runs only on these layers,
                            others fall through to SDPA. Used to limit cost.
    """

    __slots__ = ("bias_text", "bias_clip", "capture", "capture_layers",
                 "captured_maps", "active_layers")

    def __init__(self, *, bias_text=None, bias_clip=None, capture=False,
                 capture_layers=None, active_layers=None):
        self.bias_text = bias_text
        self.bias_clip = bias_clip
        self.capture = capture
        self.capture_layers = set(capture_layers) if capture_layers else None
        self.captured_maps = {}
        self.active_layers = set(active_layers) if active_layers else None


_TLS = threading.local()


def get_attn_ctx():
    return getattr(_TLS, "ctx", None)


def set_attn_ctx(ctx):
    _TLS.ctx = ctx


def clear_attn_ctx():
    _TLS.ctx = None


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # training will crash without these contiguous calls and the CUDNN limitation
    # I believe this is related to https://github.com/pytorch/pytorch/issues/133974
    # unresolved at the time of writing
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = F.scaled_dot_product_attention(q, k, v)
    out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
    return out


def attention_joint_manual(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           latent_len: int, clip_len: int, text_len: int,
                           layer_idx: int, ctx: JointAttnContext):
    """Manual joint attention with optional bias to text keys and capture
    of the latent->text softmax sub-block.

    q, k, v: (B, H, N, D), with N = latent_len + clip_len + text_len.
    Returns same shape as `attention()`: (B, N, H*D).
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    B, H, N, D = q.shape
    text_start = latent_len + clip_len
    scale = 1.0 / math.sqrt(D)

    # logits: (B, H, N, N) — fp32 for numerical stability of softmax
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale

    if ctx is not None and ctx.bias_text is not None:
        # Apply additive bias only to the latent-query rows hitting text keys.
        # bias_text: (text_len,) -> broadcast to (1, 1, latent_len, text_len)
        bt = ctx.bias_text.to(logits.device, logits.dtype)
        if bt.shape[-1] != text_len:
            # Defensive: trim/pad to text_len; should never happen in practice.
            if bt.shape[-1] > text_len:
                bt = bt[:text_len]
            else:
                pad = torch.zeros(text_len - bt.shape[-1], device=bt.device, dtype=bt.dtype)
                bt = torch.cat([bt, pad], dim=0)
        logits[:, :, :latent_len, text_start:text_start + text_len] = (
            logits[:, :, :latent_len, text_start:text_start + text_len] + bt
        )

    if ctx is not None and ctx.bias_clip is not None:
        # Apply additive bias to latent-query rows hitting CLIP keys.
        # bias_clip: scalar (0-d) or (clip_len,) -> broadcast to
        # (1, 1, latent_len, clip_len).
        bc = ctx.bias_clip.to(logits.device, logits.dtype)
        clip_start = latent_len
        clip_end = latent_len + clip_len
        if bc.dim() == 0:
            logits[:, :, :latent_len, clip_start:clip_end] = (
                logits[:, :, :latent_len, clip_start:clip_end] + bc
            )
        else:
            if bc.shape[-1] != clip_len:
                if bc.shape[-1] > clip_len:
                    bc = bc[:clip_len]
                else:
                    pad = torch.zeros(clip_len - bc.shape[-1],
                                      device=bc.device, dtype=bc.dtype)
                    bc = torch.cat([bc, pad], dim=0)
            logits[:, :, :latent_len, clip_start:clip_end] = (
                logits[:, :, :latent_len, clip_start:clip_end] + bc
            )

    attn_probs = torch.softmax(logits, dim=-1)

    if ctx is not None and ctx.capture and (
        ctx.capture_layers is None or layer_idx in ctx.capture_layers
    ):
        # Save only the small latent->text sub-block to keep memory low.
        ctx.captured_maps[layer_idx] = attn_probs[:, :, :latent_len,
                                                  text_start:text_start + text_len]

    out = torch.matmul(attn_probs.to(v.dtype), v)
    out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
    return out


class SelfAttention(nn.Module):

    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.RMSNorm(dim // nheads)
        self.k_norm = nn.RMSNorm(dim // nheads)

        self.split_into_heads = Rearrange('b n (h d j) -> b h n d j',
                                          h=nheads,
                                          d=dim // nheads,
                                          j=3)

    def pre_attention(
            self, x: torch.Tensor,
            rot: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: batch_size * n_tokens * n_channels
        qkv = self.qkv(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rot is not None:
            q = apply_rope(q, rot)
            k = apply_rope(k, rot)

        return q, k, v

    def forward(
            self,
            x: torch.Tensor,  # batch_size * n_tokens * n_channels
    ) -> torch.Tensor:
        q, k, v = self.pre_attention(x)
        out = attention(q, k, v)
        return out


class MMDitSingleBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 nhead: int,
                 mlp_ratio: float = 4.0,
                 pre_only: bool = False,
                 kernel_size: int = 7,
                 padding: int = 3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, nhead)

        self.pre_only = pre_only
        if pre_only:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        else:
            if kernel_size == 1:
                self.linear1 = nn.Linear(dim, dim)
            else:
                self.linear1 = ChannelLastConv1d(dim, dim, kernel_size=kernel_size, padding=padding)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

            if kernel_size == 1:
                self.ffn = MLP(dim, int(dim * mlp_ratio))
            else:
                self.ffn = ConvMLP(dim,
                                   int(dim * mlp_ratio),
                                   kernel_size=kernel_size,
                                   padding=padding)

            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rot: Optional[torch.Tensor]):
        # x: BS * N * D
        # cond: BS * D
        modulation = self.adaLN_modulation(c)
        if self.pre_only:
            (shift_msa, scale_msa) = modulation.chunk(2, dim=-1)
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
             gate_mlp) = modulation.chunk(6, dim=-1)

        x = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.attn.pre_attention(x, rot)
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple[torch.Tensor]):
        if self.pre_only:
            return x

        (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        x = x + self.linear1(attn_out) * gate_msa
        r = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.ffn(r) * gate_mlp

        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                rot: Optional[torch.Tensor]) -> torch.Tensor:
        # x: BS * N * D
        # cond: BS * D
        x_qkv, x_conditions = self.pre_attention(x, cond, rot)
        attn_out = attention(*x_qkv)
        x = self.post_attention(x, attn_out, x_conditions)

        return x


class JointBlock(nn.Module):

    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, pre_only: bool = False):
        super().__init__()
        self.pre_only = pre_only
        # Layer index is set by MMAudio.__init__ after construction; defaults
        # to -1 so that "always-on" hooks still work in tests / standalone use.
        self.layer_idx = -1
        self.latent_block = MMDitSingleBlock(dim,
                                             nhead,
                                             mlp_ratio,
                                             pre_only=False,
                                             kernel_size=3,
                                             padding=1)
        self.clip_block = MMDitSingleBlock(dim,
                                           nhead,
                                           mlp_ratio,
                                           pre_only=pre_only,
                                           kernel_size=3,
                                           padding=1)
        self.text_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1)

    def forward(self, latent: torch.Tensor, clip_f: torch.Tensor, text_f: torch.Tensor,
                global_c: torch.Tensor, extended_c: torch.Tensor, latent_rot: torch.Tensor,
                clip_rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # latent: BS * N1 * D
        # clip_f: BS * N2 * D
        # c: BS * (1/N) * D
        x_qkv, x_mod = self.latent_block.pre_attention(latent, extended_c, latent_rot)
        c_qkv, c_mod = self.clip_block.pre_attention(clip_f, global_c, clip_rot)
        t_qkv, t_mod = self.text_block.pre_attention(text_f, global_c, rot=None)

        latent_len = latent.shape[1]
        clip_len = clip_f.shape[1]
        text_len = text_f.shape[1]

        joint_qkv = [torch.cat([x_qkv[i], c_qkv[i], t_qkv[i]], dim=2) for i in range(3)]

        ctx = get_attn_ctx()
        use_manual = (ctx is not None and (
            ctx.bias_text is not None or ctx.bias_clip is not None or ctx.capture
        ) and (ctx.active_layers is None or self.layer_idx in ctx.active_layers))
        if use_manual:
            attn_out = attention_joint_manual(
                *joint_qkv, latent_len=latent_len, clip_len=clip_len,
                text_len=text_len, layer_idx=self.layer_idx, ctx=ctx)
        else:
            attn_out = attention(*joint_qkv)
        x_attn_out = attn_out[:, :latent_len]
        c_attn_out = attn_out[:, latent_len:latent_len + clip_len]
        t_attn_out = attn_out[:, latent_len + clip_len:]

        latent = self.latent_block.post_attention(latent, x_attn_out, x_mod)
        if not self.pre_only:
            clip_f = self.clip_block.post_attention(clip_f, c_attn_out, c_mod)
            text_f = self.text_block.post_attention(text_f, t_attn_out, t_mod)

        return latent, clip_f, text_f


class FinalBlock(nn.Module):

    def __init__(self, dim, out_dim):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = ChannelLastConv1d(dim, out_dim, kernel_size=7, padding=3)

    def forward(self, latent, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        latent = modulate(self.norm(latent), shift, scale)
        latent = self.conv(latent)
        return latent
