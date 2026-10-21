"""Adaptive Richness Prior (ARP) for V2A inference-time prompt blending.

Bridges raw labels and CoT-enriched captions via per-video alpha computed from
ImageBind video<->text alignment:

    alpha(v) = sigmoid(beta * (s_cot - s_raw))

where s_x = cosine(IB_video(v), IB_text(x)). The same alpha then drives a
linear interpolation in (a) MMAudio's CLIP-text condition and (b) DNO's
CLAP-text reward target. Both blends are L2-renormalised back to the unit
sphere (slerp first-order approximation).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def _ib_encode_text(ib_guidance, text: str) -> torch.Tensor:
    ib_guidance._init()
    inputs = {
        ib_guidance.ModalityType.TEXT: ib_guidance.ib_data.load_and_transform_text(
            [text], ib_guidance.device),
    }
    with torch.no_grad():
        embeddings = ib_guidance.model(inputs)
    t = embeddings[ib_guidance.ModalityType.TEXT]
    return F.normalize(t, dim=-1).detach()


def compute_alpha_from_ib(ib_guidance, raw_text: str, cot_text: str,
                          beta: float = 5.0) -> Tuple[float, float, float]:
    """Closed-form alpha based on IB video-text similarity gap.

    Returns (alpha, s_raw, s_cot). Caller must have called
    ``ib_guidance.precompute_video_embed(video_path)`` before this.
    """
    assert ib_guidance is not None, "ARP requires an ImageBindGuidance instance"
    assert ib_guidance._video_embed is not None, \
        "ARP: call ib_guidance.precompute_video_embed(...) first"

    e_raw = _ib_encode_text(ib_guidance, raw_text)
    e_cot = _ib_encode_text(ib_guidance, cot_text)
    s_raw = F.cosine_similarity(ib_guidance._video_embed, e_raw).mean().item()
    s_cot = F.cosine_similarity(ib_guidance._video_embed, e_cot).mean().item()
    alpha = 1.0 / (1.0 + math.exp(-float(beta) * (s_cot - s_raw)))
    return float(alpha), float(s_raw), float(s_cot)


def blend_clip_text(feature_utils, raw_text: str, cot_text: str,
                    alpha: float) -> torch.Tensor:
    """Encode (raw, cot) via MMAudio's CLIP-text and L2-renormalised lerp."""
    e_raw = feature_utils.encode_text([raw_text])
    e_cot = feature_utils.encode_text([cot_text])
    e = float(alpha) * e_cot + (1.0 - float(alpha)) * e_raw
    return F.normalize(e, dim=-1)


def encode_clip_text_pair(feature_utils, raw_text: str, cot_text: str
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cache (e_raw, e_cot) in CLIP-text space for joint-mode blending."""
    e_raw = feature_utils.encode_text([raw_text]).detach()
    e_cot = feature_utils.encode_text([cot_text]).detach()
    return e_raw, e_cot


def encode_clap_text_pair(clap_model, raw_text: str, cot_text: str
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cache (e_raw, e_cot) in CLAP-text space for joint-mode blending."""
    with torch.no_grad():
        e_raw = clap_model.get_text_embedding([raw_text], use_tensor=True).detach()
        e_cot = clap_model.get_text_embedding([cot_text], use_tensor=True).detach()
    return e_raw, e_cot


def blend_clap_text(clap_model, raw_text: str, cot_text: str,
                    alpha: float) -> torch.Tensor:
    """Encode (raw, cot) via CLAP-text and L2-renormalised lerp."""
    e_raw, e_cot = encode_clap_text_pair(clap_model, raw_text, cot_text)
    e = float(alpha) * e_cot + (1.0 - float(alpha)) * e_raw
    return F.normalize(e, dim=-1)
