#!/usr/bin/env python3
"""
FlowGuide: Training-Free Semantic Flow Guidance for V2A Generation
==================================================================

Modular guided inference on top of MMAudio.

Supports three guidance modules (can be used independently or combined):
  Module 1: CoT Enhanced Caption (--use_cot)
  Module 2: Semantic Flow Guidance (--use_flow_guidance)  [TODO Phase 2]
  Module 3: Temporal Attention Steering (--use_attn_steering)  [TODO Phase 3]

Usage:
  # Baseline MMAudio
  python guided_inference.py --video /path/to/video.mp4

  # + CoT Enhanced Caption
  python guided_inference.py --video /path/to/video.mp4 --use_cot

  # + CoT + Flow Guidance
  python guided_inference.py --video /path/to/video.mp4 --use_cot --use_flow_guidance
"""

import logging
import json
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, load_video, make_video,
                                setup_eval_logging, vfree_gaussian_lpf_rfft,
                                vfree_latent_freq_mix)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


# ============================================================
# Module 1: CoT Scene Analyzer
# ============================================================
# Original prompt (no label anchor, prone to hallucination)
COT_SYSTEM_PROMPT_V1 = """You are an expert audio analyst. Given video frames, describe ONLY the sounds that would be heard. Do NOT describe what you see visually.

Output a JSON object:
{
  "primary_sound": "<the single most prominent sound, 2-6 words>",
  "all_sounds": ["<sound 1>", "<sound 2>"],
  "acoustic_env": "<indoor/outdoor> <reverberant/dry>",
  "rich_caption": "<audio-only description, max 30 words, focus on what is HEARD not seen>"
}

Rules:
- The rich_caption must describe SOUNDS only, not visual content
- Use concrete audio terms: buzzing, humming, crackling, whooshing, etc.
- Keep it short and specific — do NOT write long sentences about the scene
- Example good caption: "A chainsaw buzzing loudly while cutting wood, with birds chirping in the background outdoors"
- Example bad caption: "The video shows a man using a chainsaw in a forest with green trees"

Output ONLY valid JSON."""

# Unified label-anchored prompt (v4): deterministic, with few-shot examples,
# optional unlikely_sounds field for negative prompting.
COT_PROMPT_TEMPLATE_V4 = """You are an expert audio description writer for video-to-audio generation.

The main sound in this video is: "{label}"

Your task: Look at the video frames and write a RICHER audio description that keeps "{label}" as the primary sound.

RULES:
1. primary_sound MUST be exactly "{label}"
2. rich_caption MUST begin with "{label}" — never omit or rephrase the label
3. rich_caption describes SOUNDS only (timbre, rhythm, intensity, environment) — never visual appearance
4. rich_caption: 15-25 words, concise and specific
5. all_sounds: 1-3 plausible background sounds from the scene
6. unlikely_sounds: 3-4 sound categories clearly UNRELATED to this scene

EXAMPLES:
Input label: "dog barking"
Output: {{"rich_caption":"Dog barking sharply and repeatedly in a grassy backyard, birds chirping softly in the background.","unlikely_sounds":["ocean waves","piano music","car engine"],"all_sounds":["birds chirping","wind rustling leaves"]}}

Input label: "playing piano"
Output: {{"rich_caption":"Playing piano with gentle, melodic keystrokes resonating in a quiet room with soft reverb.","unlikely_sounds":["traffic noise","dog barking","rain"],"all_sounds":["soft pedal thumps","room reverb"]}}

Now analyze the video frames. Output ONLY a single-line JSON object (no markdown, no newlines, no explanation). The rich_caption field MUST come first:"""

# Keep old names as aliases for backward compatibility
COT_PROMPT_TEMPLATE_V2 = COT_PROMPT_TEMPLATE_V4
COT_PROMPT_TEMPLATE_V3 = COT_PROMPT_TEMPLATE_V4


class CoTSceneAnalyzer:
    """Module 1: Use frozen MLLM to analyze video and produce rich audio caption.

    Supports CLAP-Reranked Beam Search: generate multiple candidates via beam
    search, then use CLAP text-text similarity to the label to select the
    caption most aligned with audio semantics.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda",
                 dtype=torch.float16, clap_model=None, clap_device=None,
                 num_beams=5):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        self.clap_model = clap_model
        self.clap_device = clap_device
        self.num_beams = num_beams

    def _init(self):
        if self.model is not None:
            return
        log.info(f"Loading MLLM: {self.model_name}...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map={"": self.device},
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()
        log.info("MLLM loaded.")

    def set_clap_reranker(self, clap_model, clap_device):
        """Attach a CLAP model for beam reranking after initialization."""
        self.clap_model = clap_model
        self.clap_device = clap_device

    def _clap_text_similarity(self, caption: str, label: str) -> float:
        """Compute CLAP text-text cosine similarity between caption and label."""
        if self.clap_model is None:
            return 0.0
        with torch.no_grad():
            emb_cap = self.clap_model.get_text_embedding([caption], use_tensor=True)
            emb_lab = self.clap_model.get_text_embedding([label], use_tensor=True)
            return torch.nn.functional.cosine_similarity(emb_cap, emb_lab).item()

    def analyze_video(self, video_path: str, num_frames: int = 8, label: str = None,
                       use_negative_prompt: bool = False) -> dict:
        """Analyze video frames via CLAP-Reranked Beam Search.

        1. Run beam search (num_beams candidates, deterministic)
        2. Parse each candidate as JSON
        3. Score each by CLAP text-text similarity to label (if CLAP available)
           plus heuristic quality score
        4. Return the best candidate
        """
        self._init()
        from PIL import Image
        import numpy as np

        video_info = load_video(Path(video_path), duration_sec=10.0, load_all_frames=True)
        all_frames = video_info.all_frames
        if all_frames is None:
            log.warning("No frames available for CoT analysis")
            return {"rich_caption": label or "", "all_sounds": [], "unlikely_sounds": []}

        total = len(all_frames)
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = [Image.fromarray(all_frames[i]) for i in indices]

        if label:
            prompt_text = COT_PROMPT_TEMPLATE_V4.format(label=label)
            log.info(f"  Using label-anchored prompt (v4) label='{label}'")
        else:
            prompt_text = COT_SYSTEM_PROMPT_V1
            log.info(f"  Using open-ended prompt (v1, no label)")

        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_input], images=frames, padding=True, return_tensors="pt"
        ).to(self.device)

        nb = self.num_beams
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=300, do_sample=False,
                num_beams=nb, num_return_sequences=nb,
            )

        input_len = inputs.input_ids.shape[1]
        responses = self.processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True)

        # Parse and score all beam candidates
        candidates = []
        for i, resp in enumerate(responses):
            result = self._parse_cot_response(resp)
            if result is None:
                continue
            caption = result.get("rich_caption", "")
            h_score = self._score_caption(caption, label)

            clap_score = 0.0
            if self.clap_model is not None and label:
                clap_score = self._clap_text_similarity(caption, label)

            # Combined: CLAP similarity dominates, heuristic as tiebreaker
            combined = clap_score + 0.1 * h_score
            candidates.append((combined, clap_score, h_score, result))

        if candidates:
            log.info(f"  Beam search: {len(candidates)}/{nb} valid candidates")
            for j, (comb, cs, hs, r) in enumerate(sorted(candidates, key=lambda x: -x[0])):
                cap = r.get("rich_caption", "")[:60]
                tag = " ★" if j == 0 else ""
                log.info(f"    beam {j}: clap_tt={cs:.3f} heur={hs:.2f} '{cap}...'{tag}")
        else:
            log.warning("  All beams failed to parse, falling back to label")
            return {"rich_caption": label or "", "primary_sound": label or "",
                    "all_sounds": [], "unlikely_sounds": []}

        best = max(candidates, key=lambda x: x[0])
        result = best[3]

        caption = result.get("rich_caption", "")
        if label and label.lower() not in caption.lower():
            result["rich_caption"] = f"{label}, {caption}"
            log.warning(f"  Label not in best caption, prepended.")

        log.info(f"  Selected (clap_tt={best[1]:.3f}): {result['rich_caption'][:80]}")
        return result

    @staticmethod
    def _parse_cot_response(response: str) -> dict | None:
        """Extract valid JSON dict from LLM response.
        If rich_caption is missing, constructs one from other fields."""
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            result = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

        # Fallback: construct rich_caption from available fields
        if not result.get("rich_caption"):
            primary = result.get("primary_sound", "")
            bg = result.get("all_sounds", [])
            env = result.get("acoustic_env", "")
            parts = [primary]
            if bg:
                parts.append(", ".join(bg[:2]) + " in the background")
            if env:
                parts.append(env)
            result["rich_caption"] = ", ".join(p for p in parts if p)

        return result

    @staticmethod
    def _score_caption(caption: str, label: str = None) -> float:
        """Heuristic quality score for a caption (0–1)."""
        if not caption:
            return 0.0
        score = 0.2
        words = caption.split()
        if 10 <= len(words) <= 30:
            score += 0.2
        elif 5 <= len(words) < 10 or 30 < len(words) <= 45:
            score += 0.1
        if label and label.lower() in caption.lower():
            score += 0.4
        audio_kw = {"sound", "noise", "tone", "hum", "buzz", "click", "rumble",
                    "echo", "resonat", "chirp", "crack", "whoosh", "splash",
                    "ring", "clang", "rustle", "whistle", "roar", "thump",
                    "crackl", "screech", "sizzl", "clatter", "squeak"}
        if any(k in caption.lower() for k in audio_kw):
            score += 0.1
        return min(score, 1.0)

    def get_rich_caption(self, video_path: str, label: str = None) -> str:
        """Convenience: get just the rich caption string."""
        result = self.analyze_video(video_path, label=label)
        return result.get("rich_caption", label or "")


# ============================================================
# Module 2: Semantic Flow Guidance
# ============================================================
class SemanticFlowGuidance:
    """Module 2: CLAP/ImageBind-guided flow correction during ODE solving.
    
    At each Euler step:
    1. Compute standard flow update: x_{t+dt} = x_t + dt * flow(t, x_t)
    2. Estimate clean data: x1_hat = x_t + (1-t) * flow(t, x_t)   [flow matching property]
    3. Decode x1_hat to audio via VAE + vocoder
    4. Compute CLAP similarity loss between audio and target text
    5. Backprop gradient to x_t and nudge: x_t -= alpha * grad
    
    This is analogous to Seeing-and-Hearing's approach but adapted for flow matching.
    """

    def __init__(self, guidance_strength=0.1, guidance_start_ratio=0.3,
                 guidance_end_ratio=0.9, guidance_interval=2, device="cuda",
                 strength_schedule: str = 'const',
                 strength_schedule_end: float | None = None):
        """
        strength_schedule:
          'const'         : use `guidance_strength` for every guided step
          'linear'        : linear ramp from `guidance_strength` to `strength_schedule_end`
          'exp'           : geometric ramp (useful when the span covers >1 decade)
          'cosine_warmup' : smooth cosine ramp (slow start, fast mid, slow end)
          'inverse_linear': linear ramp from HIGH to LOW (test if early flow-anchor helps)
        `strength_schedule_end` defaults to `guidance_strength` (i.e. const).
        """
        self.guidance_strength = guidance_strength
        self.guidance_start_ratio = guidance_start_ratio
        self.guidance_end_ratio = guidance_end_ratio
        self.guidance_interval = guidance_interval
        self.device = device
        self.strength_schedule = strength_schedule
        self.strength_schedule_end = (strength_schedule_end
                                      if strength_schedule_end is not None
                                      else guidance_strength)
        self.clap_model = None
        self._text_embed = None
        self._neg_text_embed = None

    def get_effective_strength(self, ti: int, start_step: int, end_step: int) -> float:
        """Return the strength to apply at guided step ti.

        Progress = (ti - start_step) / max(end_step - start_step - 1, 1), in [0, 1].
        """
        if self.strength_schedule == 'const' or self.strength_schedule_end == self.guidance_strength:
            return self.guidance_strength
        denom = max(end_step - start_step - 1, 1)
        frac = max(0.0, min(1.0, (ti - start_step) / denom))
        s0, s1 = self.guidance_strength, self.strength_schedule_end
        if self.strength_schedule == 'linear':
            return s0 + (s1 - s0) * frac
        if self.strength_schedule == 'exp':
            import math
            if s0 <= 0:
                s0 = 1e-6
            return float(s0 * ((s1 / s0) ** frac))
        if self.strength_schedule == 'cosine_warmup':
            import math
            # cosine from 0 -> 1
            cw = 0.5 - 0.5 * math.cos(math.pi * frac)
            return s0 + (s1 - s0) * cw
        if self.strength_schedule == 'inverse_linear':
            return s1 + (s0 - s1) * frac  # start at s0 if user wanted high->low
        return self.guidance_strength

    def _init_clap(self):
        if self.clap_model is not None:
            return
        log.info(f"[FlowGuide] Loading CLAP model on {self.device}...")
        import laion_clap
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        clap_ckpt = os.environ.get(
            "CLAP_CKPT", "/data/models/clap/630k-audioset-best.pt")
        self.clap_model.load_ckpt(clap_ckpt)
        self.clap_model = self.clap_model.to(self.device)
        self.clap_model.eval()
        for p in self.clap_model.parameters():
            p.requires_grad_(False)
        log.info(f"[FlowGuide] CLAP loaded on {self.device}.")

    def precompute_text_embed(self, text: str):
        """Pre-compute target text embedding."""
        self._init_clap()
        with torch.no_grad():
            self._text_embed = self.clap_model.get_text_embedding([text], use_tensor=True)
        log.info(f"[FlowGuide] Text embed computed for: '{text[:60]}...'")

    def precompute_negative_embed(self, negative_text: str):
        """Pre-compute negative text embedding for penalty term."""
        if not negative_text:
            self._neg_text_embed = None
            return
        self._init_clap()
        with torch.no_grad():
            self._neg_text_embed = self.clap_model.get_text_embedding([negative_text], use_tensor=True)
        log.info(f"[FlowGuide] Negative embed computed for: '{negative_text[:60]}...'")

    def precompute_text_embed_blend(self, raw_text: str, cot_text: str, alpha: float):
        """ARP closed-form blend of (raw, cot) CLAP text embeddings.

        Stores L2-renormalised mixture as ``self._text_embed``; also caches
        unblended ``_arp_e_raw`` / ``_arp_e_cot`` so the DNO joint mode can
        re-blend with a learnable alpha at each inner step.
        """
        self._init_clap()
        with torch.no_grad():
            e_raw = self.clap_model.get_text_embedding([raw_text], use_tensor=True)
            e_cot = self.clap_model.get_text_embedding([cot_text], use_tensor=True)
        a = float(alpha)
        e = a * e_cot + (1.0 - a) * e_raw
        self._text_embed = F.normalize(e, dim=-1)
        self._arp_e_raw = e_raw.detach()
        self._arp_e_cot = e_cot.detach()
        self._arp_alpha = a
        log.info(
            f"[FlowGuide] ARP CLAP blend alpha={a:.3f} "
            f"raw='{raw_text[:30]}...' cot='{cot_text[:30]}...'"
        )

    def guided_to_data(self, fm, cfg_ode_wrapper, x0, net, feature_utils, text=None,
                       imagebind_guidance=None, ib_weight=0.0, neg_weight=0.3,
                       harmonize_grads=True,
                       pann_guidance=None, pann_weight=0.0,
                       sync_guidance=None, sync_weight=0.0,
                       langevin_steps=1, langevin_eta=0.0,
                       grad_dump_dir=None, grad_dump_id=None,
                       step_start_idx=0, step_end_idx=None):
        """Flow matching Euler ODE with CLAP (+ optional ImageBind / neg / PANN classifier).
        
        Args:
            imagebind_guidance: Optional ImageBindGuidance for AV-Align objective.
            ib_weight: ImageBind loss weight. CLAP weight = 1 - ib_weight.
            neg_weight: Weight for CLAP negative penalty (push away from unlikely sounds).
            harmonize_grads: If True and multiple objectives are used, apply PCGrad
                to resolve gradient conflicts. If False, simply sum gradients (legacy).
            pann_guidance: Optional PANNsGuidance for AudioSet-classifier objective (思路J).
            pann_weight: Weight for PANN classifier loss.
            langevin_steps: Number of Langevin inner iterations per guided step (思路I).
                1 = classic single Euler correction. >1 enables TFG Langevin.
            langevin_eta: Langevin noise coefficient. Per-step update is
                x -= guidance_strength * grad + sqrt(2*eta)*noise.
            grad_dump_dir: If non-None, enable gradient dumping mode. Each signal
                (CLAP_pos / CLAP_neg / ImageBind / PANN) is backward'd separately
                and its flattened fp16 gradient is saved to this directory.
                Dump only happens on the first Langevin inner step (lv_i == 0).
            grad_dump_id: Sample identifier used in dump filenames. Required if
                grad_dump_dir is set.
        """
        self._init_clap()
        if text and self._text_embed is None:
            self.precompute_text_embed(text)

        dual_mode = imagebind_guidance is not None and ib_weight > 0
        neg_mode = getattr(self, '_neg_text_embed', None) is not None and neg_weight > 0
        pann_mode = pann_guidance is not None and pann_weight > 0
        sync_mode = sync_guidance is not None and sync_weight > 0
        clap_weight = 1.0 - ib_weight if dual_mode else 1.0

        num_steps = fm.num_steps
        steps = torch.linspace(0, 1 - fm.min_sigma, num_steps + 1)
        start_step = int(num_steps * self.guidance_start_ratio)
        end_step = int(num_steps * self.guidance_end_ratio)
        if step_end_idx is None:
            step_end_idx = num_steps
        seg_mode = (step_start_idx > 0) or (step_end_idx < num_steps)

        x = x0.clone()
        parts = ["CLAP"]
        if neg_mode:
            parts.append("NegPenalty")
        if dual_mode:
            parts.append("ImageBind")
        if pann_mode:
            parts.append("PANN")
        if sync_mode:
            parts.append("TCG")
        mode_str = "+".join(parts)
        langevin_mode = langevin_steps > 1 or langevin_eta > 0
        log.info(f"[FlowGuide] {mode_str} guided Euler, {num_steps} steps, "
                 f"guidance [{start_step}-{end_step}], strength={self.guidance_strength}"
                 + (f", ib_weight={ib_weight}" if dual_mode else "")
                 + (f", neg_weight={neg_weight}" if neg_mode else "")
                 + (f", pann_weight={pann_weight}" if pann_mode else "")
                 + (f", sync_weight={sync_weight}" if sync_mode else "")
                 + (f", langevin(steps={langevin_steps},eta={langevin_eta})"
                    if langevin_mode else ""))

        for ti, t_val in enumerate(steps[:-1]):
            if seg_mode and (ti < step_start_idx or ti >= step_end_idx):
                continue
            with torch.no_grad():
                flow = cfg_ode_wrapper(t_val, x)
                next_t = steps[ti + 1]
                dt = next_t - t_val

            should_guide = (start_step <= ti < end_step and
                            ti % self.guidance_interval == 0 and
                            self._text_embed is not None)

            if should_guide:
              for lv_i in range(langevin_steps):
                with torch.enable_grad():
                    x_guided = x.detach().clone().float().requires_grad_(True)

                    t_float = t_val.item()
                    with torch.no_grad():
                        flow_for_pred = cfg_ode_wrapper(t_val, x_guided.to(x.dtype))

                    x1_hat = x_guided + (1.0 - t_float) * flow_for_pred.float()

                    x1_unnorm = net.unnormalize(x1_hat.to(x.dtype))
                    spec = feature_utils.tod.vae.decode(x1_unnorm.transpose(1, 2))
                    audio = feature_utils.tod.vocoder(spec)

                    if audio.dim() == 3:
                        audio = audio.squeeze(1)
                    elif audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                    if audio.shape[-1] > 48000 * 10:
                        audio = audio[..., :48000 * 10]

                    # CLAP loss (text-audio semantic alignment)
                    audio_on_clap = audio.float().to(self.device)
                    audio_embed = self.clap_model.get_audio_embedding_from_data(
                        x=audio_on_clap, use_tensor=True)
                    clap_sim = torch.nn.functional.cosine_similarity(
                        audio_embed, self._text_embed.float().to(self.device))
                    loss_clap = -clap_sim.mean()

                    # Negative penalty: push audio away from unlikely sounds
                    loss_neg_val = 0.0
                    loss_neg = torch.tensor(0.0, device=self.device)
                    if neg_mode:
                        neg_sim = torch.nn.functional.cosine_similarity(
                            audio_embed, self._neg_text_embed.float().to(self.device))
                        loss_neg = neg_sim.mean()
                        loss_neg_val = loss_neg.item()

                    loss_ib_val = 0.0
                    loss_pann_val = 0.0
                    loss_sync_val = 0.0

                    if grad_dump_dir is not None:
                        # Dump mode: backward each signal *individually* for
                        # orthogonality analysis. Results also feed PCGrad.
                        grad_list = []
                        grad_names = []

                        # Always retain_graph on all backward calls here;
                        # the graph is thrown away when we leave the block.
                        # 1) CLAP positive
                        (clap_weight * loss_clap).backward(retain_graph=True)
                        if x_guided.grad is not None:
                            g_cp = x_guided.grad.detach().clone().to(x.dtype)
                            grad_list.append(g_cp)
                            grad_names.append('clap_pos')
                            x_guided.grad.zero_()

                        # 2) CLAP negative penalty
                        if neg_mode:
                            (neg_weight * loss_neg).backward(retain_graph=True)
                            if x_guided.grad is not None:
                                g_cn = x_guided.grad.detach().clone().to(x.dtype)
                                grad_list.append(g_cn)
                                grad_names.append('clap_neg')
                                x_guided.grad.zero_()

                        # 3) ImageBind AV-align
                        if dual_mode:
                            try:
                                loss_ib = imagebind_guidance.compute_av_loss(
                                    audio.float(), sample_rate=44100)
                                (ib_weight * loss_ib).backward(retain_graph=True)
                                loss_ib_val = -loss_ib.item()
                                if x_guided.grad is not None:
                                    g_ib = x_guided.grad.detach().clone().to(x.dtype)
                                    grad_list.append(g_ib)
                                    grad_names.append('imagebind')
                                    x_guided.grad.zero_()
                            except RuntimeError as e:
                                log.warning(f"  Step {ti}: ImageBind guidance failed: {e}")

                        # 4) PANN classifier
                        if pann_mode:
                            try:
                                loss_pann = pann_guidance.compute_class_loss(
                                    audio.float(), sample_rate=44100)
                                (pann_weight * loss_pann).backward(retain_graph=sync_mode)
                                loss_pann_val = loss_pann.item()
                                if x_guided.grad is not None:
                                    g_pn = x_guided.grad.detach().clone().to(x.dtype)
                                    grad_list.append(g_pn)
                                    grad_names.append('pann')
                                    x_guided.grad.zero_()
                            except RuntimeError as e:
                                log.warning(f"  Step {ti}: PANN guidance failed: {e}")

                        # 5) Temporal-Coherence (TCG) — heterogeneous signal
                        if sync_mode:
                            try:
                                loss_sync = sync_guidance.compute_sync_loss(
                                    audio.float(), sample_rate=44100)
                                (sync_weight * loss_sync).backward()
                                loss_sync_val = loss_sync.item()
                                if x_guided.grad is not None:
                                    g_sy = x_guided.grad.detach().clone().to(x.dtype)
                                    grad_list.append(g_sy)
                                    grad_names.append('sync')
                                    x_guided.grad.zero_()
                            except RuntimeError as e:
                                log.warning(f"  Step {ti}: TCG guidance failed: {e}")

                        # Dump each signal's gradient once per guided step
                        # (only on the first Langevin inner iteration)
                        if lv_i == 0:
                            import os as _os
                            _os.makedirs(grad_dump_dir, exist_ok=True)
                            for name, g in zip(grad_names, grad_list):
                                fname = f'{grad_dump_id}_t{ti:02d}_{name}.pt'
                                fpath = _os.path.join(grad_dump_dir, fname)
                                # Flatten and save as fp16 to save disk
                                torch.save(g.detach().cpu().reshape(-1).half(), fpath)

                        # Still aggregate via PCGrad for the actual update
                        if len(grad_list) >= 2:
                            grad = self._harmonize_gradients(grad_list)
                        elif grad_list:
                            grad = grad_list[0]
                        else:
                            grad = None

                    elif (dual_mode or pann_mode or sync_mode) and harmonize_grads:
                        # PCGrad: per-objective grads, project out conflicts.
                        grad_list = []

                        need_retain = dual_mode or pann_mode or sync_mode
                        clap_loss = clap_weight * loss_clap + neg_weight * loss_neg
                        clap_loss.backward(retain_graph=need_retain)
                        if x_guided.grad is not None:
                            grad_list.append(x_guided.grad.clone().to(x.dtype))
                            x_guided.grad.zero_()

                        if dual_mode:
                            try:
                                loss_ib = imagebind_guidance.compute_av_loss(
                                    audio.float(), sample_rate=44100)
                                (ib_weight * loss_ib).backward(retain_graph=(pann_mode or sync_mode))
                                loss_ib_val = -loss_ib.item()
                                if x_guided.grad is not None:
                                    grad_list.append(x_guided.grad.clone().to(x.dtype))
                                    x_guided.grad.zero_()
                            except RuntimeError as e:
                                log.warning(f"  Step {ti}: ImageBind guidance failed: {e}")

                        if pann_mode:
                            try:
                                loss_pann = pann_guidance.compute_class_loss(
                                    audio.float(), sample_rate=44100)
                                (pann_weight * loss_pann).backward(retain_graph=sync_mode)
                                loss_pann_val = loss_pann.item()
                                if x_guided.grad is not None:
                                    grad_list.append(x_guided.grad.clone().to(x.dtype))
                                    x_guided.grad.zero_()
                            except RuntimeError as e:
                                log.warning(f"  Step {ti}: PANN guidance failed: {e}")

                        if sync_mode:
                            try:
                                loss_sync = sync_guidance.compute_sync_loss(
                                    audio.float(), sample_rate=44100)
                                (sync_weight * loss_sync).backward()
                                loss_sync_val = loss_sync.item()
                                if x_guided.grad is not None:
                                    grad_list.append(x_guided.grad.clone().to(x.dtype))
                                    x_guided.grad.zero_()
                            except RuntimeError as e:
                                log.warning(f"  Step {ti}: TCG guidance failed: {e}")

                        if len(grad_list) >= 2:
                            grad = self._harmonize_gradients(grad_list)
                        elif grad_list:
                            grad = grad_list[0]
                        else:
                            grad = None
                    else:
                        # Legacy: sum all losses then single backward
                        total_loss = clap_weight * loss_clap + neg_weight * loss_neg
                        if dual_mode:
                            try:
                                loss_ib = imagebind_guidance.compute_av_loss(
                                    audio.float(), sample_rate=44100)
                                total_loss = total_loss + ib_weight * loss_ib
                                loss_ib_val = -loss_ib.item()
                            except RuntimeError as e:
                                log.warning(f"  Step {ti}: ImageBind guidance failed: {e}")
                        if pann_mode:
                            try:
                                loss_pann = pann_guidance.compute_class_loss(
                                    audio.float(), sample_rate=44100)
                                total_loss = total_loss + pann_weight * loss_pann
                                loss_pann_val = loss_pann.item()
                            except RuntimeError as e:
                                log.warning(f"  Step {ti}: PANN guidance failed: {e}")
                        if sync_mode:
                            try:
                                loss_sync = sync_guidance.compute_sync_loss(
                                    audio.float(), sample_rate=44100)
                                total_loss = total_loss + sync_weight * loss_sync
                                loss_sync_val = loss_sync.item()
                            except RuntimeError as e:
                                log.warning(f"  Step {ti}: TCG guidance failed: {e}")
                        try:
                            total_loss.backward()
                        except RuntimeError as e:
                            log.warning(f"  Step {ti}: backward failed: {e}")
                        grad = x_guided.grad.to(x.dtype) if x_guided.grad is not None else None

                    if grad is not None:
                        grad_norm = grad.norm()
                        if grad_norm > 0:
                            grad_unit = grad / (grad_norm + 1e-8)
                            s_eff = self.get_effective_strength(ti, start_step, end_step)
                            x = x - s_eff * grad_unit
                            # Langevin noise injection (思路I)
                            if langevin_eta > 0:
                                noise = torch.randn_like(x) * ((2 * langevin_eta) ** 0.5)
                                x = x + noise
                            if lv_i == 0 and ti % (self.guidance_interval * 2) == 0:
                                extra = ""
                                if neg_mode:
                                    extra += f", neg_sim={loss_neg_val:.4f}"
                                if dual_mode:
                                    extra += f", ib_sim={loss_ib_val:.4f}"
                                if pann_mode:
                                    extra += f", pann_loss={loss_pann_val:.4f}"
                                if sync_mode:
                                    extra += f", sync_loss={loss_sync_val:.4f}"
                                log.info(f"  Step {ti}: clap_sim={-loss_clap.item():.4f}"
                                         f"{extra}, grad_norm={grad_norm.item():.4e}")

            with torch.no_grad():
                x = x + dt * flow

        return x

    @staticmethod
    def _harmonize_gradients(grads: list[torch.Tensor]) -> torch.Tensor:
        """PCGrad-style gradient harmonization.

        For each gradient, project out any component that conflicts (negative
        cosine similarity) with the other gradients.  This prevents opposing
        objectives from cancelling each other out.
        """
        harmonized = []
        for i, gi in enumerate(grads):
            gi_h = gi.clone()
            gi_flat = gi_h.reshape(-1)
            for j, gj in enumerate(grads):
                if i == j:
                    continue
                gj_flat = gj.reshape(-1)
                dot = torch.dot(gi_flat, gj_flat)
                if dot < 0:
                    # Project out the conflicting component
                    gi_h = gi_h - (dot / (gj_flat.norm() ** 2 + 1e-12)) * gj.to(gi_h.dtype)
                    gi_flat = gi_h.reshape(-1)
            harmonized.append(gi_h)
        return sum(harmonized) / len(harmonized)


# ============================================================
# Module 3: ImageBind Video-Audio Alignment Guidance
# ============================================================
_IMAGEBIND_CACHE: dict = {}


class ImageBindGuidance:
    """Differentiable ImageBind video-audio alignment for dual-objective flow guidance.
    
    Precomputes video embedding once, then provides a differentiable audio→embedding
    pipeline so gradients can flow back to the latent during ODE solving.
    """

    IMAGEBIND_ROOT = os.environ.get(
        "IMAGEBIND_ROOT",
        "/data/repos/Seeing-and-Hearing/v2a/imagebind/imagebind")
    IMAGEBIND_CKPT = os.environ.get(
        "IMAGEBIND_CKPT", "/data/models/imagebind_huge.pth")

    def __init__(self, device="cuda:2"):
        self.device = device
        self.model = None
        self.ModalityType = None
        self.ib_data = None
        self._video_embed = None

    def _init(self):
        if self.model is not None:
            return

        cached = _IMAGEBIND_CACHE.get(self.device)
        if cached is not None:
            self.model = cached["model"]
            self.ModalityType = cached["ModalityType"]
            self.ib_data = cached["ib_data"]
            return

        import sys, shutil
        ib_parent = os.path.dirname(self.IMAGEBIND_ROOT)
        init_file = os.path.join(self.IMAGEBIND_ROOT, "__init__.py")
        init_bak = init_file + ".bak"
        if not os.path.exists(init_bak) and os.path.exists(init_file):
            shutil.copy2(init_file, init_bak)
        with open(init_file, "w") as f:
            f.write("# patched for direct import\n")
        if ib_parent not in sys.path:
            sys.path.insert(0, ib_parent)

        from imagebind.models.imagebind_model import ImageBindModel, ModalityType
        from imagebind import data as ib_data

        log.info(f"[ImageBind] Loading on {self.device}...")
        model = ImageBindModel(
            vision_embed_dim=1280, vision_num_blocks=32, vision_num_heads=16,
            text_embed_dim=1024, text_num_blocks=24, text_num_heads=16,
            out_embed_dim=1024, audio_drop_path=0.1, imu_drop_path=0.7,
        )
        model.load_state_dict(torch.load(self.IMAGEBIND_CKPT, map_location="cpu", weights_only=False))
        model.eval()
        model.to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.ModalityType = ModalityType
        self.ib_data = ib_data
        _IMAGEBIND_CACHE[self.device] = {
            "model": model,
            "ModalityType": ModalityType,
            "ib_data": ib_data,
        }
        log.info(f"[ImageBind] Loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params), cached for device {self.device}")

    @torch.no_grad()
    def precompute_video_embed(self, video_path: str):
        self._init()
        inputs = {
            self.ModalityType.VISION: self.ib_data.load_and_transform_video_data(
                [video_path], self.device),
        }
        embeddings = self.model(inputs)
        v = embeddings[self.ModalityType.VISION]
        self._video_embed = (v / v.norm(dim=-1, keepdim=True)).detach()
        log.info(f"[ImageBind] Video embedding precomputed for {Path(video_path).stem}")

    def audio_waveform_to_embed(self, audio_waveform, sample_rate=44100):
        """Differentiable: raw waveform → ImageBind audio embedding."""
        self._init()
        target_sr = 16000
        clip_duration_sec = 2
        clips_per_video = 3
        num_mel_bins = 128
        target_length = 204

        # Resample (differentiable)
        if sample_rate != target_sr:
            audio_16k = torchaudio.functional.resample(
                audio_waveform, orig_freq=sample_rate, new_freq=target_sr)
        else:
            audio_16k = audio_waveform

        if audio_16k.dim() == 2 and audio_16k.shape[0] > 1:
            audio_16k = audio_16k.mean(dim=0, keepdim=True)
        elif audio_16k.dim() == 1:
            audio_16k = audio_16k.unsqueeze(0)

        total_samples = audio_16k.shape[-1]
        total_dur = total_samples / target_sr

        if total_dur <= clip_duration_sec:
            clip_starts = [0.0]
        else:
            clip_starts = torch.linspace(0, total_dur - clip_duration_sec, clips_per_video).tolist()

        all_clips = []
        for start in clip_starts:
            s = int(start * target_sr)
            e = min(s + int(clip_duration_sec * target_sr), total_samples)
            clip = audio_16k[:, s:e].float().to(self.device)
            clip = clip - clip.mean()

            melspec = torchaudio.compliance.kaldi.fbank(
                clip, htk_compat=True, sample_frequency=target_sr,
                use_energy=False, window_type="hanning",
                num_mel_bins=num_mel_bins, dither=0.0,
                frame_length=25, frame_shift=10,
            )
            melspec = melspec.transpose(0, 1)  # (mel, frames)
            n = melspec.size(1)
            if n < target_length:
                melspec = torch.nn.functional.pad(melspec, (0, target_length - n))
            else:
                melspec = melspec[:, :target_length]

            melspec = (melspec - (-4.268)) / 9.138
            all_clips.append(melspec.unsqueeze(0))  # (1, mel, T)

        while len(all_clips) < clips_per_video:
            all_clips.append(all_clips[-1])

        audio_input = torch.stack(all_clips[:clips_per_video], dim=0).unsqueeze(0)
        embeddings = self.model({self.ModalityType.AUDIO: audio_input})
        a = embeddings[self.ModalityType.AUDIO]
        return a / a.norm(dim=-1, keepdim=True)

    def compute_av_loss(self, audio_waveform, sample_rate=44100):
        """Returns 1 - cosine_sim(audio, video). Lower = better alignment."""
        if self._video_embed is None:
            return torch.tensor(0.0, device=self.device)
        audio_embed = self.audio_waveform_to_embed(audio_waveform, sample_rate)
        sim = torch.nn.functional.cosine_similarity(audio_embed, self._video_embed)
        return -sim.mean()


# ============================================================
# Module 4: PANNs AudioSet-Classifier Guidance (思路 J)
# ============================================================
class PANNsGuidance:
    """Differentiable PANNs Cnn14 classifier guidance.

    Maps a VGGSound-style label to one or more AudioSet 527 class ids by
    substring match, then drives the generated audio towards those classes
    via -log p(target). This provides an orthogonal supervised signal vs
    CLAP's contrastive one.
    """

    PANN_CKPT = "/data/models/Cnn14_mAP=0.431.pth"

    def __init__(self, device="cuda:2"):
        self.device = device
        self.model = None
        self.labels = None  # list of 527 AudioSet class names
        self._target_ids = None  # list[int]
        self._resampler = None  # 44100 -> 32000

    def _init(self):
        if self.model is not None:
            return
        log.info(f"[PANN] Loading Cnn14 on {self.device}...")
        from panns_inference.models import Cnn14
        from panns_inference.config import labels as panns_labels
        self.labels = list(panns_labels)
        self.model = Cnn14(sample_rate=32000, window_size=1024,
                           hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                           classes_num=527)
        ckpt = torch.load(self.PANN_CKPT, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        log.info(f"[PANN] Cnn14 loaded on {self.device}.")

    def set_label(self, label: str):
        """Map a free-form label to AudioSet target class ids via substring match."""
        self._init()
        if not label:
            self._target_ids = None
            return
        words = [w.strip().lower() for w in label.replace(',', ' ').split() if len(w) > 2]
        stop = {"the", "and", "with", "for", "into", "onto", "being", "have", "has"}
        words = [w for w in words if w not in stop]

        matches = []
        low_labels = [l.lower() for l in self.labels]
        for idx, ln in enumerate(low_labels):
            if any(w in ln for w in words):
                matches.append(idx)

        if not matches:
            for idx, ln in enumerate(low_labels):
                if any(ln in label.lower() for ln in [ln]):
                    matches.append(idx)

        self._target_ids = sorted(set(matches))[:10]
        if self._target_ids:
            names = [self.labels[i] for i in self._target_ids]
            log.info(f"[PANN] '{label}' -> {len(self._target_ids)} class(es): {names}")
        else:
            log.warning(f"[PANN] No AudioSet class match for label '{label}'")

    def compute_class_loss(self, audio_waveform, sample_rate=44100):
        """Differentiable -log p(target class) via Cnn14.

        Args:
            audio_waveform: (B, T) tensor at `sample_rate`.
        Returns:
            scalar loss. Lower = stronger target-class presence.
        """
        if self._target_ids is None or self.model is None:
            return torch.tensor(0.0, device=self.device)

        audio = audio_waveform.to(self.device).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        # Resample 44100 -> 32000 via torchaudio (differentiable)
        if sample_rate != 32000:
            import torchaudio.functional as AF
            audio = AF.resample(audio, orig_freq=sample_rate, new_freq=32000)

        # Cnn14 returns dict with 'clipwise_output' (already sigmoid'd)
        out = self.model(audio, None)
        probs = out['clipwise_output']  # (B, 527)
        eps = 1e-6
        target_probs = probs[:, self._target_ids].clamp(min=eps, max=1 - eps)
        # Sum target probabilities (multi-label OR), then neg-log
        agg = 1.0 - torch.prod(1.0 - target_probs, dim=1)  # OR fusion
        loss = -torch.log(agg + eps).mean()
        return loss


# ============================================================
# Temporal-Coherence Guidance (S1 — Synchformer-align 轻量版)
# ============================================================
class TemporalCoherenceGuidance:
    """Time-axis coherence between video motion and audio energy.

    Uses MMAudio's already-computed Synchformer video features as the video
    temporal signature, and a differentiable mel-like energy envelope from
    the generated audio. Loss = 1 - Pearson correlation between the two
    normalized 1-D activity sequences over the video duration.

    Structurally different from CLAP/ImageBind/PANN guidance:
      - Not a pooled semantic/contrastive embedding alignment
      - Not a classifier posterior
      - Operates on the TIME axis of 1-D envelopes
    """

    def __init__(self, device: str = "cuda:2", n_bins: int = 8):
        self.device = device
        self.n_bins = n_bins
        self._video_activity = None  # (1, n_bins-1), already z-scored

    @torch.no_grad()
    def precompute_from_sync_features(self, sync_features: torch.Tensor):
        """sync_features shape is (B, T_sync, D) from MMAudio's Synchformer.

        Reduce to `n_bins` timesteps along the time axis, then take L2 norm
        of consecutive differences as the "motion activity" envelope.
        """
        f = sync_features.float().detach()
        if f.dim() == 4:
            B, S, T, D = f.shape
            f = f.reshape(B, S * T, D)
        elif f.dim() != 3:
            raise ValueError(
                f"[TCG] expected sync_features to have 3 or 4 dims, got {f.shape}")
        f = f.to(self.device)
        f = torch.nn.functional.interpolate(
            f.transpose(1, 2), size=self.n_bins,
            mode='linear', align_corners=False,
        ).transpose(1, 2)  # (B, n_bins, D)
        diff = f[:, 1:, :] - f[:, :-1, :]
        activity = diff.norm(dim=-1)  # (B, n_bins-1)
        activity = (activity - activity.mean(dim=-1, keepdim=True)) / (
            activity.std(dim=-1, keepdim=True) + 1e-6)
        self._video_activity = activity
        log.info(
            f"[TCG] video activity precomputed: shape={tuple(activity.shape)}, "
            f"range=[{activity.min().item():+.2f}, {activity.max().item():+.2f}]"
        )

    def compute_sync_loss(self, audio_waveform: torch.Tensor,
                          sample_rate: int = 44100) -> torch.Tensor:
        """Differentiable: raw waveform -> 1 - Pearson(activity, energy).

        Returns a scalar tensor on self.device with grad flowing back into
        audio_waveform.
        """
        if self._video_activity is None:
            return torch.tensor(0.0, device=self.device)

        audio = audio_waveform.float().to(self.device)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 2 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        # STFT energy envelope (autograd-safe in fp32)
        n_fft = 1024
        hop = 512
        window = torch.hann_window(n_fft, device=audio.device,
                                   dtype=audio.dtype)
        spec = torch.stft(
            audio, n_fft=n_fft, hop_length=hop,
            window=window, return_complex=True, center=True,
        )
        mag = spec.abs()
        energy = mag.pow(2).sum(dim=1)  # (B, T_frames)
        if energy.shape[-1] < self.n_bins - 1:
            # pad to at least n_bins-1 frames
            pad_n = (self.n_bins - 1) - energy.shape[-1]
            energy = torch.nn.functional.pad(energy, (0, pad_n))
        energy = torch.nn.functional.interpolate(
            energy.unsqueeze(1), size=self.n_bins - 1,
            mode='linear', align_corners=False,
        ).squeeze(1)  # (B, n_bins-1)
        energy = (energy - energy.mean(dim=-1, keepdim=True)) / (
            energy.std(dim=-1, keepdim=True) + 1e-6)

        v = self._video_activity.to(energy.device)
        if v.shape[0] != energy.shape[0]:
            v = v.expand(energy.shape[0], -1)
        pearson = (energy * v).mean(dim=-1)
        loss = 1.0 - pearson.mean()
        return loss


@torch.no_grad()
def compute_motion_envelope_from_sync(sync_features: torch.Tensor,
                                      n_clip_frames: int) -> torch.Tensor:
    """Reduce Synchformer features to a per-clip-frame motion magnitude.

    Strategy: L2 norm of temporal differences in the Synchformer feature
    sequence, interpolated to `n_clip_frames`. High values should correspond
    to moments of rapid visual change (useful for emphasising motion-rich
    CLIP tokens in the cross-stream bias).

    Args:
        sync_features: (B, T, D) or (B, S, T, D) Synchformer features.
        n_clip_frames: length of the CLIP token sequence in the DiT.

    Returns:
        1-D tensor of length `n_clip_frames` on CPU / same dtype as input
        (un-normalised; `build_clip_bias_vector` will min-max normalise).
    """
    f = sync_features.float().detach()
    if f.dim() == 4:
        B, S, T, D = f.shape
        f = f.reshape(B, S * T, D)
    # Interpolate time axis to n_clip_frames + 1 so that finite differences
    # give exactly n_clip_frames values.
    f = torch.nn.functional.interpolate(
        f.transpose(1, 2), size=n_clip_frames + 1,
        mode='linear', align_corners=False,
    ).transpose(1, 2)
    diff = f[:, 1:, :] - f[:, :-1, :]
    activity = diff.norm(dim=-1)  # (B, n_clip_frames)
    return activity[0]


# ============================================================
# Main: Guided Generate
# ============================================================
def guided_generate(
    clip_video,
    sync_video,
    text: list[str],
    *,
    negative_text=None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    # Guidance modules
    flow_guidance: SemanticFlowGuidance = None,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    **kwargs,
) -> torch.Tensor:
    """MMAudio generate with optional guidance modules.
    
    CoT caption replacement is handled externally (before inference_mode).
    The `text` parameter already contains the CoT-enhanced caption if applicable.
    """
    device = feature_utils.device
    dtype = feature_utils.dtype
    bs = len(text)

    # --- Standard feature extraction ---
    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(
            clip_video, batch_size=bs * clip_batch_size_multiplier)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(
            sync_video, batch_size=bs * sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    # If TCG (Temporal-Coherence) guidance is active, precompute the video
    # motion activity envelope from the already-computed Synchformer features.
    sg = kwargs.get('sync_guidance')
    if sg is not None and kwargs.get('sync_weight', 0.0) > 0:
        sg.precompute_from_sync_features(sync_features)

    if text is not None:
        text_features = feature_utils.encode_text(text)
    else:
        text_features = net.get_empty_string_sequence(bs)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                      device=device, dtype=dtype, generator=rng)
    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features, text_features)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(
        t, x, preprocessed_conditions, empty_conditions, cfg_strength)

    # --- Module 2+3: Flow Guidance (optionally with neg penalty / ImageBind) ---
    if flow_guidance is not None:
        x1 = flow_guidance.guided_to_data(fm, cfg_ode_wrapper, x0, net, feature_utils,
                                           text=text[0] if text else None,
                                           imagebind_guidance=kwargs.get('imagebind_guidance'),
                                           ib_weight=kwargs.get('ib_weight', 0.0),
                                           neg_weight=kwargs.get('neg_weight', 0.0),
                                           harmonize_grads=kwargs.get('harmonize_grads', False),
                                           pann_guidance=kwargs.get('pann_guidance'),
                                           pann_weight=kwargs.get('pann_weight', 0.0),
                                           sync_guidance=kwargs.get('sync_guidance'),
                                           sync_weight=kwargs.get('sync_weight', 0.0),
                                           langevin_steps=kwargs.get('langevin_steps', 1),
                                           langevin_eta=kwargs.get('langevin_eta', 0.0),
                                           grad_dump_dir=kwargs.get('grad_dump_dir'),
                                           grad_dump_id=kwargs.get('grad_dump_id'))
    else:
        x1 = fm.to_data(cfg_ode_wrapper, x0)

    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


def guided_generate_vfree(
    clip_video,
    sync_video,
    text: list[str],
    *,
    negative_text=None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    flow_guidance: SemanticFlowGuidance = None,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    vfree_iters: int = 2,
    vfree_cutoff: float = 0.25,
    vfree_tau: float = 0.05,
    vfree_sync_alpha: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """Guided MMAudio with V-FreeInit-style latent renoise between ODE passes."""
    device = feature_utils.device
    dtype = feature_utils.dtype
    bs = len(text)

    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(
            clip_video, batch_size=bs * clip_batch_size_multiplier)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(
            sync_video, batch_size=bs * sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    sg = kwargs.get('sync_guidance')
    if sg is not None and kwargs.get('sync_weight', 0.0) > 0:
        sg.precompute_from_sync_features(sync_features)

    if text is not None:
        text_features = feature_utils.encode_text(text)
    else:
        text_features = net.get_empty_string_sequence(bs)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features,
                                                        text_features)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(
        t, x, preprocessed_conditions, empty_conditions, cfg_strength)

    S = net.latent_seq_len
    H = vfree_gaussian_lpf_rfft(S, vfree_cutoff, device, dtype)
    x = torch.randn(bs, S, net.latent_dim, device=device, dtype=dtype, generator=rng)
    x1 = None
    for it in range(max(1, int(vfree_iters))):
        if flow_guidance is not None:
            x1 = flow_guidance.guided_to_data(fm, cfg_ode_wrapper, x, net, feature_utils,
                                               text=text[0] if text else None,
                                               imagebind_guidance=kwargs.get('imagebind_guidance'),
                                               ib_weight=kwargs.get('ib_weight', 0.0),
                                               neg_weight=kwargs.get('neg_weight', 0.0),
                                               harmonize_grads=kwargs.get('harmonize_grads', False),
                                               pann_guidance=kwargs.get('pann_guidance'),
                                               pann_weight=kwargs.get('pann_weight', 0.0),
                                               sync_guidance=kwargs.get('sync_guidance'),
                                               sync_weight=kwargs.get('sync_weight', 0.0),
                                               langevin_steps=kwargs.get('langevin_steps', 1),
                                               langevin_eta=kwargs.get('langevin_eta', 0.0),
                                               grad_dump_dir=kwargs.get('grad_dump_dir'),
                                               grad_dump_id=kwargs.get('grad_dump_id'))
        else:
            x1 = fm.to_data(cfg_ode_wrapper, x)
        if it >= int(vfree_iters) - 1:
            break
        eps = torch.randn_like(x1)
        z_fwd = (1.0 - float(vfree_tau)) * eps + float(vfree_tau) * x1
        eta = torch.randn_like(x1)
        x = vfree_latent_freq_mix(z_fwd, eta, H, S)
        if float(vfree_sync_alpha) != 0.0:
            sf = F.interpolate(
                sync_features.float().transpose(1, 2),
                size=S,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
            bias = sf[:, :, :net.latent_dim] * float(vfree_sync_alpha)
            x = x + bias.to(dtype=x.dtype)
    assert x1 is not None
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


# ============================================================
# Module 4: DNO-V2A — Direct Noise Optimization for V2A
# ============================================================
def _decode_to_audio_with_grad(x1_norm: torch.Tensor,
                                 feature_utils: FeaturesUtils,
                                 net: MMAudio,
                                 length_samples: int = 48000 * 10) -> torch.Tensor:
    """Differentiable VAE.decode + vocoder. Bypasses @torch.inference_mode."""
    net_dtype = next(net.parameters()).dtype
    x1_unnorm = net.unnormalize(x1_norm.to(net_dtype))
    spec = feature_utils.tod.vae.decode(x1_unnorm.transpose(1, 2))
    audio = feature_utils.tod.vocoder(spec)
    if audio.dim() == 3:
        audio = audio.squeeze(1)
    elif audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[-1] > length_samples:
        audio = audio[..., :length_samples]
    return audio


def compute_audio_clap_loss(audio: torch.Tensor,
                              clap_model,
                              pos_embed: torch.Tensor,
                              neg_embed: Optional[torch.Tensor] = None,
                              neg_weight: float = 0.3,
                              clap_device: str = 'cuda:1'):
    """Shared CLAP-based audio reward used by SemanticFlowGuidance and DNOOptimizer.

    Returns (loss, pos_sim, neg_sim) where loss = -pos_sim + neg_weight * neg_sim.
    """
    audio_on_clap = audio.float().to(clap_device)
    audio_embed = clap_model.get_audio_embedding_from_data(x=audio_on_clap, use_tensor=True)
    pos_sim = F.cosine_similarity(audio_embed, pos_embed.float().to(clap_device)).mean()
    loss = -pos_sim
    neg_sim_val = 0.0
    if neg_embed is not None and neg_weight > 0:
        neg_sim = F.cosine_similarity(audio_embed, neg_embed.float().to(clap_device)).mean()
        loss = loss + neg_weight * neg_sim
        neg_sim_val = neg_sim.item()
    return loss, pos_sim.item(), neg_sim_val


class DNOOptimizer:
    """Direct Noise Optimization for V2A.

    Optimizes initial latent x_T (in normalized flow-matching space) to maximize a
    CLAP-based reward, with a Gaussian-prior regularizer that keeps x_T inside the
    typical set of N(0, I).

    Three variants for d L / d x_T (controls cost vs faithfulness):
      A. ``tweedie``  : 1-step Tweedie estimate at t=0. Cheapest, identity grad.
      B. ``ste``      : Full ODE no_grad + identity STE on x_1. Mid cost.
      C. ``full``     : Differentiable Euler ODE with grad checkpointing. Faithful.
    """

    VARIANTS = ('A', 'B', 'C')

    def __init__(self, *, lr: float = 0.05, lambda_prior: float = 0.1,
                  neg_weight: float = 0.3, clap_device: str = 'cuda:1',
                  sync_guidance=None, sync_weight: float = 0.0,
                  imagebind_guidance=None, ib_weight: float = 0.0,
                  arp_mode: str = 'off', arp_init_alpha: Optional[float] = None,
                  arp_lr: float = 0.01, arp_lambda: float = 0.1,
                  arp_e_raw: Optional[torch.Tensor] = None,
                  arp_e_cot: Optional[torch.Tensor] = None):
        self.lr = lr
        self.lambda_prior = lambda_prior
        self.neg_weight = neg_weight
        self.clap_device = clap_device
        self.sync_guidance = sync_guidance
        self.sync_weight = float(sync_weight)
        self.imagebind_guidance = imagebind_guidance
        self.ib_weight = float(ib_weight)
        assert arp_mode in ('off', 'closed', 'joint'), f"unknown arp_mode {arp_mode}"
        self.arp_mode = arp_mode
        self.arp_init_alpha = arp_init_alpha
        self.arp_lr = float(arp_lr)
        self.arp_lambda = float(arp_lambda)
        self.arp_e_raw = arp_e_raw
        self.arp_e_cot = arp_e_cot

    @staticmethod
    def _x1_estimate(variant: str, x: torch.Tensor, ode_wrapper, fm,
                     net_dtype: torch.dtype) -> torch.Tensor:
        """Return an x1 estimate that allows gradient flow back to ``x``.

        ``x`` is fp32 with requires_grad=True; network expects ``net_dtype``.
        """
        if variant == 'A':
            t0 = torch.tensor(0.0, device=x.device, dtype=torch.float32)
            with torch.no_grad():
                flow0 = ode_wrapper(t0, x.detach().to(net_dtype)).float()
            return x + flow0
        if variant == 'B':
            with torch.no_grad():
                x1 = fm.to_data(ode_wrapper, x.detach().to(net_dtype)).float()
            return x1.detach() + (x - x.detach())
        if variant == 'C':
            def wrapped_ode(t, xx):
                return ode_wrapper(t, xx.to(net_dtype)).float()
            return fm.to_data_grad(wrapped_ode, x)
        raise ValueError(f"Unknown DNO variant: {variant}")

    def optimize(self, x_T_init: torch.Tensor, *,
                  ode_wrapper, fm, net, feature_utils,
                  clap_model, pos_embed: torch.Tensor,
                  neg_embed: Optional[torch.Tensor],
                  K_inner: int, variant: str,
                  length_samples: int = 48000 * 10):
        """Run K_inner Adam updates on x_T to maximize CLAP reward.

        Returns ``(x_T_star_in_net_dtype, history_list)``.
        ``history_list`` is a list of ``{iter, pos_sim, neg_sim, prior, total}`` per inner step.
        """
        if K_inner <= 0:
            return x_T_init.detach(), []

        net_dtype = next(net.parameters()).dtype
        x = x_T_init.detach().float().clone().requires_grad_(True)
        d = float(x.numel())
        history = []

        joint = (self.arp_mode == 'joint')
        theta = None
        a0 = None
        if joint:
            assert self.arp_init_alpha is not None and \
                self.arp_e_raw is not None and self.arp_e_cot is not None, \
                "joint ARP requires arp_init_alpha + arp_e_raw + arp_e_cot"
            a0 = float(min(max(self.arp_init_alpha, 1e-3), 1.0 - 1e-3))
            theta_init = math.log(a0 / (1.0 - a0))
            theta = torch.tensor(theta_init, device=self.clap_device,
                                 dtype=torch.float32, requires_grad=True)
            opt = torch.optim.Adam([
                {'params': [x], 'lr': self.lr},
                {'params': [theta], 'lr': self.arp_lr},
            ])
        else:
            opt = torch.optim.Adam([x], lr=self.lr)

        for k in range(K_inner):
            opt.zero_grad()
            with torch.enable_grad():
                x1_hat = self._x1_estimate(variant, x, ode_wrapper, fm, net_dtype)
                audio = _decode_to_audio_with_grad(x1_hat, feature_utils, net,
                                                    length_samples=length_samples)
                if joint:
                    e_raw_c = self.arp_e_raw.float().to(self.clap_device)
                    e_cot_c = self.arp_e_cot.float().to(self.clap_device)
                    alpha_t = torch.sigmoid(theta)
                    blended = alpha_t * e_cot_c + (1.0 - alpha_t) * e_raw_c
                    pos_for_loss = F.normalize(blended, dim=-1)
                else:
                    pos_for_loss = pos_embed
                loss_reward, pos_sim, neg_sim = compute_audio_clap_loss(
                    audio, clap_model, pos_for_loss, neg_embed,
                    neg_weight=self.neg_weight, clap_device=self.clap_device)
                loss_reward = loss_reward.to(x.device)
                # Probability regularizer: keep ||x||^2 close to d (typical N(0,I) shell).
                prior = (x.pow(2).sum() - d).pow(2) / (d * d)
                loss = loss_reward + self.lambda_prior * prior
                sync_val = 0.0
                ib_val = 0.0
                if self.sync_guidance is not None and self.sync_weight > 0:
                    loss_sync = self.sync_guidance.compute_sync_loss(audio).to(x.device)
                    loss = loss + self.sync_weight * loss_sync
                    sync_val = float(loss_sync.item())
                if self.imagebind_guidance is not None and self.ib_weight > 0:
                    loss_ib = self.imagebind_guidance.compute_av_loss(audio).to(x.device)
                    loss = loss + self.ib_weight * loss_ib
                    ib_val = float(loss_ib.item())
                arp_pen_val = 0.0
                alpha_val = float('nan') if not joint else float(alpha_t.detach().item())
                if joint:
                    arp_pen = self.arp_lambda * (alpha_t - a0).pow(2)
                    loss = loss + arp_pen.to(x.device)
                    arp_pen_val = float(arp_pen.item())
                loss.backward()
            opt.step()

            history.append({
                'iter': k,
                'pos_sim': float(pos_sim),
                'neg_sim': float(neg_sim),
                'prior': float(prior.item()),
                'sync_loss': sync_val,
                'ib_loss': ib_val,
                'arp_alpha': alpha_val,
                'arp_pen': arp_pen_val,
                'total': float(loss.item()),
                'x_norm_sq_over_d': float(x.detach().pow(2).sum().item() / d),
            })
            extra = ""
            if self.sync_guidance is not None and self.sync_weight > 0:
                extra += f" sync={sync_val:.4f}"
            if self.imagebind_guidance is not None and self.ib_weight > 0:
                extra += f" ib={ib_val:.4f}"
            if joint:
                extra += f" alpha={alpha_val:.3f} arp_pen={arp_pen_val:.4f}"
            log.info(
                f"  [DNO-{variant} k={k}] pos={pos_sim:.4f} neg={neg_sim:.4f}{extra} "
                f"prior={prior.item():.4e} ||x||^2/d={history[-1]['x_norm_sq_over_d']:.4f}"
            )

        return x.detach().to(net_dtype), history


def guided_generate_dno(
    clip_video,
    sync_video,
    text: list[str],
    *,
    negative_text=None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    flow_guidance: SemanticFlowGuidance = None,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    dno_variant: str = 'A',
    dno_kinner: int = 3,
    dno_lr: float = 0.05,
    dno_lambda_prior: float = 0.1,
    dno_neg_weight: float = 0.3,
    dno_sync_weight: float = 0.0,
    dno_ib_weight: float = 0.0,
    tcg_n_bins: int = 8,
    video_path: Optional[str] = None,
    dno_disable_final_guidance: bool = False,
    arp_mode: str = 'off',
    arp_alpha: Optional[float] = None,
    arp_raw_text: Optional[str] = None,
    arp_cot_text: Optional[str] = None,
    arp_lr: float = 0.01,
    arp_lambda: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    """Guided MMAudio with DNO-V2A initial-noise optimization.

    Pipeline:
      1. Sample x_T ~ N(0, I) (deterministic via ``rng``).
      2. Optimize x_T for K_inner Adam steps using DNO variant {A, B, C}.
      3. Run final ODE from x_T_star, optionally with SemanticFlowGuidance.
      4. Decode + vocode -> audio.
    """
    device = feature_utils.device
    dtype = feature_utils.dtype
    bs = len(text)

    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(
            clip_video, batch_size=bs * clip_batch_size_multiplier)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(
            sync_video, batch_size=bs * sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    if arp_mode != 'off':
        assert arp_alpha is not None and arp_raw_text is not None and arp_cot_text is not None, \
            "ARP requires arp_alpha + arp_raw_text + arp_cot_text"
        from arp import blend_clip_text  # lazy import to avoid circular
        text_features = blend_clip_text(feature_utils, arp_raw_text, arp_cot_text, arp_alpha)
        log.info(
            f"[ARP] CLIP blend alpha={float(arp_alpha):.3f} "
            f"raw='{arp_raw_text[:30]}' cot='{arp_cot_text[:30]}'"
        )
    elif text is not None:
        text_features = feature_utils.encode_text(text)
    else:
        text_features = net.get_empty_string_sequence(bs)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features,
                                                        text_features)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(
        t, x, preprocessed_conditions, empty_conditions, cfg_strength)

    S = net.latent_seq_len
    x_T = torch.randn(bs, S, net.latent_dim, device=device, dtype=dtype, generator=rng)

    # Make sure CLAP is loaded and embeddings are precomputed (caller should have done it).
    assert flow_guidance is not None and flow_guidance.clap_model is not None, \
        "DNO requires a SemanticFlowGuidance with CLAP loaded and pos/neg embeddings precomputed."
    assert flow_guidance._text_embed is not None, \
        "Call flow_guidance.precompute_text_embed(...) before guided_generate_dno"

    sync_g = None
    ib_g = None
    if dno_sync_weight > 0:
        sync_g = TemporalCoherenceGuidance(device=flow_guidance.device, n_bins=tcg_n_bins)
        sync_g.precompute_from_sync_features(sync_features)
    if dno_ib_weight > 0:
        assert video_path is not None, \
            "dno_ib_weight > 0 requires video_path for ImageBind video embed precompute"
        ib_g = ImageBindGuidance(device=flow_guidance.device)
        ib_g.precompute_video_embed(video_path)

    arp_e_raw = None
    arp_e_cot = None
    arp_init_alpha = None
    if arp_mode == 'joint':
        assert arp_alpha is not None, "joint ARP requires arp_alpha as initialisation"
        arp_e_raw = getattr(flow_guidance, '_arp_e_raw', None)
        arp_e_cot = getattr(flow_guidance, '_arp_e_cot', None)
        assert arp_e_raw is not None and arp_e_cot is not None, \
            "joint ARP needs flow_guidance.precompute_text_embed_blend(...) to be called first"
        arp_init_alpha = float(arp_alpha)

    optimizer = DNOOptimizer(
        lr=dno_lr,
        lambda_prior=dno_lambda_prior,
        neg_weight=dno_neg_weight,
        clap_device=flow_guidance.device,
        sync_guidance=sync_g,
        sync_weight=dno_sync_weight,
        imagebind_guidance=ib_g,
        ib_weight=dno_ib_weight,
        arp_mode=arp_mode,
        arp_init_alpha=arp_init_alpha,
        arp_lr=arp_lr,
        arp_lambda=arp_lambda,
        arp_e_raw=arp_e_raw,
        arp_e_cot=arp_e_cot,
    )

    x_T_star, history = optimizer.optimize(
        x_T,
        ode_wrapper=cfg_ode_wrapper,
        fm=fm,
        net=net,
        feature_utils=feature_utils,
        clap_model=flow_guidance.clap_model,
        pos_embed=flow_guidance._text_embed,
        neg_embed=flow_guidance._neg_text_embed,
        K_inner=int(dno_kinner),
        variant=dno_variant,
    )

    if dno_disable_final_guidance or flow_guidance is None:
        with torch.no_grad():
            x1 = fm.to_data(cfg_ode_wrapper, x_T_star)
    else:
        x1 = flow_guidance.guided_to_data(
            fm, cfg_ode_wrapper, x_T_star, net, feature_utils,
            text=text[0] if text else None,
            imagebind_guidance=kwargs.get('imagebind_guidance'),
            ib_weight=kwargs.get('ib_weight', 0.0),
            neg_weight=kwargs.get('neg_weight', 0.0),
            harmonize_grads=kwargs.get('harmonize_grads', False),
            pann_guidance=kwargs.get('pann_guidance'),
            pann_weight=kwargs.get('pann_weight', 0.0),
            sync_guidance=kwargs.get('sync_guidance'),
            sync_weight=kwargs.get('sync_weight', 0.0),
            langevin_steps=kwargs.get('langevin_steps', 1),
            langevin_eta=kwargs.get('langevin_eta', 0.0),
        )

    x1 = net.unnormalize(x1.to(dtype))
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio, history


def guided_generate_timecomp(
    clip_video,
    sync_video,
    cap_a: str,
    cap_b: str,
    *,
    negative_text: list[str] = None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    flow_guidance: SemanticFlowGuidance = None,
    swap_t_threshold: float = 0.5,
    clap_caption_for_loss: str = None,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    **kwargs,
):
    """F2: Time-Compositional Caption Guidance.

    Model conditioning swaps between two captions at ODE timestep
    `swap_t_threshold` (t in [0,1] where t->0 is noisy start, t->1 is data).
    Concretely, at integration time t < swap_t_threshold use cap_a's text
    conditioning; otherwise use cap_b. This conditions the early ODE phase
    (which establishes coarse structure) on cap_a and the late phase
    (refinement) on cap_b.

    The CLAP guidance loss inside SemanticFlowGuidance still uses
    `clap_caption_for_loss` (default = cap_b) so the gradient signal is
    consistent across steps.

    Returns: audio tensor (1, T) on `device`.
    """
    device = feature_utils.device
    dtype = feature_utils.dtype
    bs = 1

    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(
            clip_video, batch_size=bs * clip_batch_size_multiplier)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(
            sync_video, batch_size=bs * sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    text_feat_a = feature_utils.encode_text([cap_a])
    text_feat_b = feature_utils.encode_text([cap_b])

    if negative_text is not None:
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                     device=device, dtype=dtype, generator=rng)

    cond_a = net.preprocess_conditions(clip_features, sync_features, text_feat_a)
    cond_b = net.preprocess_conditions(clip_features, sync_features, text_feat_b)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    def cfg_ode_wrapper_swap(t, x):
        # t is a 0-d tensor on cpu (matches FlowMatching.run_t0_to_t1 semantics).
        t_val = t.item() if torch.is_tensor(t) else float(t)
        cond = cond_a if t_val < swap_t_threshold else cond_b
        return net.ode_wrapper(t, x, cond, empty_conditions, cfg_strength)

    if flow_guidance is not None:
        clap_caption = clap_caption_for_loss if clap_caption_for_loss else cap_b
        flow_guidance._text_embed = None
        flow_guidance._neg_text_embed = None
        flow_guidance.precompute_text_embed(clap_caption)
        x1 = flow_guidance.guided_to_data(
            fm, cfg_ode_wrapper_swap, x0, net, feature_utils,
            text=clap_caption,
            imagebind_guidance=kwargs.get('imagebind_guidance'),
            ib_weight=kwargs.get('ib_weight', 0.0),
            neg_weight=kwargs.get('neg_weight', 0.0),
            harmonize_grads=kwargs.get('harmonize_grads', False),
            pann_guidance=kwargs.get('pann_guidance'),
            pann_weight=kwargs.get('pann_weight', 0.0),
            sync_guidance=kwargs.get('sync_guidance'),
            sync_weight=kwargs.get('sync_weight', 0.0),
            langevin_steps=kwargs.get('langevin_steps', 1),
            langevin_eta=kwargs.get('langevin_eta', 0.0),
        )
    else:
        x1 = fm.to_data(cfg_ode_wrapper_swap, x0)

    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


def aae_generate(
    clip_video,
    sync_video,
    text: list[str],
    *,
    phrase_token_groups: list[list[list[int]]],
    negative_text: list[str] = None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    aae_lr: float = 2.0,
    aae_threshold: float = 0.2,
    aae_iter: int = 1,
    aae_t_window: tuple[float, float] = (0.0, 0.5),
    aae_layers=(8, 9, 10, 11, 12),
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    **kwargs,
):
    """A: Audio-Attend-and-Excite.

    Adapts the Attend-and-Excite idea (Chefer et al., 2023) to MMAudio's
    MM-DiT joint self-attention. At every ODE step in `aae_t_window`, we:
      1) set the joint attention hook to capture the latent->text attention
         sub-block on `aae_layers`,
      2) run a (cond-only) forward with `x_t` requiring gradient,
      3) compute s_p = max over latent positions of the mean attention to
         each phrase's token positions,
      4) hinge-loss `sum_p relu(threshold - s_p)^2`,
      5) take a single (or `aae_iter`) gradient step `x_t <- x_t - lr * grad`.
    Then we run the standard CFG ODE step on the updated x_t.

    `phrase_token_groups[b]` is a list of phrase-token-position lists for the
    b-th sample. Empty groups are skipped. Returns (audio, debug_dict) where
    debug_dict tracks per-step max attention and grad norms.
    """
    from mmaudio.model.transformer_layers import (
        JointAttnContext, set_attn_ctx, clear_attn_ctx, get_attn_ctx,
    )

    device = feature_utils.device
    dtype = feature_utils.dtype
    bs = len(text)

    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(
            clip_video, batch_size=bs * clip_batch_size_multiplier)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(
            sync_video, batch_size=bs * sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    text_features = feature_utils.encode_text(text)
    if negative_text is not None:
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                     device=device, dtype=dtype, generator=rng)
    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features, text_features)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    layer_set = set(int(l) for l in aae_layers)
    has_phrases = any(any(g for g in groups) for groups in phrase_token_groups)
    if not has_phrases:
        log.info('[AAE] WARNING: no phrase groups; falling back to vanilla CFG inference')

    log.info(f'[AAE] lr={aae_lr} thr={aae_threshold} iter={aae_iter} '
             f'window={aae_t_window} layers={sorted(layer_set)} '
             f'has_phrases={has_phrases}')

    debug = {'s_p_before': [], 'grad_norm': [], 'x_norm': [], 'loss': []}
    steps = torch.linspace(0, 1 - fm.min_sigma, fm.num_steps + 1)
    x = x0
    for ti, t_val in enumerate(steps[:-1]):
        t_float = float(t_val.item() if torch.is_tensor(t_val) else t_val)
        in_window = aae_t_window[0] <= t_float <= aae_t_window[1] and has_phrases

        if in_window:
            for it in range(aae_iter):
                with torch.enable_grad():
                    x_g = x.detach().clone().requires_grad_(True)
                    ctx = JointAttnContext(
                        capture=True, capture_layers=layer_set,
                        active_layers=layer_set,
                    )
                    set_attn_ctx(ctx)
                    t_b = t_val * torch.ones(len(x_g), device=x_g.device, dtype=x_g.dtype)
                    _ = net.predict_flow(x_g, t_b, preprocessed_conditions)
                    clear_attn_ctx()

                    if not ctx.captured_maps:
                        break
                    # Average over selected layers and heads:
                    # each map is (B, H, latent_seq, text_len).
                    layer_means = [m.mean(dim=1) for m in ctx.captured_maps.values()]
                    avg = sum(layer_means) / len(layer_means)  # (B, latent_seq, text_len)

                    losses = []
                    s_max_for_log = []
                    for b in range(avg.shape[0]):
                        for grp in phrase_token_groups[b]:
                            if not grp:
                                continue
                            grp_t = torch.tensor(grp, device=avg.device, dtype=torch.long)
                            phrase_attn = avg[b].index_select(-1, grp_t).mean(dim=-1)
                            s_p = phrase_attn.max()
                            s_max_for_log.append(float(s_p.detach().item()))
                            losses.append(F.relu(aae_threshold - s_p) ** 2)
                    if not losses:
                        break
                    loss = torch.stack(losses).mean()
                    loss_val = float(loss.detach().item())
                    if loss_val <= 0:
                        if it == 0:
                            debug['s_p_before'].append(s_max_for_log)
                            debug['loss'].append(loss_val)
                        break
                    grad = torch.autograd.grad(loss, x_g, retain_graph=False)[0]
                    gn = float(grad.norm().item())
                    # Normalise grad to unit norm and step by `aae_lr`, which is then
                    # interpreted as the absolute step magnitude in latent space.
                    # (Mirrors SemanticFlowGuidance.guided_to_data convention.)
                    if gn > 1e-12:
                        grad_unit = grad / (gn + 1e-8)
                    else:
                        break
                    x = (x.float() - aae_lr * grad_unit.float()).to(x.dtype)
                    if it == 0:
                        debug['s_p_before'].append(s_max_for_log)
                        debug['grad_norm'].append(gn)
                        debug['x_norm'].append(float(x.float().norm().item()))
                        debug['loss'].append(loss_val)

        clear_attn_ctx()
        with torch.no_grad():
            t_b = t_val * torch.ones(len(x), device=x.device, dtype=x.dtype)
            flow_cond = net.predict_flow(x, t_b, preprocessed_conditions)
            if cfg_strength < 1.0:
                flow = flow_cond
            else:
                flow_uncond = net.predict_flow(x, t_b, empty_conditions)
                flow = cfg_strength * flow_cond + (1 - cfg_strength) * flow_uncond
            dt = steps[ti + 1] - t_val
            x = x + dt * flow

    x1 = net.unnormalize(x)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)

    # Compress debug for JSON serialisation
    summary = {
        's_p_min': min((min(s) for s in debug['s_p_before'] if s), default=None),
        's_p_max': max((max(s) for s in debug['s_p_before'] if s), default=None),
        'grad_norm_mean': (sum(debug['grad_norm']) / len(debug['grad_norm'])
                           if debug['grad_norm'] else 0.0),
        'x_norm_first': debug['x_norm'][0] if debug['x_norm'] else None,
        'x_norm_last': debug['x_norm'][-1] if debug['x_norm'] else None,
        'loss_mean': (sum(debug['loss']) / len(debug['loss'])
                      if debug['loss'] else 0.0),
        'num_grad_steps': len(debug['grad_norm']),
    }
    return audio, summary


def aae_v2_generate(
    clip_video,
    sync_video,
    text: list[str],
    *,
    phrase_token_groups: list[list[list[int]]],
    negative_text: list[str] = None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    aae_lr: float = 1.5,
    aae_threshold: float = 0.3,
    aae_iter: int = 2,
    aae_t_window: tuple[float, float] = (0.0, 0.5),
    aae_layers=(5, 6),
    aae_schedule: str = 'cos2',
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    **kwargs,
):
    """A2: AAE Pivot — late-layer + scheduled gradient version.

    Same objective/loss as `aae_generate`, but tuned with lessons from the
    Method C (token reweight) grid search:
      * late layers (5, 6) as default — these are where C's additive bias
        had the strongest IB AV-Align signal;
      * early-half ODE window (0.0, 0.5);
      * iterated gradient (2 per step) to compensate for steeper attention
        at late layers;
      * higher attention threshold (0.3) — late-layer maps are more peaky;
      * slightly lower step magnitude (1.5);
      * NEW: per-step scalar schedule `aae_schedule` applied to the
        effective step magnitude, so gradient injection is largest at the
        start of the window and fades toward the end.
    """
    from mmaudio.model.transformer_layers import (
        JointAttnContext, set_attn_ctx, clear_attn_ctx,
    )
    from token_locator import SCHEDULES

    device = feature_utils.device
    dtype = feature_utils.dtype
    bs = len(text)

    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(
            clip_video, batch_size=bs * clip_batch_size_multiplier)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(
            sync_video, batch_size=bs * sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    text_features = feature_utils.encode_text(text)
    if negative_text is not None:
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                     device=device, dtype=dtype, generator=rng)
    preprocessed_conditions = net.preprocess_conditions(
        clip_features, sync_features, text_features)
    empty_conditions = net.get_empty_conditions(
        bs,
        negative_text_features=(negative_text_features
                                if negative_text is not None else None))

    layer_set = set(int(l) for l in aae_layers)
    has_phrases = any(any(g for g in groups) for groups in phrase_token_groups)
    if not has_phrases:
        log.info('[AAE-v2] WARNING: no phrase groups; falling back to vanilla CFG inference')

    sched_fn = SCHEDULES.get(aae_schedule, SCHEDULES['const'])

    log.info(f'[AAE-v2] lr={aae_lr} thr={aae_threshold} iter={aae_iter} '
             f'window={aae_t_window} layers={sorted(layer_set)} '
             f'schedule={aae_schedule} has_phrases={has_phrases}')

    debug = {'s_p_before': [], 'grad_norm': [], 'x_norm': [],
             'loss': [], 'effective_lr': []}
    steps = torch.linspace(0, 1 - fm.min_sigma, fm.num_steps + 1)
    x = x0
    for ti, t_val in enumerate(steps[:-1]):
        t_float = float(t_val.item() if torch.is_tensor(t_val) else t_val)
        in_window = aae_t_window[0] <= t_float <= aae_t_window[1] and has_phrases
        s_t = sched_fn(t_float, t_window=aae_t_window) if in_window else 0.0
        effective_lr = aae_lr * s_t

        if in_window and effective_lr > 0:
            for it in range(aae_iter):
                with torch.enable_grad():
                    x_g = x.detach().clone().requires_grad_(True)
                    ctx = JointAttnContext(
                        capture=True, capture_layers=layer_set,
                        active_layers=layer_set,
                    )
                    set_attn_ctx(ctx)
                    t_b = t_val * torch.ones(
                        len(x_g), device=x_g.device, dtype=x_g.dtype)
                    _ = net.predict_flow(x_g, t_b, preprocessed_conditions)
                    clear_attn_ctx()

                    if not ctx.captured_maps:
                        break
                    layer_means = [m.mean(dim=1) for m in ctx.captured_maps.values()]
                    avg = sum(layer_means) / len(layer_means)

                    losses = []
                    s_max_for_log = []
                    for b in range(avg.shape[0]):
                        for grp in phrase_token_groups[b]:
                            if not grp:
                                continue
                            grp_t = torch.tensor(grp, device=avg.device, dtype=torch.long)
                            phrase_attn = avg[b].index_select(-1, grp_t).mean(dim=-1)
                            s_p = phrase_attn.max()
                            s_max_for_log.append(float(s_p.detach().item()))
                            losses.append(F.relu(aae_threshold - s_p) ** 2)
                    if not losses:
                        break
                    loss = torch.stack(losses).mean()
                    loss_val = float(loss.detach().item())
                    if loss_val <= 0:
                        if it == 0:
                            debug['s_p_before'].append(s_max_for_log)
                            debug['loss'].append(loss_val)
                            debug['effective_lr'].append(effective_lr)
                        break
                    grad = torch.autograd.grad(loss, x_g, retain_graph=False)[0]
                    gn = float(grad.norm().item())
                    if gn > 1e-12:
                        grad_unit = grad / (gn + 1e-8)
                    else:
                        break
                    x = (x.float() - effective_lr * grad_unit.float()).to(x.dtype)
                    if it == 0:
                        debug['s_p_before'].append(s_max_for_log)
                        debug['grad_norm'].append(gn)
                        debug['x_norm'].append(float(x.float().norm().item()))
                        debug['loss'].append(loss_val)
                        debug['effective_lr'].append(effective_lr)

        clear_attn_ctx()
        with torch.no_grad():
            t_b = t_val * torch.ones(len(x), device=x.device, dtype=x.dtype)
            flow_cond = net.predict_flow(x, t_b, preprocessed_conditions)
            if cfg_strength < 1.0:
                flow = flow_cond
            else:
                flow_uncond = net.predict_flow(x, t_b, empty_conditions)
                flow = cfg_strength * flow_cond + (1 - cfg_strength) * flow_uncond
            dt = steps[ti + 1] - t_val
            x = x + dt * flow

    x1 = net.unnormalize(x)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)

    summary = {
        's_p_min': min((min(s) for s in debug['s_p_before'] if s), default=None),
        's_p_max': max((max(s) for s in debug['s_p_before'] if s), default=None),
        'grad_norm_mean': (sum(debug['grad_norm']) / len(debug['grad_norm'])
                           if debug['grad_norm'] else 0.0),
        'x_norm_first': debug['x_norm'][0] if debug['x_norm'] else None,
        'x_norm_last': debug['x_norm'][-1] if debug['x_norm'] else None,
        'loss_mean': (sum(debug['loss']) / len(debug['loss'])
                      if debug['loss'] else 0.0),
        'num_grad_steps': len(debug['grad_norm']),
        'effective_lr_first': (debug['effective_lr'][0]
                               if debug['effective_lr'] else None),
        'effective_lr_last': (debug['effective_lr'][-1]
                              if debug['effective_lr'] else None),
    }
    return audio, summary


def reweight_generate(
    clip_video,
    sync_video,
    text: list[str],
    *,
    key_token_idx: list[int],
    negative_text: list[str] = None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    rw_alpha: float = 2.0,
    rw_beta: float = 1.0,
    rw_schedule: str = 'cos2',
    rw_t_window: tuple[float, float] = (0.0, 0.7),
    stop_token_idx: list[int] = None,
    suppress_token_idx: list[int] = None,
    rw_gamma: float = 1.0,
    rw_layers: list[int] = None,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    **kwargs,
) -> torch.Tensor:
    """C: Token-level Reweight Schedule (Audio P2P style).

    Adds an additive bias `b` to the latent->text attention logits BEFORE
    softmax in every JointBlock that's in `rw_layers` (default: all). The
    bias is `log(rw_alpha)*s(t)` for `key_token_idx` positions and
    `log(rw_beta)*s(t)` for `stop_token_idx` positions, where `s(t)` is the
    schedule (`const`, `cos2`, or `linear`) over the ODE time in `rw_t_window`.

    rw_alpha > 1 emphasises key tokens; rw_beta < 1 attenuates stop tokens.
    No gradients are required, so this is nearly as fast as standard inference
    (overhead ~30% from manual attention in the rw_t_window).
    """
    from mmaudio.model.transformer_layers import (
        JointAttnContext, set_attn_ctx, clear_attn_ctx,
    )
    from token_locator import SCHEDULES, build_bias_vector

    device = feature_utils.device
    dtype = feature_utils.dtype
    bs = len(text)

    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(
            clip_video, batch_size=bs * clip_batch_size_multiplier)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(
            sync_video, batch_size=bs * sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    text_features = feature_utils.encode_text(text)
    if negative_text is not None:
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                     device=device, dtype=dtype, generator=rng)
    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features, text_features)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    sched_fn = SCHEDULES.get(rw_schedule, SCHEDULES['cos2'])
    text_len = net._text_seq_len

    def cfg_ode_with_reweight(t_val, x, bias):
        """CFG wrapper that applies the reweight bias ONLY on the cond branch."""
        t_b = t_val * torch.ones(len(x), device=x.device, dtype=x.dtype)
        if bias is not None:
            set_attn_ctx(JointAttnContext(
                bias_text=bias, capture=False, active_layers=rw_layers))
        else:
            clear_attn_ctx()
        flow_cond = net.predict_flow(x, t_b, preprocessed_conditions)
        clear_attn_ctx()
        if cfg_strength < 1.0:
            return flow_cond
        flow_uncond = net.predict_flow(x, t_b, empty_conditions)
        return cfg_strength * flow_cond + (1 - cfg_strength) * flow_uncond

    if not key_token_idx:
        log.info('[Reweight] WARNING: empty key_token_idx, falling back to vanilla inference')
        with torch.no_grad():
            x = x0
            steps = torch.linspace(0, 1 - fm.min_sigma, fm.num_steps + 1)
            for ti, t_val in enumerate(steps[:-1]):
                flow = cfg_ode_with_reweight(t_val, x, None)
                dt = steps[ti + 1] - t_val
                x = x + dt * flow
        x1 = x
    else:
        log.info(f'[Reweight] alpha={rw_alpha} beta={rw_beta} gamma={rw_gamma} '
                 f'sched={rw_schedule} window={rw_t_window} key_pos={len(key_token_idx)} '
                 f'stop_pos={len(stop_token_idx) if stop_token_idx else 0} '
                 f'suppress_pos={len(suppress_token_idx) if suppress_token_idx else 0} '
                 f'layers={"ALL" if rw_layers is None else rw_layers}')
        with torch.no_grad():
            x = x0
            steps = torch.linspace(0, 1 - fm.min_sigma, fm.num_steps + 1)
            for ti, t_val in enumerate(steps[:-1]):
                t_float = float(t_val.item() if torch.is_tensor(t_val) else t_val)
                s_t = sched_fn(t_float, t_window=rw_t_window)
                if s_t > 0:
                    bias = build_bias_vector(
                        key_token_idx, text_len=text_len,
                        alpha=rw_alpha, stop_idx=stop_token_idx or (), beta=rw_beta,
                        suppress_idx=suppress_token_idx or (), gamma=rw_gamma,
                        schedule_scale=s_t,
                        device=device, dtype=torch.float32,
                    )
                else:
                    bias = None
                flow = cfg_ode_with_reweight(t_val, x, bias)
                dt = steps[ti + 1] - t_val
                x = x + dt * flow
            clear_attn_ctx()
        x1 = x

    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


def crossstream_generate(
    clip_video,
    sync_video,
    text: list[str],
    *,
    cross_alpha: float = 2.0,
    cross_mode: str = 'uniform',
    cross_schedule: str = 'const',
    cross_t_window: tuple[float, float] = (0.0, 0.5),
    cross_layers: list[int] = None,
    text_key_token_idx: list[int] = None,
    text_alpha: float = 1.0,
    negative_text: list[str] = None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    **kwargs,
) -> torch.Tensor:
    """Cross-stream Bias (CS): amplify latent->CLIP attention.

    Applies an additive bias to the latent->CLIP attention logits in the
    specified JointBlocks during the ODE window [cross_t_start, cross_t_end].

    Modes:
      uniform: every CLIP token gets the same bias log(cross_alpha) * s(t).
      motion : bias per CLIP token is scaled by a Synchformer-derived motion
               envelope (normalised to [0, 1]), so high-motion frames get
               stronger emphasis — intended to improve AV-Align.

    Optional composition with Method C (text reweight): if
    `text_key_token_idx` is provided and `text_alpha != 1.0`, bias_text is
    built and injected in the SAME JointAttnContext as bias_clip, so both
    sub-blocks are amplified simultaneously.

    The bias is applied ONLY on the cond branch of CFG (uncond branch is
    protected by `clear_attn_ctx()`), matching `reweight_generate`.
    """
    from mmaudio.model.transformer_layers import (
        JointAttnContext, set_attn_ctx, clear_attn_ctx,
    )
    from token_locator import (
        SCHEDULES, build_bias_vector, build_clip_bias_vector,
    )

    device = feature_utils.device
    dtype = feature_utils.dtype
    bs = len(text)

    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(
            clip_video, batch_size=bs * clip_batch_size_multiplier)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(
            sync_video, batch_size=bs * sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    text_features = feature_utils.encode_text(text)
    if negative_text is not None:
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                     device=device, dtype=dtype, generator=rng)
    preprocessed_conditions = net.preprocess_conditions(
        clip_features, sync_features, text_features)
    empty_conditions = net.get_empty_conditions(
        bs,
        negative_text_features=(negative_text_features
                                if negative_text is not None else None))

    sched_fn = SCHEDULES.get(cross_schedule, SCHEDULES['const'])
    clip_len = clip_features.shape[1]
    text_len = net._text_seq_len

    motion_env_raw = None
    if cross_mode == 'motion':
        motion_env_raw = compute_motion_envelope_from_sync(
            sync_features, n_clip_frames=clip_len)

    use_text = (text_key_token_idx is not None and len(text_key_token_idx) > 0
                and text_alpha != 1.0)

    log.info(f'[CrossStream] alpha={cross_alpha} mode={cross_mode} '
             f'sched={cross_schedule} window={cross_t_window} '
             f'layers={"ALL" if cross_layers is None else cross_layers} '
             f'clip_len={clip_len} with_text={use_text} '
             f'text_alpha={text_alpha if use_text else 1.0}')

    def cfg_ode_with_cross(t_val, x, b_clip, b_text):
        t_b = t_val * torch.ones(len(x), device=x.device, dtype=x.dtype)
        if b_clip is not None or b_text is not None:
            set_attn_ctx(JointAttnContext(
                bias_text=b_text, bias_clip=b_clip,
                capture=False, active_layers=cross_layers))
        else:
            clear_attn_ctx()
        flow_cond = net.predict_flow(x, t_b, preprocessed_conditions)
        clear_attn_ctx()
        if cfg_strength < 1.0:
            return flow_cond
        flow_uncond = net.predict_flow(x, t_b, empty_conditions)
        return cfg_strength * flow_cond + (1 - cfg_strength) * flow_uncond

    with torch.no_grad():
        x = x0
        steps = torch.linspace(0, 1 - fm.min_sigma, fm.num_steps + 1)
        for ti, t_val in enumerate(steps[:-1]):
            t_float = float(t_val.item() if torch.is_tensor(t_val) else t_val)
            s_t = sched_fn(t_float, t_window=cross_t_window)
            if s_t > 0 and cross_alpha != 1.0:
                b_clip = build_clip_bias_vector(
                    clip_len=clip_len, alpha=cross_alpha,
                    schedule_scale=s_t,
                    motion_envelope=motion_env_raw,
                    device=device, dtype=torch.float32,
                )
            else:
                b_clip = None
            if s_t > 0 and use_text:
                b_text = build_bias_vector(
                    text_key_token_idx, text_len=text_len,
                    alpha=text_alpha, schedule_scale=s_t,
                    device=device, dtype=torch.float32,
                )
            else:
                b_text = None
            flow = cfg_ode_with_cross(t_val, x, b_clip, b_text)
            dt = steps[ti + 1] - t_val
            x = x + dt * flow
        clear_attn_ctx()

    x1 = net.unnormalize(x)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


def compose_generate(
    clip_video,
    sync_video,
    main_text: list[str],
    *,
    sub_prompts: list[str],
    compose_w_main: float = 0.5,
    compose_w_sub_total: float = 0.5,
    compose_template: str = None,
    compose_only_sub: bool = False,
    compose_sum_mode: bool = False,
    negative_text: list[str] = None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    **kwargs,
) -> torch.Tensor:
    """D: Compositional Decoding via Composable-CFG.

    For each ODE step, evaluate the flow for the main prompt AND every
    sub-prompt (short per-source phrase from CoT's all_sounds), then compose
    via
        flow = flow_uncond + cfg * [ w_main * (flow_main - flow_uncond)
                                   + Σ w_sub_i * (flow_sub_i - flow_uncond) ]

    By default w_main + Σ w_sub_i = 1 (average mode) so the total CFG
    magnitude is identical to standard CFG. Set `compose_sum_mode=True` to
    use Composable-Diffusion's additive variant (Σ w = k + 1) — stronger but
    less stable.

    Args:
      main_text           : length-1 list, the full CoT caption (kept as anchor)
      sub_prompts         : k phrases (from cot_v4.raw.all_sounds)
      compose_w_main      : weight for main prompt (0 if compose_only_sub)
      compose_w_sub_total : TOTAL weight for all sub-prompts (divided by k)
      compose_template    : optional f-string, e.g. "the sound of {x}"
      compose_only_sub    : if True, drop main prompt entirely (w_main=0,
                            sub-prompts alone cover Σ w = 1)
      compose_sum_mode    : if True, use Σw=k+1 instead of normalizing to 1

    Cost: (k + 2) forwards per ODE step (vs 2 for vanilla CFG). For k=2 this
    is 2× baseline, for k=3 it is 2.5× baseline.
    """
    device = feature_utils.device
    dtype = feature_utils.dtype
    bs = len(main_text)

    subs = [s.strip() for s in (sub_prompts or []) if s and s.strip()]
    if compose_template:
        subs = [compose_template.format(x=s) for s in subs]
    k = len(subs)

    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(
            clip_video, batch_size=bs * clip_batch_size_multiplier)
    else:
        clip_features = net.get_empty_clip_sequence(bs)
    if sync_video is not None:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(
            sync_video, batch_size=bs * sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    text_features_main = feature_utils.encode_text(main_text)
    text_features_subs = feature_utils.encode_text(subs) if k > 0 else None
    if negative_text is not None:
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    cond_main = net.preprocess_conditions(
        clip_features, sync_features, text_features_main)
    cond_subs = []
    for i in range(k):
        tf = text_features_subs[i:i + 1]
        cond_subs.append(net.preprocess_conditions(
            clip_features, sync_features, tf))
    empty_conditions = net.get_empty_conditions(
        bs,
        negative_text_features=(negative_text_features
                                if negative_text is not None else None))

    if compose_only_sub and k > 0:
        w_main = 0.0
        w_sub = 1.0 / k
    else:
        if compose_sum_mode:
            w_main = 1.0
            w_sub = 1.0
        else:
            if k > 0:
                w_main = float(compose_w_main)
                w_sub = float(compose_w_sub_total) / k
            else:
                w_main = 1.0
                w_sub = 0.0

    log.info(f'[Compose] k={k} w_main={w_main:.3f} w_sub_each={w_sub:.3f} '
             f'only_sub={compose_only_sub} sum_mode={compose_sum_mode} '
             f'template={compose_template!r} subs={subs}')

    x0 = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                     device=device, dtype=dtype, generator=rng)

    def step_flow(t_val, x):
        t_b = t_val * torch.ones(len(x), device=x.device, dtype=x.dtype)
        flow_uncond = net.predict_flow(x, t_b, empty_conditions)
        if cfg_strength < 1.0 and k == 0:
            return net.predict_flow(x, t_b, cond_main)
        delta = 0.0
        if w_main != 0.0:
            flow_main = net.predict_flow(x, t_b, cond_main)
            delta = delta + w_main * (flow_main - flow_uncond)
        for c in cond_subs:
            flow_sub = net.predict_flow(x, t_b, c)
            delta = delta + w_sub * (flow_sub - flow_uncond)
        return flow_uncond + cfg_strength * delta

    with torch.no_grad():
        x = x0
        steps = torch.linspace(0, 1 - fm.min_sigma, fm.num_steps + 1)
        for ti, t_val in enumerate(steps[:-1]):
            flow = step_flow(t_val, x)
            dt = steps[ti + 1] - t_val
            x = x + dt * flow

    x1 = net.unnormalize(x)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


def beam_generate(
    clip_video,
    sync_video,
    text: list[str],
    *,
    negative_text: list[str] = None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    flow_guidance: SemanticFlowGuidance,
    clap_eval_model,
    label_for_clap: str,
    beam_n: int = 4,
    beam_branch_step: int = 12,
    fork_noise_std: float = 0.05,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    **kwargs,
):
    """F3: Beam-Search over Flow ODE Trajectories.

    Phase A (steps [0, beam_branch_step)): single shared guided ODE trajectory.
    Phase B (steps [beam_branch_step, num_steps)): fork into K=beam_n branches by
        adding small Gaussian perturbations, then run K independent guided ODEs.
    Selection: decode K terminal latents and pick the one with highest CLAP score
        against label_for_clap.

    Returns: (best_audio (1, T) on CPU, best_score, all_scores list).

    Notes:
      - Only flow_guidance != None is supported (this is the use case).
      - Cost vs Best-of-N=K: Phase A is shared, so total cost ≈
        (branch_step + K * (num_steps - branch_step)) ODE steps.
      - Branch fork uses additive Gaussian noise on the latent at branch_step;
        this is sufficient to break determinism in Phase B because subsequent
        guidance updates are nonlinear in x.
    """
    device = feature_utils.device
    dtype = feature_utils.dtype
    bs = len(text)
    assert bs == 1, "beam_generate currently only supports bs=1"
    assert flow_guidance is not None, "beam_generate requires flow_guidance"

    # --- Standard feature extraction (mirrors guided_generate) ---
    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(
            clip_video, batch_size=bs * clip_batch_size_multiplier)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(
            sync_video, batch_size=bs * sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    sg = kwargs.get('sync_guidance')
    if sg is not None and kwargs.get('sync_weight', 0.0) > 0:
        sg.precompute_from_sync_features(sync_features)

    if text is not None:
        text_features = feature_utils.encode_text(text)
    else:
        text_features = net.get_empty_string_sequence(bs)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                     device=device, dtype=dtype, generator=rng)
    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features, text_features)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(
        t, x, preprocessed_conditions, empty_conditions, cfg_strength)

    num_steps = fm.num_steps
    branch = max(1, min(beam_branch_step, num_steps - 1))

    # Phase A: shared trajectory [0, branch)
    log.info(f"[Beam] Phase A: single guided ODE on steps [0, {branch})")
    x_branch = flow_guidance.guided_to_data(
        fm, cfg_ode_wrapper, x0, net, feature_utils,
        text=text[0] if text else None,
        imagebind_guidance=kwargs.get('imagebind_guidance'),
        ib_weight=kwargs.get('ib_weight', 0.0),
        neg_weight=kwargs.get('neg_weight', 0.0),
        harmonize_grads=kwargs.get('harmonize_grads', False),
        pann_guidance=kwargs.get('pann_guidance'),
        pann_weight=kwargs.get('pann_weight', 0.0),
        sync_guidance=kwargs.get('sync_guidance'),
        sync_weight=kwargs.get('sync_weight', 0.0),
        langevin_steps=kwargs.get('langevin_steps', 1),
        langevin_eta=kwargs.get('langevin_eta', 0.0),
        step_start_idx=0, step_end_idx=branch,
    )

    # Phase B: K independent guided ODEs on [branch, num_steps), each from a
    # perturbed copy of x_branch.
    branch_std = x_branch.float().std().item()
    fork_sigma = max(1e-6, fork_noise_std * branch_std)
    log.info(f"[Beam] Phase B: K={beam_n} branches on [{branch}, {num_steps}), "
             f"fork_sigma={fork_sigma:.4e} (= {fork_noise_std} * latent_std={branch_std:.4e})")

    best_score = -1e9
    best_audio = None
    all_scores = []
    for k in range(beam_n):
        # Deterministic but per-branch distinct perturbation.
        gen_b = torch.Generator(device=device).manual_seed(
            int(rng.initial_seed()) + 17 + 1009 * k)
        eps = torch.randn(x_branch.shape, device=device, dtype=dtype, generator=gen_b) * fork_sigma
        x_k = (x_branch + eps).contiguous()

        x_k_final = flow_guidance.guided_to_data(
            fm, cfg_ode_wrapper, x_k, net, feature_utils,
            text=text[0] if text else None,
            imagebind_guidance=kwargs.get('imagebind_guidance'),
            ib_weight=kwargs.get('ib_weight', 0.0),
            neg_weight=kwargs.get('neg_weight', 0.0),
            harmonize_grads=kwargs.get('harmonize_grads', False),
            pann_guidance=kwargs.get('pann_guidance'),
            pann_weight=kwargs.get('pann_weight', 0.0),
            sync_guidance=kwargs.get('sync_guidance'),
            sync_weight=kwargs.get('sync_weight', 0.0),
            langevin_steps=kwargs.get('langevin_steps', 1),
            langevin_eta=kwargs.get('langevin_eta', 0.0),
            step_start_idx=branch, step_end_idx=num_steps,
        )

        x1_k = net.unnormalize(x_k_final)
        spec_k = feature_utils.decode(x1_k)
        audio_k = feature_utils.vocode(spec_k).float().cpu()[0]

        wav = audio_k.mean(0, keepdim=True)
        if wav.shape[-1] > 48000 * 10:
            wav = wav[:, :48000 * 10]
        with torch.no_grad():
            a_emb = clap_eval_model.get_audio_embedding_from_data(x=wav, use_tensor=True)
            t_emb = clap_eval_model.get_text_embedding([label_for_clap], use_tensor=True)
            score = torch.nn.functional.cosine_similarity(a_emb, t_emb).item()
        all_scores.append(score)
        log.info(f"[Beam]   branch {k}: CLAP={score:.4f}")

        if score > best_score:
            best_score = score
            best_audio = audio_k

    log.info(f"[Beam] best={best_score:.4f}  all={[f'{s:.3f}' for s in all_scores]}")
    return best_audio, best_score, all_scores


def main():
    setup_eval_logging()

    parser = ArgumentParser(description="FlowGuide: Training-Free Semantic Flow Guidance for V2A")
    parser.add_argument('--variant', type=str, default='large_44k_v2')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--negative_prompt', type=str, default='')
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default='./output_guided')
    parser.add_argument('--full_precision', action='store_true')

    # Guidance modules
    parser.add_argument('--use_cot', action='store_true', help='Enable CoT Enhanced Caption (Module 1)')
    parser.add_argument('--cot_model', type=str,
                        default='/data/models/huggingface/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5')
    parser.add_argument('--cot_device', type=str, default='cuda:1',
                        help='Device for MLLM (separate from MMAudio)')
    parser.add_argument('--use_flow_guidance', action='store_true',
                        help='Enable Semantic Flow Guidance (Module 2)')
    parser.add_argument('--guidance_strength', type=float, default=0.05)

    args = parser.parse_args()

    model: ModelConfig = all_model_cfg[args.variant]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    video_path = Path(args.video).expanduser()
    output_dir = args.output.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    # Load MMAudio
    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded MMAudio weights from {model.model_path}')

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=args.num_steps)

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    # Load video
    video_info = load_video(video_path, args.duration)
    clip_frames = video_info.clip_frames.unsqueeze(0)
    sync_frames = video_info.sync_frames.unsqueeze(0)
    duration = video_info.duration_sec

    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    # Initialize guidance modules & run CoT BEFORE inference_mode
    cot_caption = None
    if args.use_cot:
        cot_analyzer = CoTSceneAnalyzer(
            model_name=args.cot_model,
            device=args.cot_device,
            dtype=torch.float16,
        )
        log.info("[Module 1] Running CoT Scene Analysis (outside inference_mode)...")
        cot_result = cot_analyzer.analyze_video(str(video_path))
        cot_caption = cot_result.get("rich_caption", "")
        log.info(f"[Module 1] Rich caption: {cot_caption}")
        # Free MLLM memory
        del cot_analyzer
        torch.cuda.empty_cache()

    flow_guidance = None
    if args.use_flow_guidance:
        flow_guidance = SemanticFlowGuidance(
            guidance_strength=args.guidance_strength,
            device=device,
        )

    # --- Run baseline first ---
    log.info("=" * 60)
    log.info("Running BASELINE (standard MMAudio)...")
    log.info(f"Prompt: '{args.prompt}'")
    rng.manual_seed(args.seed)
    with torch.inference_mode():
        baseline_audio = guided_generate(
            clip_frames, sync_frames, [args.prompt],
            negative_text=[args.negative_prompt],
            feature_utils=feature_utils, net=net, fm=fm, rng=rng,
            cfg_strength=args.cfg_strength,
        )
    baseline_audio = baseline_audio.float().cpu()[0]
    baseline_path = output_dir / f'{video_path.stem}_baseline.flac'
    torchaudio.save(baseline_path, baseline_audio, seq_cfg.sampling_rate)
    log.info(f"Baseline saved: {baseline_path}")

    # --- Run guided version ---
    log.info("=" * 60)
    modules_str = []
    if args.use_cot:
        modules_str.append("CoT")
    if args.use_flow_guidance:
        modules_str.append("FlowGuide")
    guide_label = "+".join(modules_str) if modules_str else "NoGuide"
    log.info(f"Running GUIDED ({guide_label})...")

    # Use CoT caption as the prompt for guided run
    guided_prompt = cot_caption if cot_caption else args.prompt
    log.info(f"Guided prompt: '{guided_prompt}'")

    rng.manual_seed(args.seed)  # Same seed for fair comparison
    # Flow guidance needs gradients, so don't use inference_mode when it's enabled
    if flow_guidance is not None:
        with torch.no_grad():
            guided_audio = guided_generate(
                clip_frames, sync_frames, [guided_prompt],
                negative_text=[args.negative_prompt],
                feature_utils=feature_utils, net=net, fm=fm, rng=rng,
                cfg_strength=args.cfg_strength,
                flow_guidance=flow_guidance,
            )
    else:
        with torch.inference_mode():
            guided_audio = guided_generate(
                clip_frames, sync_frames, [guided_prompt],
                negative_text=[args.negative_prompt],
                feature_utils=feature_utils, net=net, fm=fm, rng=rng,
                cfg_strength=args.cfg_strength,
                flow_guidance=flow_guidance,
            )
    guided_audio = guided_audio.float().cpu()[0]
    guided_path = output_dir / f'{video_path.stem}_guided_{guide_label}.flac'
    torchaudio.save(guided_path, guided_audio, seq_cfg.sampling_rate)
    log.info(f"Guided saved: {guided_path}")

    log.info("=" * 60)
    log.info(f"Compare: {baseline_path} vs {guided_path}")
    log.info(f'Memory usage: {torch.cuda.max_memory_allocated() / (2**30):.2f} GB')


if __name__ == '__main__':
    main()
