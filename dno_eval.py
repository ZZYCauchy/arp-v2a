#!/usr/bin/env python3
"""Generate DNO-V2A audio for a video list.

Mirrors ``vfree_eval.py`` layout. Wraps ``guided_generate_dno`` and supports
all three DNO variants (A/B/C), K_inner sweep, lambda_prior tuning, and an
optional ablation flag to disable the final flow guidance pass (isolates the
noise-only contribution).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch
import torchaudio

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mmaudio.eval_utils import ModelConfig, all_model_cfg, load_video, setup_eval_logging  # noqa: E402
from mmaudio.model.flow_matching import FlowMatching  # noqa: E402
from mmaudio.model.networks import get_my_mmaudio  # noqa: E402
from mmaudio.model.utils.features_utils import FeaturesUtils  # noqa: E402
from guided_inference import SemanticFlowGuidance, guided_generate_dno  # noqa: E402

log = logging.getLogger(__name__)


def load_items(video_list: Path, test_dir: Path, label_file: Path | None, max_samples: int):
    """Load (vid_id, label, video path). Inline ``vid,label`` rows use the CSV label;
    VGGSound-style ``video_list_key`` rows still override from ``label_file`` when present."""
    rows: list[tuple[str, str]] = []
    with open(video_list) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                vid, lab = line.split(",", 1)
                rows.append((vid.strip(), lab.strip()))
            else:
                rows.append((line.strip(), ""))
    rows = rows[:max_samples]
    labels = {}
    if label_file and label_file.exists():
        with open(label_file) as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    key = f"{row[0].strip()}_{int(row[1].strip()):06d}"
                    labels[key] = row[2].strip()
    out = []
    for v, inline in rows:
        out.append({
            "vid_id": v,
            "label": labels.get(v, inline),
            "video": test_dir / f"{v}.mp4",
        })
    return out


def _pad_frames_time(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Pad or trim along first dim (time) to target_T (MMAudio expects full clip)."""
    if x.shape[0] >= target_T:
        return x[:target_T]
    pad = target_T - x.shape[0]
    tail = x[-1:].expand(pad, *x.shape[1:])
    return torch.cat([x, tail], dim=0)


def main():
    setup_eval_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", type=str, default="/data/datasets/vggsound/test-videos")
    p.add_argument("--label_file", type=str, default="/data/datasets/vggsound.csv")
    p.add_argument("--video_list", type=str, required=True)
    p.add_argument("--cot_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="/data/test_output/dno_runs")
    p.add_argument("--method_name", type=str, default="",
                   help="Subdir under output_dir (default: auto from DNO params)")
    p.add_argument("--variant", type=str, default="large_44k_v2")
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--clap_device", type=str, default="cuda:1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_strength", type=float, default=4.5)
    p.add_argument("--guidance_strength", type=float, default=0.1)
    p.add_argument("--neg_weight", type=float, default=0.3)
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--dno_variant", type=str, default="A", choices=["A", "B", "C"])
    p.add_argument("--dno_kinner", type=int, default=3)
    p.add_argument("--dno_lr", type=float, default=0.05)
    p.add_argument("--dno_lambda_prior", type=float, default=0.1)
    p.add_argument("--dno_disable_final_guidance", action="store_true")
    p.add_argument("--dno_sync_weight", type=float, default=0.0,
                   help="Weight for TCG sync reward in DNO inner loop (0=off)")
    p.add_argument("--dno_ib_weight", type=float, default=0.0,
                   help="Weight for ImageBind AV reward in DNO inner loop (0=off)")
    p.add_argument("--tcg_n_bins", type=int, default=8,
                   help="Number of time bins for TCG envelope correlation")
    p.add_argument("--save_history", action="store_true",
                   help="Save per-video DNO inner-loop history JSON")
    p.add_argument("--arp_mode", type=str, default="off", choices=["off", "closed", "joint"],
                   help="Adaptive Richness Prior mode: blend raw + CoT in CLIP+CLAP spaces")
    p.add_argument("--arp_beta", type=float, default=5.0,
                   help="Sigmoid temperature for closed-form alpha = sigmoid(beta * (s_cot - s_raw))")
    p.add_argument("--arp_lr", type=float, default=0.01,
                   help="Adam LR for the joint-mode alpha logit (theta)")
    p.add_argument("--arp_lambda", type=float, default=0.1,
                   help="Quadratic anchor on (alpha - alpha_init) in joint mode")
    args = p.parse_args()

    test_dir = Path(args.test_dir)
    out_root = Path(args.output_dir)
    if args.method_name:
        method = args.method_name
    else:
        nofg = "_nofg" if args.dno_disable_final_guidance else ""
        sw = f"_sw{args.dno_sync_weight:g}" if args.dno_sync_weight > 0 else ""
        ibw = f"_ibw{args.dno_ib_weight:g}" if args.dno_ib_weight > 0 else ""
        if args.arp_mode == "closed":
            arp_suf = f"_arpC_b{args.arp_beta:g}"
        elif args.arp_mode == "joint":
            arp_suf = f"_arpJ_b{args.arp_beta:g}_lr{args.arp_lr:g}_lp{args.arp_lambda:g}"
        else:
            arp_suf = ""
        method = (
            f"dno_{args.dno_variant}_K{args.dno_kinner}"
            f"_lr{args.dno_lr:g}_lp{args.dno_lambda_prior:g}{sw}{ibw}{arp_suf}{nofg}"
        )
    out_dir = out_root / method
    out_dir.mkdir(parents=True, exist_ok=True)
    history_dir = out_dir / "_history"
    if args.save_history:
        history_dir.mkdir(exist_ok=True)

    with open(args.cot_json) as f:
        cot_data = json.load(f)

    items = load_items(Path(args.video_list), test_dir, Path(args.label_file), args.max_samples)

    device = args.device
    dtype = torch.bfloat16
    model_cfg: ModelConfig = all_model_cfg[args.variant]
    model_cfg.download_if_needed()
    seq_cfg = model_cfg.seq_cfg

    net = get_my_mmaudio(model_cfg.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model_cfg.model_path, map_location=device, weights_only=True))
    rng = torch.Generator(device=device)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=args.num_steps)

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model_cfg.vae_path,
        synchformer_ckpt=model_cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=model_cfg.mode,
        bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
        need_vae_encoder=False,
    ).to(device, dtype).eval()

    flow_guidance = SemanticFlowGuidance(
        guidance_strength=args.guidance_strength,
        guidance_start_ratio=0.3,
        guidance_end_ratio=0.9,
        guidance_interval=2,
        device=args.clap_device,
    )
    # Force CLAP load now (DNO needs it).
    flow_guidance._init_clap()

    arp_ib = None
    arp_records = []
    if args.arp_mode != "off":
        from guided_inference import ImageBindGuidance
        arp_ib = ImageBindGuidance(device=args.device)
        arp_ib._init()
        log.info(
            f"[ARP] mode={args.arp_mode} beta={args.arp_beta} "
            f"lr={args.arp_lr} lambda={args.arp_lambda}"
        )

    for i, item in enumerate(items):
        vid_id = item["vid_id"]
        label = item["label"]
        vp = item["video"]
        save_p = out_dir / f"{vid_id}.flac"
        if save_p.exists() and save_p.stat().st_size > 0:
            log.info("[%s/%s] skip %s", i + 1, len(items), vid_id)
            continue
        cd = cot_data.get(vid_id, {})
        if isinstance(cd, dict) and "rich_caption" in cd:
            cot_caption = cd["rich_caption"]
            neg_text = cd.get("neg_text", "") or ""
        else:
            cot_caption = label
            neg_text = ""

        log.info("[%s/%s] %s", i + 1, len(items), vid_id)
        vinfo = load_video(vp, duration_sec=args.duration, load_all_frames=False)
        need_clip = seq_cfg.clip_seq_len
        need_sync = int(seq_cfg.duration * seq_cfg.sync_frame_rate)
        clip_frames = _pad_frames_time(vinfo.clip_frames, need_clip).unsqueeze(0)
        sync_frames = _pad_frames_time(vinfo.sync_frames, need_sync).unsqueeze(0)

        flow_guidance._text_embed = None
        flow_guidance._neg_text_embed = None
        flow_guidance._arp_e_raw = None
        flow_guidance._arp_e_cot = None
        flow_guidance._arp_alpha = None

        arp_alpha = None
        if args.arp_mode != "off":
            from arp import compute_alpha_from_ib
            arp_ib._video_embed = None
            arp_ib.precompute_video_embed(str(vp))
            arp_alpha, s_raw, s_cot = compute_alpha_from_ib(
                arp_ib, label, cot_caption, beta=args.arp_beta)
            log.info(
                f"  [ARP] s_raw={s_raw:.4f} s_cot={s_cot:.4f} "
                f"diff={s_cot - s_raw:+.4f} alpha={arp_alpha:.3f}"
            )
            arp_records.append({
                "vid_id": vid_id,
                "raw": label,
                "cot": cot_caption,
                "s_raw": s_raw,
                "s_cot": s_cot,
                "alpha_init": arp_alpha,
            })
            flow_guidance.precompute_text_embed_blend(label, cot_caption, arp_alpha)
        else:
            flow_guidance.precompute_text_embed(cot_caption)
        if neg_text:
            flow_guidance.precompute_negative_embed(neg_text)

        rng.manual_seed(args.seed)

        cfg_neg = [neg_text] if neg_text else None

        with torch.no_grad():
            audio, history = guided_generate_dno(
                clip_frames,
                sync_frames,
                [cot_caption],
                negative_text=cfg_neg,
                feature_utils=feature_utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=args.cfg_strength,
                flow_guidance=flow_guidance,
                neg_weight=args.neg_weight,
                dno_variant=args.dno_variant,
                dno_kinner=args.dno_kinner,
                dno_lr=args.dno_lr,
                dno_lambda_prior=args.dno_lambda_prior,
                dno_neg_weight=args.neg_weight,
                dno_sync_weight=args.dno_sync_weight,
                dno_ib_weight=args.dno_ib_weight,
                tcg_n_bins=args.tcg_n_bins,
                video_path=str(vp),
                dno_disable_final_guidance=args.dno_disable_final_guidance,
                arp_mode=args.arp_mode,
                arp_alpha=arp_alpha,
                arp_raw_text=label if args.arp_mode != "off" else None,
                arp_cot_text=cot_caption if args.arp_mode != "off" else None,
                arp_lr=args.arp_lr,
                arp_lambda=args.arp_lambda,
            )

        wav = audio.float().cpu()[0]
        torchaudio.save(save_p, wav, seq_cfg.sampling_rate)
        if args.save_history:
            with open(history_dir / f"{vid_id}.json", "w") as f:
                json.dump(history, f, indent=2)

    meta = {
        "method": method,
        "n_items": len(items),
        "dno_variant": args.dno_variant,
        "dno_kinner": args.dno_kinner,
        "dno_lr": args.dno_lr,
        "dno_lambda_prior": args.dno_lambda_prior,
        "dno_disable_final_guidance": bool(args.dno_disable_final_guidance),
        "dno_sync_weight": args.dno_sync_weight,
        "dno_ib_weight": args.dno_ib_weight,
        "arp_mode": args.arp_mode,
        "arp_beta": args.arp_beta,
        "arp_lr": args.arp_lr,
        "arp_lambda": args.arp_lambda,
        "video_list": str(args.video_list),
    }
    with open(out_root / f"{method}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if arp_records:
        with open(out_dir / "_arp_alphas.json", "w") as f:
            json.dump(arp_records, f, indent=2)
        alphas = [r["alpha_init"] for r in arp_records]
        if alphas:
            log.info(
                f"[ARP] alpha stats: n={len(alphas)} "
                f"mean={sum(alphas)/len(alphas):.3f} "
                f"min={min(alphas):.3f} max={max(alphas):.3f}"
            )
    log.info("done -> %s", out_dir)


if __name__ == "__main__":
    main()
