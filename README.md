# Adaptive Richness Prior (ARP) for Training-Free Video-to-Audio Generation

A minimal runnable implementation built on the frozen [MMAudio](https://github.com/hkchengrex/MMAudio) backbone (`large_44k_v2`). This repository contains the inference code used for the main results in the paper: per-video prompt blending with ARP and initial noise optimization with DNO. Model weights are not included.

The `mmaudio/` directory and `LICENSE` are derived from MMAudio (MIT License, Copyright (c) 2024 Sony Research Inc.).

## Configuration Used in the Paper

The main results use the following configuration:

- Backbone: `large_44k_v2`
- ARP-Closed: `beta=10`
- DNO: variant B, `K=2`, Adam learning rate `0.05`
- Reward weights: CLAP `1.0`, ImageBind `0.3`, synchronization `1.0`
- Sampling: 25 steps, CFG `4.5`, random seed `42`

## External Dependencies

Use the default locations below or specify the paths through the corresponding environment variables:

| Dependency | Default location | Environment variable or download behavior |
| --- | --- | --- |
| MMAudio weights | `./weights/`, `./ext_weights/` | Downloaded from MMAudio's public links on the first run |
| CLAP checkpoint | `/data/models/clap/630k-audioset-best.pt` | `CLAP_CKPT` |
| ImageBind source code | `/data/repos/Seeing-and-Hearing/v2a/imagebind/imagebind` | `IMAGEBIND_ROOT` |
| ImageBind checkpoint | `/data/models/imagebind_huge.pth` | `IMAGEBIND_CKPT` |

Download the CLAP checkpoint from [LAION CLAP 630k](https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-best.pt) and the ImageBind checkpoint from [imagebind_huge.pth](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth).

## Installation

```bash
pip install -r requirements.txt
```

ImageBind source code must also be available locally and importable. Set `IMAGEBIND_ROOT` to the directory containing `models/imagebind_model.py`. Run the commands from the repository root, because ImageBind reads `bpe/bpe_simple_vocab_16e6.txt.gz` relative to the current working directory.

## Run a Single Sample

Prepare two input files:

- `videos.txt`: one `video_id,short_label` entry per line. Each video must be located at `{test_dir}/{video_id}.mp4`.
- `cot.json`: a JSON object in the format `{"video_id": {"rich_caption": "...", "neg_text": "..."}}`.

```bash
python dno_eval.py \
  --test_dir /path/to/videos \
  --video_list videos.txt \
  --cot_json cot.json \
  --output_dir outputs \
  --max_samples 1 \
  --device cuda:0 \
  --clap_device cuda:1 \
  --variant large_44k_v2 \
  --num_steps 25 \
  --cfg_strength 4.5 \
  --seed 42 \
  --duration 8 \
  --dno_variant B \
  --dno_kinner 2 \
  --dno_lr 0.05 \
  --dno_lambda_prior 0.1 \
  --dno_sync_weight 1.0 \
  --dno_ib_weight 0.3 \
  --arp_mode closed \
  --arp_beta 10
```

Generated audio is saved to `outputs/<method>/<video_id>.flac`. To use ARP-Joint, replace `--arp_mode closed` with `--arp_mode joint`.

Two GPUs are recommended: one for MMAudio and one for CLAP. A single GPU with 40 GB of VRAM may run out of memory during backpropagation.
