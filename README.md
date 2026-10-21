# Adaptive Richness Prior (ARP) for training-free video-to-audio

最小可运行实现。骨干是冻结的 [MMAudio](https://github.com/hkchengrex/MMAudio)（`large_44k_v2`）。本仓库只保留论文主结果用到的推理代码：逐视频提示词融合（ARP）和初始噪声优化（DNO）。权重不随仓库分发。

`mmaudio/` 与 `LICENSE` 来自 MMAudio（MIT，Copyright (c) 2024 Sony Research Inc.）。

## 论文设置

主结果对应下面这组参数：

- 骨干：`large_44k_v2`
- ARP-Closed，`beta=10`
- DNO：变体 B，`K=2`，Adam 学习率 `0.05`
- 奖励权重：CLAP `1.0`，ImageBind `0.3`，同步项 `1.0`
- 采样：25 步，CFG `4.5`，种子 `42`

## 外部依赖

把下列文件放到仓库根目录，或通过环境变量指定：

| 用途 | 默认路径 | 环境变量 |
| --- | --- | --- |
| MMAudio 权重 | `./weights/`、`./ext_weights/` | 首次运行会按 MMAudio 的公开链接下载 |
| CLAP | `/data/models/clap/630k-audioset-best.pt` | `CLAP_CKPT` |
| ImageBind 代码 | `/data/repos/Seeing-and-Hearing/v2a/imagebind/imagebind` | `IMAGEBIND_ROOT` |
| ImageBind 权重 | `/data/models/imagebind_huge.pth` | `IMAGEBIND_CKPT` |

CLAP 权重可用 [LAION CLAP 630k](https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-best.pt)。ImageBind 权重可用 [imagebind_huge.pth](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth)。

## 安装

```bash
pip install -r requirements.txt
```

还需要本机可导入的 ImageBind 源码，并把 `IMAGEBIND_ROOT` 指到含 `models/imagebind_model.py` 的目录。ImageBind 会从当前工作目录读取 `bpe/bpe_simple_vocab_16e6.txt.gz`，因此请在本仓库根目录运行。

## 跑一条样本

准备两份输入：

- `videos.txt`：每行 `视频ID,短标签`，视频文件为 `{test_dir}/{视频ID}.mp4`
- `cot.json`：`{视频ID: {"rich_caption": "...", "neg_text": "..."}}`

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

成功后会写出 `outputs/<method>/<视频ID>.flac`。把 `--arp_mode closed` 换成 `joint` 即 ARP-Joint；此时 `--arp_lr 0.05 --arp_lambda 0.1` 与论文消融一致。

显存建议两张卡：一张放 MMAudio，一张放 CLAP。单卡 40GB 容易在反向传播时不够。
