# Training

This guide covers training LeWM on PushT: picking a GPU, launching a run on a cloud machine, where checkpoints go, and how to evaluate them. For dataset setup, see [datasets.md](datasets.md).

## Workload

With the default config (`config/train/lewm.yaml` + `config/train/data/pusht.yaml`):

| Setting | Value |
|---|---|
| Model | ~15M parameters, ViT-tiny encoder (patch 14) |
| Image size | `img_size: 112` (`224` is the paper setting) |
| Frames per sample | 4 (`history_size: 3` + `num_preds: 1`), `frameskip: 5` |
| Batch size | 128 → 512 images per step |
| Precision | bf16 |
| Epochs | 100 |

The model is small, so GPU memory is not the limit. The usual bottleneck is **data loading**: decoding and preprocessing 512 frames per step on the CPU.

## Choosing a GPU

| GPU | Verdict |
|---|---|
| **L40S 48GB** or **A100 40/80GB** | **Recommended.** Good bf16 speed, and these instances usually have many CPU cores. |
| H100 | Faster, but the data loader will probably hold it back. Not worth the extra cost. |
| RTX 4090 / A10G / L4 (24GB) | Work fine. L4 is roughly 2–3× slower. |

CPU matters as much as the GPU:

- Pick an instance with **16+ vCPUs**, and raise `num_workers` from the default 6.
- Keep the dataset on **local NVMe**, not a network volume.

A single L40S or A100 costs roughly $1–2/hr on RunPod, Lambda, or Vast.ai.

## How long it takes

These are estimates, not measurements (based on the paper's "a few hours on a single GPU"):

| GPU | 100 epochs |
|---|---|
| A100 / L40S | ~3–8 hours |
| RTX 4090 / A10G | similar to slightly slower |
| L4 | roughly 2–3× longer |

To get a real number for your run:

1. After a few minutes, read `it/s` and the steps per epoch from the progress bar.
2. **Total time ≈ (steps per epoch ÷ it/s) × epochs.**
3. Check `nvidia-smi`. If GPU use stays below ~80%, data loading is the bottleneck: raise `num_workers` or use a machine with more CPUs.

## Launching a run on a cloud machine

### 1. Set up the code

```bash
git clone https://github.com/taot/le-wm.git le-wm && cd le-wm
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync && source .venv/bin/activate
```

### 2. Log in and set storage paths

```bash
hf auth login
wandb login
cp .env.example .env
```

In `.env`, set `STABLEWM_HOME` to the instance's local NVMe disk, e.g. `export STABLEWM_HOME=/workspace/swm`. `train.py` reads it automatically. Run `source .env` to use `$STABLEWM_HOME` in shell commands too.

### 3. Download the dataset

Download it yourself before training. See [datasets.md](datasets.md#download) for the command and why it is more reliable than the automatic download.

### 4. Smoke test

Before a long paid run, check that the dataset, login, and wandb all work:

```bash
python train.py data=pusht trainer.max_epochs=1 +trainer.limit_train_batches=50
```

`limit_train_batches` is not in the config, so it needs the `+` prefix.

### 5. Full run

Start a `tmux` session, so training keeps going if you disconnect:

```bash
tmux new -s train
```

Then start training:

```bash
python train.py data=pusht num_workers=16 \
    wandb.enabled=True wandb.config.entity=<your_entity> wandb.config.project=lewm
```

**Why the wandb settings go on the command line:** `config/train/launcher/local.yaml` sets `wandb.enabled: False` and `entity: lewm`, and it is applied after `lewm.yaml` (because `_self_` comes first in the `defaults` list). So wandb values written in `lewm.yaml` are overwritten. Pass them on the command line, or move `_self_` to the end of the `defaults` list.

## Checkpoints

Each run writes one folder, `$STABLEWM_HOME/checkpoints/<env>/<subdir>/`:

- `env` comes from the data config (`pusht`, `tworoom`, `reacher`, `cube`).
- `subdir` defaults to `<date>_<time>_img<img_size>_s<seed>`.

| File | Contents |
|---|---|
| `config.yaml` | full training config |
| `config.json` | model config, needed to load the weights |
| `weights_epoch_NNN.pt` | model weights after each epoch |
| `lewm_weights.ckpt` | full training state, used to resume |

### Back up to a Hugging Face bucket

Add `bucket.enabled=True` to mirror the run folder to `hf://buckets/librakevin/lewm-checkpoints/<env>/<subdir>/`. It syncs in the background every `bucket.every_n_epochs` epochs (default 5), and once more when training ends or crashes. It uses your `hf auth login`.

To resume on a new machine, pull the run folder, then pass the same `subdir`:

```bash
hf buckets sync hf://buckets/librakevin/lewm-checkpoints/pusht/<subdir> $STABLEWM_HOME/checkpoints/pusht/<subdir>
python train.py data=pusht subdir=<subdir> bucket.enabled=True
```

The same `hf buckets sync` command copies a run to your own machine, e.g. to use it in the playground.

## Evaluating a run

Pass `policy=` as a weights file, relative to `$STABLEWM_HOME/checkpoints/`:

```bash
python eval.py --config-name=pusht.yaml policy=pusht/<subdir>/weights_epoch_100.pt eval.img_size=112
```

- `--config-name=pusht.yaml`: start from a dataset state; the goal is the expert's state 25 steps later.
- `--config-name=pusht_full.yaml`: random start; the goal is the block on the green T.
- `eval.img_size` must match the `img_size` the run was trained with (the `img112` in the folder name).

For the full steps (getting a run from the bucket, overrides, where the results go: the run's `eval/` folder), see [evaluation.md](evaluation.md).

For pretrained checkpoints from the paper, see [checkpoints.md](checkpoints.md). For the baseline training scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.
