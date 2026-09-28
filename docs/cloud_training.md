# Training on a cloud GPU

This guide covers choosing a cloud GPU for LeWM, estimating training time on PushT, and launching a run.

> **Note:** The time figures below are estimates, not measurements. They are based on the paper's claim ("trainable on a single GPU in a few hours") and the default config. To get a real number, measure throughput during the first few minutes of your run (see [Measuring your actual time](#measuring-your-actual-time)).

## Workload

With the default config (`config/train/lewm.yaml` + `config/train/data/pusht.yaml`):

| Setting | Value |
|---|---|
| Model | ~15M parameters, ViT-tiny encoder (patch 14) |
| Image size | 224 × 224 |
| Frames per sample | 4 (`history_size: 3` + `num_preds: 1`), `frameskip: 5` |
| Batch size | 128 → 512 images per step |
| Precision | bf16 |
| Epochs | 100 |

The model is small, so GPU memory is not the constraint. The likely bottleneck is **data loading**: decoding and preprocessing 512 frames per step on the CPU.

## Choosing a GPU

| GPU | Verdict |
|---|---|
| **L40S 48GB** or **A100 40/80GB** | **Recommended.** Good bf16 throughput, and cloud instances usually come with many CPU cores. |
| H100 | Faster, but the data loader will probably hold it back. Not worth the extra cost for a model this size. |
| RTX 4090 / A10G / L4 (24GB) | Work fine. L4 is roughly 2–3× slower. |

**Prioritize CPU over GPU:**

- Choose an instance with **16+ vCPUs**, and raise `num_workers` from the default 6.
- Keep the dataset on **local NVMe**, not a network volume.

Single L40S or A100 instances cost roughly $1–2/hr on RunPod, Lambda, or Vast.ai.

## Time estimate for PushT

| GPU | Estimated time (100 epochs) |
|---|---|
| A100 / L40S | ~3–8 hours |
| RTX 4090 / A10G | similar to slightly slower |
| L4 | roughly 2–3× longer |

The range is wide because data-loading speed (CPU cores, disk) matters as much as the GPU.

### Measuring your actual time

1. After a few minutes of training, read `it/s` from the Lightning progress bar. The progress bar also shows the number of steps per epoch.
2. Compute: **total time ≈ (steps per epoch ÷ it/s) × 100 epochs**.
3. Check `nvidia-smi`. If GPU utilization stays below ~80%, data loading is the bottleneck: increase `num_workers` or use a machine with more CPUs.

## Launching a run

### 1. Get the code onto the machine

Push your local changes to a fork (or `rsync` the repo), then:

```bash
git clone https://github.com/taot/le-wm.git le-wm && cd le-wm
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync && source .venv/bin/activate
```

### 2. Authenticate and set storage paths

```bash
hf auth login
wandb login
export STABLEWM_HOME=/workspace/swm
export LOCAL_DATASET_DIR=/workspace/swm
```

Point both paths at the instance's local NVMe disk.

### 3. Download the dataset

`train.py` downloads `librakevin/lewm-pusht` on first use, but downloading it yourself first is more reliable. `hf download` runs files in parallel and can resume, while the built-in loader fetches one file at a time. If the built-in download is interrupted, the loader later treats the partial folder as a complete cached dataset, because it only checks that the folder isn't empty.

Download the dataset into the folder the loader checks, `$LOCAL_DATASET_DIR/datasets/<user>--<repo>/`:

```bash
hf download librakevin/lewm-pusht --repo-type dataset \
  --local-dir $LOCAL_DATASET_DIR/datasets/librakevin--lewm-pusht
```

If the download stops partway, run the same command again to resume it. Afterward, `ls $LOCAL_DATASET_DIR/datasets/librakevin--lewm-pusht` should list `pusht_expert_train.lance/`.

### 4. Smoke test

Before starting a long paid run, check that the dataset, auth, and wandb all work:

```bash
python train.py data=pusht trainer.max_epochs=1 +trainer.limit_train_batches=50
```

`limit_train_batches` is not in the config, so it needs the `+` prefix.

### 5. Full run

Start `tmux` session:

```
tmux new -s train
```

Start training:

```bash
python train.py \
    wandb.enabled=True \
    wandb.config.entity=librakevin-workday \
    wandb.config.project=lewm \
    data=pusht \
    num_workers=16 \
    trainer.max_epochs=3
```

**Why the wandb flags are needed:** In the `defaults` list of `lewm.yaml`, `_self_` comes before `launcher: local`. Because Hydra applies defaults in order, `launcher/local.yaml` overrides the `wandb` section in `lewm.yaml` with `enabled: False` and `entity: lewm`. Either pass the wandb settings on the command line as shown, or move `_self_` to the end of the defaults list.

### 6. Save the checkpoints

Each run writes one folder, `$STABLEWM_HOME/checkpoints/<env>/<subdir>/`:

- `env` comes from the data config (`pusht`, `tworoom`, `reacher`, `cube`).
- `subdir` defaults to `<date>_<time>_img<img_size>_s<seed>`.

| File | Contents |
|---|---|
| `config.yaml` | full training config |
| `config.json` | model config, needed by `load_pretrained` |
| `weights_epoch_NNN.pt` | model weights after each epoch |
| `lewm_weights.ckpt` | full training state, used to resume |

Add `bucket.enabled=True` to mirror the folder to `hf://buckets/librakevin/lewm-checkpoints/<env>/<subdir>/`. The sync runs in the background every `bucket.every_n_epochs` epochs (default 5), and once more when training ends or crashes. It needs the `hf auth login` from step 2.

```bash
python train.py data=pusht num_workers=16 bucket.enabled=True wandb.enabled=True ...
```

To resume on a new instance, pull the run folder, then pass the same `subdir`:

```bash
hf buckets sync hf://buckets/librakevin/lewm-checkpoints/pusht/<subdir> $STABLEWM_HOME/checkpoints/pusht/<subdir>
python train.py data=pusht subdir=<subdir> bucket.enabled=True ...
```

To evaluate, point `policy` at a weights file relative to `$STABLEWM_HOME/checkpoints`, for example `python eval.py policy=pusht/<subdir>/weights_epoch_100.pt`.
