# Ablation: remove SIGReg

Status: plan, not started.

Related plan: `plans/ablation_no_batchnorm.md`.

## Goal

Find how much SIGReg adds to planning success. SIGReg is the loss term
that pushes the embeddings to N(0, I), so that they do not collapse.

## Background

The total loss is in `train.py:55`:

```python
output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
```

`lambd` is `loss.sigreg.weight` in `config/train/lewm.yaml` (default 0.09).

`SIGReg` (`module.py:11`) has no learned parameters. It has only buffers
(`t`, `phi`, `weights`). So with `weight=0` it sends no gradient, and the
model trains as if SIGReg is not there.

`sigreg_loss` is still calculated and logged when `weight=0`. Use it as a
collapse monitor.

## No code change is necessary

| Change | Override |
|---|---|
| Remove SIGReg | `loss.sigreg.weight=0` |

## Baseline (run A)

Run A is an existing run. Do not train it again:

- HF bucket: `librakevin/lewm-checkpoints`, folder
  `pusht/2026-09-28_114752_img112_s3072`
  (<https://huggingface.co/buckets/librakevin/lewm-checkpoints/tree/pusht/2026-09-28_114752_img112_s3072>)
- Files: `config.yaml`, `config.json`, `weights_epoch_001.pt` to
  `weights_epoch_003.pt`.

Its `config.yaml` is the same as the config from
`launcher=runpod trainer.max_epochs=3`:

| Key | Baseline value | Where it comes from |
|---|---|---|
| `trainer.max_epochs` | **3** | command line (the default in `lewm.yaml` is different) |
| `num_workers` | 16 | `launcher/runpod.yaml` |
| `wandb.enabled` | true | `launcher/runpod.yaml` |
| `wandb.config.entity` | `librakevin-workday` | `launcher/runpod.yaml` |
| `wandb.config.project` | `lewm` | `launcher/runpod.yaml` |

All other values (seed 3072, `img_size` 112, batch size 128, lr 5e-5,
weight decay 1e-3, bf16, gradient clip 1.0, SIGReg knots 17 and
`num_proj` 1024, the data, and the model with BatchNorm) come from
`lewm.yaml` and are the same. I checked this on 2026-10-14 with a diff of
the bucket `config.yaml` against
`python train.py --cfg job launcher=runpod data=pusht trainer.max_epochs=3`.
The only other differences are bookkeeping (`subdir`, `run_name`,
`wandb.config.name`, `data.env`, and `action_encoder.input_dim`, which
`train.py` sets at runtime to 10).

## Runs

Run B uses the baseline config. The only change is `loss.sigreg.weight=0`.

| Run | SIGReg | BatchNorm | Epochs | Location |
|---|---|---|---|---|
| A (baseline) | on (0.09) | on | 3 | HF bucket, `pusht/2026-09-28_114752_img112_s3072` |
| B | off (0) | on | 3 | `pusht/<date>_<time>_img112_s3072_nosigreg` |

If time permits: repeat with a second seed (for example 3073) to see the
noise between seeds.

## Steps

### 1. Smoke test (local or RunPod, a few minutes)

```bash
python train.py data=pusht trainer.max_epochs=1 +trainer.limit_train_batches=50 \
    loss.sigreg.weight=0 \
    subdir=abl_smoke_nosigreg
```

Check:

- The run starts with no error.
- `train/sigreg_loss` is logged.
- `train/loss == train/pred_loss`.

Then delete the `abl_smoke_nosigreg` checkpoint folder.

### 2. Full run (RunPod)

Same config as run A, only with `loss.sigreg.weight=0`:

```bash
python train.py launcher=runpod data=pusht trainer.max_epochs=3 \
    loss.sigreg.weight=0 \
    'subdir=${now:%Y-%m-%d_%H%M%S}_img${img_size}_s${seed}_nosigreg'
```

**Run names:** each run folder has the start date and time, like the
baseline. The `subdir` override above adds a label at the end, for example
`pusht/2026-10-14_063811_img112_s3072_nosigreg`. Keep the single quotes, so
that the shell does not expand `${...}`. Hydra fills in the values. The
date and time are not known before the run starts, so read the real folder
name from the first lines of the training log (or from wandb) and write it
in the runs table. To resume a run, pass that real name: `subdir=<name>`.

Before the run starts, check the config: add `--cfg job` to the command
above, and diff the output against the baseline `config.yaml`. The only
differences must be `loss.sigreg.weight` and the bookkeeping keys.

Look at the wandb curves during the run. If `pred_loss` goes to about 0 and
`sigreg_loss` goes up a lot, the model has collapsed. Record the step where
this occurs. Do not stop the run early: with only 3 epochs, all 3
checkpoints are necessary for the comparison with A.

### 3. Eval

Evaluate all 3 epochs of A and B with the same eval config. If the
baseline eval numbers were made with a different eval config, run A again
too.

```bash
B=<real subdir of run B>   # for example 2026-10-14_063811_img112_s3072_nosigreg
for r in 2026-09-28_114752_img112_s3072 $B; do
  for e in 001 002 003; do
    python eval.py --config-name=pusht.yaml policy=pusht/$r/weights_epoch_$e.pt eval.img_size=112
  done
done
```

To evaluate A, first download its folder from the HF bucket to
`$STABLEWM_HOME/checkpoints/pusht/2026-09-28_114752_img112_s3072/`.

### 4. Collapse metrics (optional, needs a small new script)

The losses alone do not show dimensional collapse. Add a script (for example
`scripts/embed_stats.py`) that loads a checkpoint, encodes about 2000
validation frames, and reports:

- the mean per-dimension std of `emb`,
- the effective rank of `emb` (exp of the entropy of the normalized singular
  values),
- the number of singular values above 1% of the largest.

Run it on the last epoch of A and B. The BatchNorm plan uses the same script.

## What to record

| Run | final `pred_loss` | final `sigreg_loss` | eff. rank | success @1 | @2 | @3 |
|---|---|---|---|---|---|---|
| A | | | | | | |
| B | | | | | | |

## Expected results

Collapse is likely: `pred_loss` near 0, `sigreg_loss` high, low effective
rank, and success near the random policy.

## Risks

- A collapsed model can still give a non-zero success rate by chance. Compare
  with `policy=random` (see `docs/evaluation.md`).
