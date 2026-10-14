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

## Runs

Use the same seed, data, epochs and `img_size` as the baseline.

| Run | SIGReg | BatchNorm | `subdir` |
|---|---|---|---|
| A (baseline) | on | on | existing baseline, or `abl_A_base_s3072` |
| B | off | on | `abl_B_nosigreg_s3072` |

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

Use the same wandb flags as the baseline (see `docs/training.md`).

```bash
python train.py data=pusht num_workers=16 \
    wandb.enabled=True wandb.config.entity=<entity> wandb.config.project=lewm \
    loss.sigreg.weight=0 \
    subdir=abl_B_nosigreg_s3072
```

Look at the wandb curves early (first 1 to 3 epochs). If `pred_loss` goes
to about 0 and `sigreg_loss` goes up a lot, the model has collapsed. You
can stop the run early to save money. Record the epoch where this occurs.

### 3. Eval

Evaluate the same epochs as the baseline:

```bash
for e in 010 050 100; do
  python eval.py --config-name=pusht.yaml policy=pusht/abl_B_nosigreg_s3072/weights_epoch_$e.pt eval.img_size=112
done
```

If the run was stopped early, evaluate the last saved epoch.

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

| Run | final `pred_loss` | final `sigreg_loss` | eff. rank | success @10 | @50 | @100 |
|---|---|---|---|---|---|---|
| A | | | | | | |
| B | | | | | | |

## Expected results

Collapse is likely: `pred_loss` near 0, `sigreg_loss` high, low effective
rank, and success near the random policy.

## Risks

- A collapsed model can still give a non-zero success rate by chance. Compare
  with `policy=random` (see `docs/evaluation.md`).
