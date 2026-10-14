# Ablation: remove BatchNorm

Status: plan, not started.

Related plan: `plans/ablation_no_sigreg.md`. Run D below needs run B from
that plan for comparison.

## Goal

Find how much the BatchNorm in `projector` and `pred_proj` adds to planning
success, with and without SIGReg.

## Background

The BatchNorm is set in `config/train/model/lewm.yaml`:

```yaml
projector:
  _target_: module.MLP
  ...
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true
```

`pred_proj` has the same setting. `MLP` (`module.py:224`) puts `norm_fn` on
the **hidden** layer only: `Linear -> norm_fn -> GELU -> Linear`. If
`norm_fn` is `None`, `MLP` uses `nn.Identity()`.

Note: the BatchNorm is not on the output. The last `Linear` comes after it.
So BatchNorm does not force the output embedding to have unit variance. The
last `Linear` can still map everything to a constant. BatchNorm can still
change the training dynamics, because it couples the samples in a batch.
This ablation tells us if that has an effect.

## No code change is necessary

| Change | Override |
|---|---|
| Remove BatchNorm | `model.projector.norm_fn=null model.pred_proj.norm_fn=null` |
| Remove SIGReg (run D only) | `loss.sigreg.weight=0` |

`SaveCkptCallback` saves `cfg.model` with the weights. So `eval.py` builds
the model again with `nn.Identity()`, and the state dict keys agree (no BN
running stats). Check this in step 2.

## Runs

Use the same seed, data, epochs and `img_size` as the baseline.

| Run | SIGReg | BatchNorm | `subdir` |
|---|---|---|---|
| A (baseline) | on | on | existing baseline, or `abl_A_base_s3072` |
| B (from the SIGReg plan) | off | on | `abl_B_nosigreg_s3072` |
| C | on | off | `abl_C_nobn_s3072` |
| D | off | off | `abl_D_nosigreg_nobn_s3072` |

If budget is small: do D first (with B, it shows if BatchNorm gives any
protection). C is the least important.

## Steps

### 1. Smoke test (local or RunPod, a few minutes)

```bash
python train.py data=pusht trainer.max_epochs=1 +trainer.limit_train_batches=50 \
    model.projector.norm_fn=null model.pred_proj.norm_fn=null \
    subdir=abl_smoke_nobn
```

Check:

- The run starts with no error.
- The printed model has no `BatchNorm1d` in `projector` and `pred_proj`.

### 2. Check that eval loads a no-BN checkpoint

```bash
python eval.py --config-name=pusht.yaml policy=pusht/abl_smoke_nobn/weights_epoch_001.pt eval.img_size=112
```

It must load with no state-dict error. The success rate is not important
here. Then delete the `abl_smoke_nobn` checkpoint folder.

### 3. Full runs (RunPod)

Use the same wandb flags as the baseline (see `docs/training.md`).

```bash
# C: no BatchNorm
python train.py data=pusht num_workers=16 \
    wandb.enabled=True wandb.config.entity=<entity> wandb.config.project=lewm \
    model.projector.norm_fn=null model.pred_proj.norm_fn=null \
    subdir=abl_C_nobn_s3072

# D: no SIGReg and no BatchNorm
python train.py data=pusht num_workers=16 \
    wandb.enabled=True wandb.config.entity=<entity> wandb.config.project=lewm \
    loss.sigreg.weight=0 \
    model.projector.norm_fn=null model.pred_proj.norm_fn=null \
    subdir=abl_D_nosigreg_nobn_s3072
```

For D, look at the wandb curves early (first 1 to 3 epochs). If `pred_loss`
goes to about 0 and `sigreg_loss` goes up a lot, the model has collapsed.
You can stop the run early to save money. Record the epoch where this
occurs, and compare it with run B.

### 4. Eval

Evaluate the same epochs as the baseline:

```bash
for r in abl_C_nobn_s3072 abl_D_nosigreg_nobn_s3072; do
  for e in 010 050 100; do
    python eval.py --config-name=pusht.yaml policy=pusht/$r/weights_epoch_$e.pt eval.img_size=112
  done
done
```

If a run was stopped early, evaluate the last saved epoch.

### 5. Collapse metrics (optional)

Use the `scripts/embed_stats.py` script from the SIGReg plan. Run it on the
last epoch of C and D.

## What to record

| Run | final `pred_loss` | final `sigreg_loss` | eff. rank | collapse epoch | success @10 | @50 | @100 |
|---|---|---|---|---|---|---|---|
| A | | | | | | | |
| B | | | | | | | |
| C | | | | | | | |
| D | | | | | | | |

## Expected results

- **C:** close to A. SIGReg alone should be enough to prevent collapse.
- **D:** collapse is likely.
- **B vs D:** if B collapses later or less than D, then BatchNorm gives some
  protection on its own.

## Risks

- Without BatchNorm, training can be less stable. If C shows loss spikes,
  record this. Do not tune the learning rate for C only, because then the
  comparison is not fair.
- A collapsed model can still give a non-zero success rate by chance. Compare
  with `policy=random` (see `docs/evaluation.md`).
