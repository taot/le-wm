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

Run A is the existing baseline in the HF bucket. See "Baseline (run A)" in
`plans/ablation_no_sigreg.md` for its location and config. Runs C and D use
the same config (`launcher=runpod trainer.max_epochs=3`), with only the
changes in the table below.

| Run | SIGReg | BatchNorm | Epochs | Location |
|---|---|---|---|---|
| A (baseline) | on (0.09) | on | 3 | HF bucket, `pusht/2026-09-28_114752_img112_s3072` |
| B (from the SIGReg plan) | off (0) | on | 3 | `pusht/<date>_<time>_img112_s3072_nosigreg` |
| C | on (0.09) | off | 3 | `pusht/<date>_<time>_img112_s3072_nobn` |
| D | off (0) | off | 3 | `pusht/<date>_<time>_img112_s3072_nosigreg_nobn` |

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

Same config as run A, with only the BatchNorm (and for D, the SIGReg)
overrides:

```bash
# C: no BatchNorm
python train.py launcher=runpod data=pusht trainer.max_epochs=3 \
    model.projector.norm_fn=null model.pred_proj.norm_fn=null \
    'subdir=${now:%Y-%m-%d_%H%M%S}_img${img_size}_s${seed}_nobn'

# D: no SIGReg and no BatchNorm
python train.py launcher=runpod data=pusht trainer.max_epochs=3 \
    loss.sigreg.weight=0 \
    model.projector.norm_fn=null model.pred_proj.norm_fn=null \
    'subdir=${now:%Y-%m-%d_%H%M%S}_img${img_size}_s${seed}_nosigreg_nobn'
```

**Run names:** each run folder has the start date and time, like the
baseline. The `subdir` override above adds a label at the end, for example
`pusht/2026-10-14_063811_img112_s3072_nobn`. Keep the single quotes, so
that the shell does not expand `${...}`. Hydra fills in the values. The
date and time are not known before the run starts, so read the real folder
name from the first lines of the training log (or from wandb) and write it
in the runs table. To resume a run, pass that real name: `subdir=<name>`.

Before each run starts, check the config: add `--cfg job` to the command,
and diff the output against the baseline `config.yaml`. The only
differences must be the overrides above and the bookkeeping keys.

For D, look at the wandb curves during the run. If `pred_loss` goes to
about 0 and `sigreg_loss` goes up a lot, the model has collapsed. Record the
step where this occurs, and compare it with run B. Do not stop the run
early: with only 3 epochs, all 3 checkpoints are necessary.

### 4. Eval

Evaluate all 3 epochs, with the same eval config as A and B:

```bash
C=<real subdir of run C>   # for example 2026-10-14_063811_img112_s3072_nobn
D=<real subdir of run D>   # for example 2026-10-14_091502_img112_s3072_nosigreg_nobn
for r in $C $D; do
  for e in 001 002 003; do
    python eval.py --config-name=pusht.yaml policy=pusht/$r/weights_epoch_$e.pt eval.img_size=112
  done
done
```

### 5. Collapse metrics (optional)

Use the `scripts/embed_stats.py` script from the SIGReg plan. Run it on the
last epoch of C and D.

## What to record

| Run | final `pred_loss` | final `sigreg_loss` | eff. rank | collapse step | success @1 | @2 | @3 |
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
