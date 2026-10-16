# Ablation: remove the projector, and the final ViT LayerNorm

Status: plan, not started.

Related plans: `plans/ablation_no_sigreg.md` (run A and the run name
format), `plans/ablation_no_batchnorm.md` (runs C and D).

## Goal

Test the reason that the paper gives for the projector. Section 3.1 (Model
Architecture, page 4) says:

> The projection step is necessary because the final ViT layer applies a
> Layer Normalization, which prevents our anti-collapse objective from
> being optimized effectively.

We want to know:

1. Without the projector, does SIGReg fail on the LayerNorm output? (run E)
2. Without the projector **and** without the final LayerNorm, does the
   model work as well as the baseline? If yes, the projector is only a fix
   for the LayerNorm. (run F)

## Background

The encoder is in `jepa.py:31`:

```python
output = self.encoder(pixels, interpolate_pos_encoding=True)
pixels_emb = output.last_hidden_state[:, 0]  # cls token
emb = self.projector(pixels_emb)
```

- The encoder is a HuggingFace `ViTModel` (from `vit_hf` in
  `stable_pretraining/backbone/utils.py`). Its `last_hidden_state` comes
  from a final `self.layernorm(...)`
  (`transformers/models/vit/modeling_vit.py:385`).
- The projector is `MLP` (`module.py:224`):
  `Linear(192→2048) → BatchNorm → GELU → Linear(2048→192)`. It has about
  0.79M parameters.
- SIGReg and the prediction targets both use `emb`, the projector output.
  The predictor works in this space, and planning measures distances in it.

Why LayerNorm can be a problem for SIGReg: for each frame, LayerNorm
subtracts the mean of the 192 features and divides by their std. So,
before the learned `γ` and `β`, each output vector:

- lies in a 191-d subspace (its features sum to 0), and
- has a length of almost exactly √192.

After `γ ⊙ u + β`, the vectors are on a thin, curved shell. A sample from
N(0, I) uses all 192 directions, and its length changes from sample to
sample. So SIGReg can never be fully satisfied on LayerNorm output.

The paper does not test this claim. Its ablations (Section 4.3 and
Appendix G) cover the SIGReg settings (number of projections, knots,
weight λ), the latent dimension, the predictor size, a reconstruction
loss, the encoder type (ViT 96% compared with ResNet-18 94%), predictor
dropout, and the planning solver. None of them removes the projector or
the final LayerNorm. (Checked by reading the paper with ChatGPT on
2026-10-16.)

This plan answers two open questions:

1. How much worse is the model when SIGReg acts directly on the `[CLS]`
   output (with its LayerNorm)? (run E)
2. Without the final LayerNorm, is the projector still necessary? (run F)

A third question, what the Linear layers and the BatchNorm in the
projector each contribute, is partly answered by run C in
`plans/ablation_no_batchnorm.md` (projector kept, BatchNorm removed).

## Runs

| Run | Projector | Final ViT LayerNorm | SIGReg | Epochs | Location |
|---|---|---|---|---|---|
| A (baseline) | yes | yes | on (0.09) | 3 | HF bucket, `pusht/2026-09-28_114752_img112_s3072` |
| C (from the BatchNorm plan, for comparison) | yes, **without BatchNorm** | yes | on (0.09) | 3 | `pusht/<date>_<time>_img112_s3072_nobn` |
| E | **no** | yes | on (0.09) | 3 | `pusht/<date>_<time>_img112_s3072_noproj` |
| F | **no** | **no** | on (0.09) | 3 | `pusht/<date>_<time>_img112_s3072_noproj_noln` |

Use two runs, not one. If one run removes both the projector and the
LayerNorm, we cannot tell which change caused the result.

`pred_proj` stays in all runs. Only the target side (the encoder output)
changes.

## Code changes

### Run E: none

`model.projector=null` sets the projector to `None`, and `JEPA` then uses
`nn.Identity()` (`jepa.py:28`). I checked the override with
`python train.py --cfg job data=pusht model.projector=null`: it gives
`projector: null`.

### Run F: a small encoder wrapper

Add this function to `module.py` (also change the import to
`from typing import Any, Callable`):

```python
def vit_hf_no_final_ln(**kwargs: Any) -> nn.Module:
    """vit_hf without the final LayerNorm on last_hidden_state."""
    from stable_pretraining.backbone.utils import vit_hf

    model = vit_hf(**kwargs)
    model.layernorm = nn.Identity()
    return model
```

Then select it with an override:
`model.encoder._target_=module.vit_hf_no_final_ln`.

Use a config `_target_`. Do not patch the model in `train.py`.
`SaveCkptCallback` saves `cfg.model` as `config.json`, and `eval.py` builds
the model again from it (`swm.wm.utils.load_pretrained`, which calls
Hydra `instantiate`). So eval builds the same model, without the
LayerNorm, and the state dict keys agree (no `layernorm.weight` or
`layernorm.bias`).

The inner ViT layers keep their own LayerNorms (`layernorm_before`,
`layernorm_after`). Only the last LayerNorm, after all blocks, is removed.

## Steps

### 1. Smoke tests (local or RunPod, a few minutes)

```bash
# E
python train.py data=pusht trainer.max_epochs=1 +trainer.limit_train_batches=50 \
    model.projector=null \
    subdir=abl_smoke_noproj

# F (after the code change)
python train.py data=pusht trainer.max_epochs=1 +trainer.limit_train_batches=50 \
    model.projector=null model.encoder._target_=module.vit_hf_no_final_ln \
    subdir=abl_smoke_noproj_noln
```

Check:

- Each run starts with no error.
- The printed model has no `projector` MLP (E and F), and the encoder has
  `(layernorm): Identity()` at the end (F only).
- `config.json` in the checkpoint folder has `"projector": null` (E and
  F) and `"_target_": "module.vit_hf_no_final_ln"` (F).

### 2. Check that eval loads both smoke checkpoints

```bash
python eval.py --config-name=pusht.yaml policy=pusht/abl_smoke_noproj/weights_epoch_001.pt eval.img_size=112
python eval.py --config-name=pusht.yaml policy=pusht/abl_smoke_noproj_noln/weights_epoch_001.pt eval.img_size=112
```

Each must load with no state-dict error. The success rate is not important
here. Then delete both smoke checkpoint folders.

### 3. Short runs: compare the loss curves first (RunPod, ~15 min each)

The paper's claim is about optimization. So the effect should be visible
early, in `train/sigreg_loss`. Before you pay for full runs, train E and F
for a part of the first epoch:

```bash
python train.py launcher=runpod data=pusht trainer.max_epochs=1 \
    +trainer.limit_train_batches=2000 \
    model.projector=null \
    'subdir=${now:%Y-%m-%d_%H%M%S}_img${img_size}_s${seed}_noproj_short'
```

(For F, add `model.encoder._target_=module.vit_hf_no_final_ln` and use the
label `_noproj_noln_short`.)

The scheduler steps once per epoch, so the learning rate in these steps is
the same as in the first steps of run A. Compare `train/sigreg_loss` and
`train/pred_loss` with run A in wandb, at the same step.

Decide:

- If E and F both look like A, the full runs are still useful (planning
  success can differ), but the main claim is already not supported.
- If E is clearly worse than A, and F is close to A, the full runs confirm
  the claim. Continue.
- If F is clearly worse than A too, the projector helps for a reason other
  than the LayerNorm. Continue, and note this.

Delete the short-run folders after you record the curves.

### 4. Full runs (RunPod)

Same config as run A, with only the overrides for each run:

```bash
# E: no projector
python train.py launcher=runpod data=pusht trainer.max_epochs=3 \
    model.projector=null \
    'subdir=${now:%Y-%m-%d_%H%M%S}_img${img_size}_s${seed}_noproj' \
    > train_noproj.log 2>&1

# F: no projector and no final LayerNorm
python train.py launcher=runpod data=pusht trainer.max_epochs=3 \
    model.projector=null model.encoder._target_=module.vit_hf_no_final_ln \
    'subdir=${now:%Y-%m-%d_%H%M%S}_img${img_size}_s${seed}_noproj_noln' \
    > train_noproj_noln.log 2>&1
```

Each run takes about 1.75 hours (`docs/training.md`).

**Run names:** see "Run names" in `plans/ablation_no_sigreg.md`. Keep the
single quotes. Read the real folder name from the first lines of the
training log, and write it in the runs table.

Before each run starts, check the config: add `--cfg job` to the command,
and diff the output against the baseline `config.yaml`. The only
differences must be the overrides above and the bookkeeping keys.

Do not stop a run early: with only 3 epochs, all 3 checkpoints are
necessary.

### 5. Eval

Evaluate all 3 epochs, with the same eval config as A:

```bash
E=<real subdir of run E>   # for example 2026-10-20_101500_img112_s3072_noproj
F=<real subdir of run F>   # for example 2026-10-20_123000_img112_s3072_noproj_noln
for r in $E $F; do
  for e in 001 002 003; do
    python eval.py --config-name=pusht.yaml policy=pusht/$r/weights_epoch_$e.pt eval.img_size=112
  done
done
```

### 6. Embedding analysis

Use the same extraction script and plots as the SIGReg comparison (A
compared with B). Run them on all 3 epochs of E and F.

In E and F, `cls` and `emb` are the same vector, because the projector is
an identity. In F, `cls` is the output before the (removed) LayerNorm.

For E, also check the shell shape directly: plot the histogram of
`‖emb − β‖` over frames. If the LayerNorm limits the latents, this
histogram is very narrow. For N(0, I) in 192-d, the length is about
13.9 ± 0.7.

## What to record

| Run | `sigreg_loss` @2000 steps | final `pred_loss` | final `sigreg_loss` | eff. rank | success @1 | @2 | @3 |
|---|---|---|---|---|---|---|---|
| A | | | | | | | |
| E | | | | | | | |
| F | | | | | | | |

## Expected results

- **E:** `sigreg_loss` stays higher than in A for the full run. The
  latents fill only a thin shell. Planning success is probably lower than A,
  but the model may not collapse fully, because SIGReg still pushes
  against collapse.
- **F:** this is the open question. If F is close to A, the projector is
  only a fix for the LayerNorm. If F is worse than A, the projector also
  helps for another reason (more parameters, or a space that is separate
  from the backbone).

## Risks

- **Fewer parameters.** E and F have about 0.79M fewer parameters than A
  (ViT-tiny has about 5.5M). If E or F is worse, part of the difference
  can come from this. Write this down with the result.
- **One BatchNorm less.** The projector has a BatchNorm. E and F remove it,
  but `pred_proj` keeps its BatchNorm. Compare with run C in
  `plans/ablation_no_batchnorm.md`, which removes both BatchNorms but keeps
  the projector.
- **Free output scale in F.** Without the final LayerNorm, the ViT output
  scale is not limited. SIGReg should control it. If F shows loss spikes or
  NaN, look at the scale of `emb` first. Do not tune the learning rate for F
  only, because then the comparison is not fair.
- A collapsed model can still give a non-zero success rate by chance.
  Compare with `policy=random` (see `docs/evaluation.md`).
