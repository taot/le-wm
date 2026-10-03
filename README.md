# LeWorldModel (fork)

> **This repo is a fork of [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)**, the official code for the paper [*LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*](https://arxiv.org/pdf/2603.19312v1). The original README is kept as [README_orig.md](README_orig.md).

What this fork adds on top of the original:

- PushT is trained on a Lance copy of the dataset ([`librakevin/lewm-pusht`](https://huggingface.co/datasets/librakevin/lewm-pusht)), which is faster to load.
- Training runs can be resumed from their last epoch.
- A web playground for PushT.
- Guides for training on a cloud GPU and for working with the datasets.

## About this code

`jepa.py` holds the LeWM model and `train.py` trains it. The code builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) (environments, planning, evaluation) and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) (training). Configs use [Hydra](https://hydra.cc/) and live under `config/train/` and `config/eval/`.

## Quickstart

**Install:**
```bash
uv sync
source .venv/bin/activate
```

**Data:** datasets and checkpoints are stored under `$STABLEWM_HOME` (default `~/.stable_worldmodel`). Set it once per machine in a git-ignored `.env` file, which the scripts read automatically:
```bash
cp .env.example .env
```
The PushT dataset downloads automatically on first use. To download it yourself, use other environments, or convert `.h5` files, see [docs/datasets.md](docs/datasets.md).

**Train:**
```bash
python train.py data=pusht wandb.enabled=True wandb.config.entity=<your_entity> wandb.config.project=<your_project>
```
Pass the wandb settings on the command line; values in `lewm.yaml` get overwritten (see [docs/training.md](docs/training.md#5-full-run)). Checkpoints are saved to `$STABLEWM_HOME/checkpoints/pusht/<subdir>/`.

**Evaluate** (planning with the trained model):
```bash
python eval.py --config-name=pusht.yaml policy=pusht/<subdir>/weights_epoch_100.pt eval.img_size=112
```
Results and videos go to the run's `eval/` folder (see [docs/evaluation.md](docs/evaluation.md)).

**Playground:** drive the PushT agent with the mouse, or hand control to the planner. Takes the same overrides as `eval.py`:
```bash
python playground.py policy=pusht/<subdir>/weights_epoch_100.pt eval.img_size=112
```
Then open http://localhost:8000. On a remote GPU machine, tunnel the port first: `ssh -N -L 8000:localhost:8000 <host>`.

## Documentation

- [docs/training.md](docs/training.md): GPU choice, time estimates, running on a cloud machine, checkpoints, resuming, evaluation.
- [docs/evaluation.md](docs/evaluation.md): evaluating a trained checkpoint, locally or on a cloud GPU, and reading the results.
- [docs/datasets.md](docs/datasets.md): where datasets live, downloading, HDF5 → Lance conversion, browsing a dataset.
- [docs/pusht_dataset.md](docs/pusht_dataset.md): what is inside the PushT dataset.
- [docs/checkpoints.md](docs/checkpoints.md): the paper's pretrained LeWM and baseline checkpoints.
