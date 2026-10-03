# Evaluation

This guide covers how to evaluate a trained LeWM checkpoint: getting the checkpoint, running `eval.py`, and finding the results. The commands are the same on your own machine and on a cloud GPU machine. The only difference is one line in `.env` (step 1).

For how checkpoints are produced, see [training.md](training.md#checkpoints). For the paper's pretrained checkpoints, see [checkpoints.md](checkpoints.md).

## Requirements

- An NVIDIA GPU with CUDA. `eval.py` moves the model to `cuda`, so it does not run on CPU. A small laptop GPU works, but CEM planning is several times slower than on a cloud GPU.
- The repo set up with `uv sync && source .venv/bin/activate` (see [training.md](training.md#1-set-up-the-code)).

## How paths work

There is one setting, `STABLEWM_HOME`: the root folder for everything. Each run keeps all of its files, including eval results, in one folder:

```
$STABLEWM_HOME/
├── datasets/                      # datasets, e.g. librakevin--lewm-pusht/
└── checkpoints/
    └── <env>/<subdir>/            # one training run, e.g. pusht/2026-09-28_114752_img112_s3072/
        ├── config.json            # model settings, needed to load the weights
        ├── weights_epoch_NNN.pt   # weights after each epoch
        └── eval/
            ├── pusht_results.txt        # metrics of every eval of this run (all epochs)
            └── weights_epoch_NNN/       # videos of the eval of that epoch
```

In commands, a run is named `<env>/<subdir>` (the path under `checkpoints/`) and a checkpoint is `<env>/<subdir>/weights_epoch_NNN.pt`.

## 1. Set the storage root (once per machine)

Copy the example file and edit the one line in it:

```bash
cp .env.example .env
```

| Where | `STABLEWM_HOME` |
|---|---|
| Your own machine | keep the default, `${HOME}/.stable_worldmodel` |
| Cloud GPU machine | the fast local disk, e.g. `/workspace/swm` |

`.env` is git-ignored, so each machine keeps its own. `train.py`, `eval.py` and `playground.py` read it automatically. To use `$STABLEWM_HOME` in shell commands, load it into each new shell:

```bash
source .env
```

A value already exported in the shell takes priority over `.env`.

## 2. Get the checkpoint

If the run was trained on this machine, it is already in place. Skip to step 3.

If it was trained on another machine, copy its run folder to `$STABLEWM_HOME/checkpoints/<env>/<subdir>/` here, e.g. with `scp -r`.

## 3. Get the dataset

The eval starts episodes from states in the PushT dataset (`librakevin/lewm-pusht`), and it uses the dataset to normalize actions and states. If the dataset is not in `$STABLEWM_HOME/datasets/` yet, `eval.py` downloads it on first use. Downloading it yourself first is more reliable; see [datasets.md](datasets.md#download).

## 4. Run the eval

Read `img_size` from the run folder name (`img112` means 112) and pass it as `eval.img_size`. It must match the size the model was trained with. The config default is 224.

**From dataset states** (the default, 50 episodes):

```bash
python eval.py --config-name=pusht.yaml policy=pusht/<subdir>/weights_epoch_003.pt eval.img_size=112
```

Each episode starts from a state in the dataset, and the goal is the expert's state 25 steps later. The planner has 50 steps to reach it.

**Random baseline**, to compare against. Its results go to `$STABLEWM_HOME/eval/random/`:

```bash
python eval.py --config-name=pusht.yaml policy=random
```

At the start, `eval.py` prints where the results and videos will go.

### Useful overrides

Add these to the end of the command, Hydra style:

| Override | Effect |
|---|---|
| `eval.num_eval=10` | fewer episodes, for a quick check |
| `seed=0` | a different set of start states |
| `eval.goal_offset_steps=50 eval.eval_budget=100` | a further goal and a bigger step budget (`pusht.yaml` only) |
| `plan_config.horizon=5` | planning horizon. `horizon × action_block` must be ≤ `eval.eval_budget` |

### Running in the background

An eval can take a while. On a remote machine, run it so it survives a dropped SSH connection, and keep a log:

```bash
nohup python eval.py --config-name=pusht.yaml policy=pusht/<subdir>/weights_epoch_100.pt eval.img_size=112 > eval.log 2>&1 &
```

Follow it with:

```bash
tail -f eval.log
```

## 5. Read the results

Everything is in the run's `eval/` folder (see [How paths work](#how-paths-work)):

- `pusht_results.txt`: each eval appends its config and metrics, so all epochs you evaluated are in one file.
- `weights_epoch_NNN/`: one `.mp4` per episode.

Each eval appends a block that ends like this:

```
==== RESULTS ====
metrics: {'success_rate': ..., 'episode_successes': ..., ...}
evaluation_time: ... seconds
```

`success_rate` (in percent) is the main number. `episode_successes` shows which episodes succeeded. The metrics are also printed at the end of the console output.

To watch the videos from a cloud machine, copy the folder to your own machine. Write the remote path out in full, because the remote `$STABLEWM_HOME` is not set on your machine:

```bash
scp -r <host>:/workspace/swm/checkpoints/pusht/<subdir>/eval ./eval
```

## Comparing epochs

To compare several checkpoints, loop over them. All results are appended to the same `pusht_results.txt`:

```bash
for e in 010 050 100; do python eval.py --config-name=pusht.yaml policy=pusht/<subdir>/weights_epoch_$e.pt eval.img_size=112; done
```

## Watching the planner

`playground.py` takes the same overrides as `eval.py`. It lets you drive the agent with the mouse, or hand control to the planner:

```bash
python playground.py policy=pusht/<subdir>/weights_epoch_100.pt eval.img_size=112
```

Open http://localhost:8000. On a cloud machine, first tunnel the port from your own machine:

```bash
ssh -N -L 8000:localhost:8000 <host>
```
