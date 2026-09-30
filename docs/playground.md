# Running the PushT playground

`playground.py` serves a web page where you can drive the PushT agent with the mouse, or hand control to the world-model planner (the same CEM policy as `eval.py`) or the expert demo. You can switch between the two eval setups:

- **25-step** (`config/eval/pusht.yaml`): start from a dataset state; the goal is the expert's state 25 steps later.
- **Full solve** (`config/eval/pusht_full.yaml`): random start; the goal is the block on the green T.

The server takes the same Hydra overrides as `eval.py`, plus `--host` (default `127.0.0.1`) and `--port` (default `8000`).

## What you need

- **A checkpoint** in `$STABLEWM_HOME/checkpoints/` (default `~/.stable_worldmodel/checkpoints/`). Each run folder holds `config.json` and `weights_epoch_*.pt`. Pass it as `policy=pusht/<subdir>/weights_epoch_NNN.pt`, relative to `checkpoints/`.
- **The dataset** `librakevin/lewm-pusht`. The server downloads it on first start if it isn't cached yet. See [cloud_training.md](cloud_training.md) for how to download it yourself, which is more reliable.
- **A CUDA GPU.** The model is always loaded onto `cuda`. To play by hand without a GPU or checkpoint, use `policy=random` (the planner button then takes random actions).
- **`eval.img_size`** must match the image size the checkpoint was trained with (e.g. `112` for an `img112` run).

## Run locally

From the repo folder:

```bash
uv run python playground.py policy=pusht/<subdir>/weights_epoch_003.pt eval.img_size=112
```

When it prints `Playground ready on http://127.0.0.1:8000`, open http://localhost:8000 in your browser.

To copy a checkpoint trained on RunPod to your machine, sync it from the HF bucket:

```bash
hf buckets sync hf://buckets/librakevin/lewm-checkpoints/pusht/<subdir> ~/.stable_worldmodel/checkpoints/pusht/<subdir>
```

A small laptop GPU (e.g. an MX550 with 2 GB) may be slow or run out of memory while planning. If so, run it on RunPod instead.

## Run on RunPod

The server only listens on the pod's `127.0.0.1`, so you reach it through an SSH tunnel.

1. On the pod, set up the repo and environment as in [cloud_training.md](cloud_training.md) (clone, `uv sync`, `export STABLEWM_HOME=/workspace/swm`). Then start the server:

   ```bash
   ssh runpod
   cd le-wm && source .venv/bin/activate
   export STABLEWM_HOME=/workspace/swm
   python playground.py policy=pusht/<subdir>/weights_epoch_003.pt eval.img_size=112
   ```

2. On your own machine, in a second terminal, open the tunnel and leave it running:

   ```bash
   ssh -N -L 8000:localhost:8000 runpod
   ```

3. Open http://localhost:8000 in your browser.

To keep the server running after you disconnect, start it inside `tmux` on the pod.

## Tips

- **Another epoch:** change the `weights_epoch_NNN.pt` in `policy=`.
- **Port in use:** add `--port 8080` (and use `8080` in the tunnel and URL).
- **Several tabs:** every open tab views and drives the same episode.
