"""Encode a fixed set of PushT frames with LeWM checkpoints and save the embeddings.

The frames are decoded one time, and every checkpoint encodes the same frames.
For each checkpoint, the script writes one .npz file to
$STABLEWM_HOME/embeddings/<run>/<checkpoint name>.npz. The plot notebook is
playground/sigreg_collapse_notebook.py.

Two groups of frames are saved (key "group"):
  0  random frames from the full dataset (for the distribution statistics)
  1  all frames of a few random episodes (for the trajectory plots)

The train/val split in train.py is a split of overlapping clips, so the
validation frames are also in training clips. This script therefore samples
frames from the full dataset.

Usage (from the repo root):
  uv run python scripts/extract_embeddings.py
  uv run python scripts/extract_embeddings.py --runs pusht/<run A> pusht/<run B> --epochs 3
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # jepa.py, module.py and utils.py are in the repo root
# STABLEWM_HOME (storage root) from the repo's .env; a value set in the shell wins.
load_dotenv(REPO / ".env")

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from jepa import JEPA
from utils import get_img_preprocessor

DEFAULT_RUNS = [
    "pusht/2026-09-28_114752_img112_s3072",  # run A: with SIGReg (weight 0.09)
    "pusht/2026-10-14_115337_img112_s3072_nosigreg",  # run B: without SIGReg (weight 0)
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="run folders, relative to $STABLEWM_HOME/checkpoints")
    p.add_argument("--epochs", nargs="+", type=int, default=None, help="epochs to encode (default: all weights_epoch_*.pt)")
    p.add_argument("--dataset", default="librakevin/lewm-pusht")
    p.add_argument("--num-frames", type=int, default=20000, help="number of random frames (group 0)")
    p.add_argument("--num-episodes", type=int, default=10, help="number of full episodes (group 1)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path, default=None, help="default: $STABLEWM_HOME/embeddings")
    return p.parse_args()


def find_checkpoints(runs: list[str], epochs: list[int] | None) -> list[str]:
    """Checkpoint paths relative to $STABLEWM_HOME/checkpoints, for example pusht/<run>/weights_epoch_003.pt."""
    ckpt_root = swm.data.utils.get_cache_dir(sub_folder="checkpoints")
    ckpts = []
    for run in runs:
        run_dir = ckpt_root / run
        if epochs is None:
            files = sorted(run_dir.glob("weights_epoch_*.pt"))
        else:
            files = [run_dir / f"weights_epoch_{e:03d}.pt" for e in epochs]
        missing = [f for f in files if not f.exists()]
        if not files or missing:
            raise FileNotFoundError(f"no checkpoint or missing checkpoint in {run_dir}: {missing}")
        ckpts += [str(f.relative_to(ckpt_root)) for f in files]
    return ckpts


def run_config(ckpt: str) -> dict:
    """Training config.yaml of the run that owns the checkpoint."""
    path = swm.data.utils.get_cache_dir(sub_folder="checkpoints") / Path(ckpt).parent / "config.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def select_frames(
    lengths: np.ndarray, offsets: np.ndarray, num_frames: int, num_episodes: int, seed: int
) -> dict[str, np.ndarray]:
    """Dataset rows to encode, with their episode, step and group."""
    rng = np.random.default_rng(seed)
    total = int(lengths.sum())
    random_rows = np.sort(rng.choice(total, size=num_frames, replace=False))
    episodes = np.sort(rng.choice(len(lengths), size=num_episodes, replace=False))
    episode_rows = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in episodes])

    rows = np.concatenate([random_rows, episode_rows]).astype(np.int64)
    group = np.concatenate([np.zeros(len(random_rows)), np.ones(len(episode_rows))]).astype(np.int8)
    episode = np.searchsorted(offsets, rows, side="right") - 1
    step = rows - offsets[episode]
    return {"row": rows, "group": group, "episode": episode, "step": step}


def encode(model: JEPA, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(CLS token before the projector, projector output). Same steps as JEPA.encode."""
    cls = model.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    return cls, model.projector(cls)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or swm.data.utils.get_cache_dir(sub_folder="embeddings")
    ckpts = find_checkpoints(args.runs, args.epochs)
    configs = {c: run_config(c) for c in ckpts}

    img_sizes = {cfg["img_size"] for cfg in configs.values()}
    if len(img_sizes) != 1:
        raise ValueError(f"all checkpoints must use the same img_size, got {img_sizes}")
    img_size = img_sizes.pop()

    # num_steps=1 and frameskip=1: dataset item i is dataset row i
    dataset = swm.data.load_dataset(
        args.dataset,
        keys_to_load=["pixels", "state", "proprio"],
        transform=get_img_preprocessor(source="pixels", target="pixels", img_size=img_size),
    )
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)
    frames = select_frames(lengths, offsets, args.num_frames, args.num_episodes, args.seed)
    n = len(frames["row"])
    print(f"{n} frames ({args.num_frames} random, {n - args.num_frames} from {args.num_episodes} episodes)")

    models: dict[str, JEPA] = {}
    for c in ckpts:
        model = swm.wm.utils.load_pretrained(c)
        assert isinstance(model, JEPA)
        models[c] = model.to(args.device).eval()  # eval: BatchNorm uses its running statistics
        print(f"loaded {c}  (sigreg weight {configs[c]['loss']['sigreg']['weight']})")

    cls_out = {c: [] for c in ckpts}
    emb_out = {c: [] for c in ckpts}
    state, proprio = [], []
    start = time.time()
    with torch.inference_mode():
        for i in range(0, n, args.batch_size):
            samples = dataset.__getitems__(frames["row"][i : i + args.batch_size].tolist())
            pixels = torch.stack([s["pixels"][0] for s in samples]).to(args.device)
            state.append(torch.stack([s["state"][0] for s in samples]).numpy())
            proprio.append(torch.stack([s["proprio"][0] for s in samples]).numpy())
            for c, model in models.items():
                cls, emb = encode(model, pixels)
                cls_out[c].append(cls.float().cpu().numpy())
                emb_out[c].append(emb.float().cpu().numpy())
            done = min(i + args.batch_size, n)
            print(f"\r{done}/{n} frames, {time.time() - start:.0f} s", end="", flush=True)
    print()

    for c in ckpts:
        cfg = configs[c]
        path = out_dir / Path(c).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            cls=np.concatenate(cls_out[c]),
            emb=np.concatenate(emb_out[c]),
            state=np.concatenate(state),
            proprio=np.concatenate(proprio),
            **frames,
            checkpoint=c,
            epoch=int(Path(c).stem.split("_")[-1]),
            sigreg_weight=float(cfg["loss"]["sigreg"]["weight"]),
            img_size=img_size,
            seed=args.seed,
        )
        print(f"saved {path}")


if __name__ == "__main__":
    main()
