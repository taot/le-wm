import os
from pathlib import Path

from dotenv import load_dotenv

# STABLEWM_HOME (storage root) from the repo's .env; a value set in the shell wins.
load_dotenv(Path(__file__).parent / ".env")

os.environ["MUJOCO_GL"] = "egl"
# Cloud containers often report every host CPU (e.g. 252 on RunPod) while the
# cgroup quota allows far fewer; torch then oversubscribes threads on the tiny
# per-step ops and each env step takes seconds. Must be set before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import time
import warnings
from typing import Any

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

# eval only forks via subprocess (fork_exec -> exec), so lancedb state is never used in the child
warnings.filterwarnings(
    "ignore", message="lancedb fork support is experimental", category=RuntimeWarning
)

def img_transform(cfg: DictConfig) -> transforms.Compose:
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_dataset(cfg: DictConfig, dataset_name: str) -> Any:
    cache_dir = cfg.get("cache_dir")  # None: $STABLEWM_HOME
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    return dataset

def fit_process(
    cfg: DictConfig, dataset: Any
) -> dict[str, preprocessing.StandardScaler]:
    """Fit the per-column StandardScalers the policy uses to normalize inputs."""
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def build_policy(
    cfg: DictConfig, process: dict[str, preprocessing.StandardScaler]
) -> swm.policy.RandomPolicy | swm.policy.WorldModelPolicy:
    """CEM world-model policy for cfg.policy, or a random policy."""
    if cfg.get("policy", "random") == "random":
        return swm.policy.RandomPolicy()

    model = swm.wm.utils.load_pretrained(cfg.policy)
    model = model.to("cuda")
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }
    return swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )


def eval_paths(policy: str) -> tuple[Path, Path]:
    """(results folder, video folder) for a policy, kept inside its run folder.

    policy=<env>/<subdir>/weights_epoch_NNN.pt -> checkpoints/<env>/<subdir>/eval/
    for the results files (appended, shared by all epochs) and .../eval/weights_epoch_NNN/
    for the videos. A run folder or HF repo name uses checkpoints/<name>/eval/;
    policy=random uses $STABLEWM_HOME/eval/random/.
    """
    if policy == "random":
        results_path = Path(swm.data.utils.get_cache_dir(), "eval", "random")
        return results_path, results_path
    ckpt = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), policy)
    if ckpt.suffix == ".pt":
        results_path = ckpt.parent / "eval"
        return results_path, results_path / ckpt.stem
    results_path = ckpt / "eval"
    return results_path, results_path


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig) -> None:
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = fit_process(cfg, dataset)
    policy = build_policy(cfg, process)

    results_path, video_path = eval_paths(cfg.policy)
    print(f"results: {results_path / cfg.output.filename}")
    print(f"videos:  {video_path}")

    world.set_policy(policy)
    video_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    # sample the episodes and the starting indices.
    # Episodes are addressed positionally (as load_chunk expects), derived from
    # the dataset's episode lengths/offsets rather than index columns, which
    # LanceDataset does not expose via column_names/get_row_data.
    episode_len = np.asarray(dataset.lengths)
    episode_offsets = np.asarray(dataset.offsets)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1

    # per-row episode position and step within the episode
    row_episode = np.repeat(np.arange(len(episode_len)), episode_len)
    row_step = np.arange(episode_len.sum()) - np.repeat(episode_offsets - episode_offsets[0], episode_len)

    # remove all the rows for which step > max_start_idx of its episode
    valid_mask = row_step <= max_start_idx[row_episode]
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = row_episode[random_episode_indices]
    eval_start_idx = row_step[random_episode_indices]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=video_path,
    )
    end_time = time.time()

    print(metrics)

    results_file = results_path / cfg.output.filename
    with results_file.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
