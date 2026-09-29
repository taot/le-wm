import os

os.environ["MUJOCO_GL"] = "egl"
# Cloud containers often report every host CPU (e.g. 252 on RunPod) while the
# cgroup quota allows far fewer; torch then oversubscribes threads on the tiny
# per-step ops and each env step takes seconds. Must be set before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import time
import warnings
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

# eval only forks via subprocess (fork_exec -> exec), so lancedb state is never used in the child
warnings.filterwarnings(
    "ignore", message="lancedb fork support is experimental", category=RuntimeWarning
)

class BlockOnGoalSuccess(gym.Wrapper):
    """Full-solve success: the block is on the green-T goal_pose.

    Uses the env's own thresholds (position < 20, angle < pi/9) but on the
    block only, ignoring the agent position that PushT's eval_state includes.
    """

    def step(self, action):
        obs, reward, _, truncated, info = self.env.step(action)
        env = self.env.unwrapped
        pos_diff = np.linalg.norm(np.asarray(env.block.position) - env.goal_pose[:2])
        angle_diff = abs((env.block.angle - env.goal_pose[2] + np.pi) % (2 * np.pi) - np.pi)
        terminated = bool(pos_diff < 20 and angle_diff < np.pi / 9)
        return obs, reward, terminated, truncated, info


def evaluate_full_solve(cfg, world, video_path):
    """Run num_eval episodes from random starts toward the green-T goal_pose.

    The goal image is rendered with the block on goal_pose and the agent at
    eval.goal_agent_pos (where expert demos that end on the T leave it).
    """
    world.reset(seed=cfg.seed)
    goal_pose = world.envs.envs[0].unwrapped.goal_pose
    goal_state = np.concatenate([cfg.eval.goal_agent_pos, goal_pose, [0.0, 0.0]])

    video_path.mkdir(parents=True, exist_ok=True)
    metrics = world.evaluate(
        episodes=cfg.eval.num_eval,
        seed=cfg.seed,
        options={"goal_state": goal_state},
        video=video_path,
        reset_mode="wait",
    )

    # with reset_mode="wait", envs that finish early keep recording their frozen
    # last frame into episode_remaining_*.mp4; those clips are not episodes.
    for f in video_path.glob("episode_remaining_*.mp4"):
        f.unlink()
    Image.fromarray(world.envs.envs[0].unwrapped._goal).save(video_path / "goal.png")
    return metrics


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_dataset(cfg, dataset_name):
    cache_dir = cfg.get("cache_dir") or os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # dataset: reach the expert state goal_offset_steps ahead of a dataset start.
    # full_solve: from a random start, push the block onto the green-T goal_pose.
    full_solve = cfg.eval.get("mode", "dataset") == "full_solve"

    # create world environment
    if full_solve:
        cfg.world.max_episode_steps = cfg.eval.eval_budget
        world = swm.World(
            **cfg.world, image_shape=(224, 224), extra_wrappers=[BlockOnGoalSuccess]
        )
    else:
        cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
        world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if full_solve:
        metrics = evaluate_full_solve(cfg, world, results_path / "full_solve")
    else:
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
            video=results_path,
        )
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
