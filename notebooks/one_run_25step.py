import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Imports for the whole notebook. eval.py lives in the repo root, so the
    # root is put on sys.path first. eval.py must be imported before torch:
    # it sets MUJOCO_GL and caps torch's CPU threads on import.
    import sys
    import tempfile
    from pathlib import Path
    from typing import Any

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    import eval as ev  # noqa: E402  (sets env vars before torch loads)
    import gymnasium as gym
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import stable_worldmodel as swm
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    return (
        Any,
        OmegaConf,
        Path,
        ROOT,
        compose,
        ev,
        gym,
        initialize_config_dir,
        mo,
        np,
        plt,
        swm,
        tempfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One 25-step episode, run by the world-model planner

    This runs **one** episode of the 25-step eval (`config/eval/pusht.yaml`), with the
    same planner and the same setup as `eval.py`:

    - **start**: a state in the middle of an expert demo;
    - **goal**: the expert's own state 25 steps later (the policy sees its dataset image);
    - **budget**: 50 env steps;
    - **success**: agent **and** block within 20 px of the goal state, and the block
      within 20° of the goal angle. The episode stops as soon as that happens.

    You pick which of `eval.py`'s 50 episodes to run (same seed, same starts), then
    press **Run**. Needs a CUDA GPU and a checkpoint.
    """)
    return


@app.cell
def _(ROOT, compose, initialize_config_dir):
    # Settings. CHECKPOINT is relative to $STABLEWM_HOME/checkpoints/
    # (default ~/.stable_worldmodel/checkpoints/). IMG_SIZE must match the
    # image size the checkpoint was trained with (112 for an img112 run).
    CHECKPOINT = "pusht/2026-09-28_114752_img112_s3072/weights_epoch_003.pt"
    IMG_SIZE = 112

    # Load config/eval/pusht.yaml exactly as eval.py would see it with these overrides.
    with initialize_config_dir(config_dir=str(ROOT / "config/eval"), version_base=None):
        cfg = compose("pusht", [f"policy={CHECKPOINT}", f"eval.img_size={IMG_SIZE}"])
    OFFSET = cfg.eval.goal_offset_steps  # 25
    BUDGET = cfg.eval.eval_budget  # 50
    return BUDGET, CHECKPOINT, IMG_SIZE, OFFSET, cfg


@app.cell
def _(CHECKPOINT, IMG_SIZE, ROOT, compose, initialize_config_dir):
    with initialize_config_dir(config_dir=str(ROOT / "config/eval"), version_base=None):
        cfg2 = compose("pusht", [f"policy={CHECKPOINT}", f"eval.img_size={IMG_SIZE}"])
    return (cfg2,)


@app.cell
def _(cfg2):
    print(f"{cfg2.keys()=}")
    print(f"{cfg2.cache_dir=}")
    print(f"{cfg2.policy=}")
    return


@app.cell
def _(cfg, ev):
    # Load the expert dataset, fit the input scalers and build the CEM policy
    # (the checkpoint is loaded onto the GPU here). Takes a little while.
    dataset = ev.get_dataset(cfg, cfg.eval.dataset_name)
    process = ev.fit_process(cfg, dataset)
    policy = ev.build_policy(cfg, process)
    return dataset, policy


@app.cell
def _(policy):
    type(policy)
    return


@app.cell
def _(OFFSET, cfg, dataset, np):
    # Recreate eval.py's 50 (demo, start step) picks with the same seed, so
    # "episode k" here is episode k of an eval.py run.
    _len = np.asarray(dataset.lengths)
    _off = np.asarray(dataset.offsets)
    _row_ep = np.repeat(np.arange(len(_len)), _len)
    _row_step = np.arange(_len.sum()) - np.repeat(_off - _off[0], _len)
    _valid = np.nonzero(_row_step <= (_len - OFFSET - 1)[_row_ep])[0]
    _pick = np.random.default_rng(cfg.seed).choice(len(_valid) - 1, size=cfg.eval.num_eval, replace=False)
    _pick = np.sort(_valid[_pick])
    eval_episodes = _row_ep[_pick]
    eval_starts = _row_step[_pick]
    return eval_episodes, eval_starts


@app.cell
def _(cfg, mo):
    # Which of eval.py's episodes to run.
    pick = mo.ui.number(start=0, stop=cfg.eval.num_eval - 1, value=0, step=1, label="eval.py episode")
    pick
    return (pick,)


@app.cell
def _(OFFSET, dataset, eval_episodes, eval_starts, np, pick):
    # The chosen demo window: expert states and images from start to start + 25.
    ep_idx = int(eval_episodes[pick.value])
    start_step = int(eval_starts[pick.value])
    _chunk = dataset.load_chunk(np.array([ep_idx]), np.array([start_step]), np.array([start_step + OFFSET + 1]))[0]
    expert_states = np.asarray(_chunk["state"], dtype=np.float64)  # (26, 7)
    expert_pixels = _chunk["pixels"].permute(0, 2, 3, 1).numpy()  # (26, 224, 224, 3)
    return ep_idx, expert_pixels, expert_states, start_step


@app.cell
def _(OFFSET, ep_idx, expert_pixels, mo, plt, start_step):
    # Start and goal images, as the policy sees them.
    _fig, _ax = plt.subplots(1, 2, figsize=(7, 3.6))
    for _a, _img, _t in zip(_ax, (expert_pixels[0], expert_pixels[-1]), ("start", f"goal = expert {OFFSET} steps later")):
        _a.imshow(_img)
        _a.set_title(_t, fontsize=10)
        _a.axis("off")
    _fig.tight_layout()
    mo.vstack([mo.md(f"**Demo {ep_idx}, step {start_step} → {start_step + OFFSET}**"), _fig])
    return


@app.cell
def _(Any, gym, np):
    # The success rule. World counts an episode as solved (and stops it) the first
    # step `terminated` is True, so this wrapper decides success by replacing the
    # env's own `terminated`. For now it is a copy of PushT's eval_state:
    # agent + block position error < 20 (one 4-d distance) and block angle < 20 deg.
    # goal_state is what the _set_goal_state callable stored on the env.
    class GoalStateSuccess(gym.Wrapper):
        def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            obs, reward, _, truncated, info = self.env.step(action)
            env = self.env.unwrapped
            goal = np.asarray(env.goal_state, dtype=np.float64)
            cur = np.array([*env.agent.position, *env.block.position, env.block.angle], dtype=np.float64)
            pos_diff = np.linalg.norm(cur[:4] - goal[:4])
            angle_diff = np.abs(cur[4] - goal[4]) % (2 * np.pi)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            terminated = bool(pos_diff < 15 and angle_diff < np.pi / 9)
            return obs, reward, terminated, truncated, info

    return (GoalStateSuccess,)


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run this episode")
    run_btn
    return (run_btn,)


@app.cell
def _(expert_states, start_step):
    print(f"{start_step=}")
    print(f"{len(expert_states)=}")
    return


@app.cell
def _(expert_states, run_states):
    print(f"{expert_states[-1]=}")
    print(f"{run_states[-1]=}")
    return


@app.cell
def _(
    Any,
    BUDGET,
    GoalStateSuccess,
    OFFSET,
    OmegaConf,
    Path,
    cfg,
    dataset,
    ep_idx,
    mo,
    np,
    policy,
    run_btn,
    start_step,
    swm,
    tempfile,
):
    # Run the episode through world.evaluate, exactly like eval.py but with a
    # single env. The env's step is wrapped to record, after every step, the
    # true state (agent xy, block xy, block angle), the image and the success flag.
    mo.stop(not run_btn.value, mo.md("Press **Run this episode** to start."))

    def env_state(env: Any) -> np.ndarray:
        return np.array([*env.agent.position, *env.block.position, env.block.angle], dtype=np.float64)

    world = swm.World(
        cfg.world.env_name,
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=2 * BUDGET,
        extra_wrappers=[GoalStateSuccess],
    )
    world.set_policy(policy)
    _env = world.envs.envs[0].unwrapped
    log: dict[str, list] = {"state": [], "pixels": [], "success": []}
    _step = world.envs.step

    def logged_step(actions: np.ndarray, mask: np.ndarray | None = None) -> tuple:
        out = _step(actions, mask=mask)
        _pix = out[-1]["pixels"][0]
        log["state"].append(env_state(_env))
        log["pixels"].append(np.asarray(_pix[-1] if _pix.ndim > 3 else _pix).copy())
        log["success"].append(bool(out[2][0]))
        return out

    world.envs.step = logged_step
    video_dir = Path(tempfile.mkdtemp(prefix="one_run_25step_"))
    with mo.status.spinner(title="Planning and stepping (up to 50 env steps)..."):
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=[start_step],
            goal_offset=OFFSET,
            eval_budget=BUDGET,
            episodes_idx=[ep_idx],
            callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
            video=video_dir,
        )
    run_states = np.stack(log["state"])
    run_pixels = np.stack(log["pixels"])
    run_success = np.array(log["success"])
    world.close()
    return metrics, run_pixels, run_states, run_success, video_dir


@app.cell
def _(BUDGET, expert_states, metrics, mo, np, run_states, run_success):
    # Result summary: success or not, how many steps, and the final error.
    def errors(states: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """25-step success distances: (agent+block position error, block angle error in deg)."""
        pos = np.linalg.norm(states[:, :4] - goal[:4], axis=-1)
        d = np.abs(states[:, 4] - goal[4]) % (2 * np.pi)
        return pos, np.degrees(np.minimum(d, 2 * np.pi - d))

    goal = expert_states[-1]
    run_pos, run_ang = errors(run_states, goal)
    exp_pos, exp_ang = errors(expert_states, goal)
    _ok = bool(metrics["episode_successes"][0])
    _n = int(np.argmax(run_success)) + 1 if _ok else len(run_states)
    mo.md(
        f"""
    ## Result: {"✅ success" if _ok else "❌ failed"}

    | | Value |
    |---|---|
    | steps used | {_n} of {BUDGET} (the expert took 25) |
    | final position error (agent + block) | {run_pos[-1]:.0f} px (needs < 20) |
    | final block angle error | {run_ang[-1]:.0f}° (needs < 20°) |
    """
    )
    return exp_ang, exp_pos, run_ang, run_pos


@app.cell
def _(mo, run_pixels):
    # Slider to scrub through the run, one env step at a time.
    frame = mo.ui.slider(start=1, stop=len(run_pixels), value=len(run_pixels), step=1, label="Step", show_value=True, full_width=True)
    frame
    return (frame,)


@app.cell
def _(expert_pixels, expert_states, frame, np, plt, run_pixels, run_states):
    # Left: what the planner's run looks like at the chosen step, with the path
    # of the agent (orange) and block (blue) so far; dashed = what the expert did.
    # Right: the goal image.
    _s = 224 / 512  # world is 512 px, images are 224 px
    _fig, _ax = plt.subplots(1, 2, figsize=(8, 4))
    _ax[0].imshow(run_pixels[frame.value - 1])
    _path = np.vstack([expert_states[:1, :5], run_states[: frame.value]])
    _ax[0].plot(_path[:, 2] * _s, _path[:, 3] * _s, color="#2a78d6", lw=2, label="block (planner)")
    _ax[0].plot(_path[:, 0] * _s, _path[:, 1] * _s, color="#eb6834", lw=2, label="agent (planner)")
    _ax[0].plot(expert_states[:, 2] * _s, expert_states[:, 3] * _s, color="#2a78d6", lw=1, ls="--", alpha=0.7, label="block (expert)")
    _ax[0].plot(expert_states[:, 0] * _s, expert_states[:, 1] * _s, color="#eb6834", lw=1, ls="--", alpha=0.7, label="agent (expert)")
    _ax[0].legend(loc="upper right", fontsize=7, frameon=False)
    _ax[0].set_title(f"planner, step {frame.value}", fontsize=10)
    _ax[1].imshow(expert_pixels[-1])
    _ax[1].set_title("goal", fontsize=10)
    for _a in _ax:
        _a.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(exp_ang, exp_pos, plt, run_ang, run_pos):
    # Distance to the goal over time, planner vs expert. The run succeeds the
    # first step both curves are under their dotted 20 line.
    _fig, _ax = plt.subplots(1, 2, figsize=(9, 3))
    for _a, _run, _exp, _t in ((_ax[0], run_pos, exp_pos, "position error, agent + block (px)"), (_ax[1], run_ang, exp_ang, "block angle error (°)")):
        _a.plot(range(1, len(_run) + 1), _run, color="#2a78d6", lw=2, label="planner")
        _a.plot(range(len(_exp)), _exp, color="#888888", lw=1.5, ls="--", label="expert")
        _a.axhline(20, color="#c0392b", lw=1, ls=":", label="success line")
        _a.set_title(_t, fontsize=10)
        _a.set_xlabel("env step")
        _a.spines[["top", "right"]].set_visible(False)
    _ax[0].legend(fontsize=8, frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, video_dir):
    # The video world.evaluate saved (planner | expert | goal panels), same as eval.py writes.
    _mp4 = sorted(video_dir.glob("*.mp4"))
    mo.video(src=_mp4[0], controls=True) if _mp4 else mo.md(f"No video found in `{video_dir}`.")
    return


if __name__ == "__main__":
    app.run()
