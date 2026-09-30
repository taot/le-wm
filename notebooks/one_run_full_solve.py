import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Imports for the whole notebook. eval.py lives in the repo root, so the
    # root is put on sys.path first. eval.py must be imported before torch:
    # it sets MUJOCO_GL and caps torch's CPU threads on import.
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    import eval as ev  # noqa: E402  (sets env vars before torch loads)
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import stable_worldmodel as swm
    from hydra import compose, initialize_config_dir

    return ROOT, compose, ev, initialize_config_dir, mo, np, plt, swm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One full-solve episode, step by step

    This runs **one** episode of the full-solve eval (`config/eval/pusht_full.yaml`,
    `eval.py:evaluate_full_solve`), but writes the loop out by hand instead of calling
    `world.evaluate`, so you can see every piece:

    1. **start**: a random PushT start (agent, block position and angle), picked by the seed;
    2. **goal**: the block sitting on the green T (`env.goal_pose`), with the agent parked at
       `eval.goal_agent_pos`. The env renders that state once, and that image is what the
       policy is asked to reach;
    3. **loop**: ask the policy for an action, step the env, repeat, for at most
       `eval_budget` = 300 steps;
    4. **success**: the block alone is within 20 px and 20° of the green T
       (`eval.BlockOnGoalSuccess`); the agent position does not matter. The episode stops
       the first step this is true.

    Compared with the 25-step eval: there the start and goal come from an expert demo
    25 steps apart and the budget is 50; here the goal is always the green T, and the
    start can be far from it.

    Needs a CUDA GPU and a checkpoint.
    """)
    return


@app.cell
def _(ROOT, compose, initialize_config_dir):
    # Settings. CHECKPOINT is relative to $STABLEWM_HOME/checkpoints/
    # (default ~/.stable_worldmodel/checkpoints/). IMG_SIZE must match the
    # image size the checkpoint was trained with (112 for an img112 run).
    CHECKPOINT = "pusht/2026-09-28_114752_img112_s3072/weights_epoch_003.pt"
    IMG_SIZE = 112

    # Load config/eval/pusht_full.yaml exactly as eval.py would see it with these overrides.
    with initialize_config_dir(config_dir=str(ROOT / "config/eval"), version_base=None):
        cfg = compose("pusht_full", [f"policy={CHECKPOINT}", f"eval.img_size={IMG_SIZE}"])
    BUDGET = cfg.eval.eval_budget  # 300
    return BUDGET, cfg


@app.cell
def _(cfg, ev):
    # Load the expert dataset (only used to fit the input scalers), and build
    # the CEM policy (the checkpoint is loaded onto the GPU here). Takes a little while.
    dataset = ev.get_dataset(cfg, cfg.eval.dataset_name)
    process = ev.fit_process(cfg, dataset)
    policy = ev.build_policy(cfg, process)
    return (policy,)


@app.cell
def _(BUDGET, cfg, ev, swm):
    # One env, wrapped exactly like eval.py's full-solve world: episodes are cut
    # at BUDGET steps, and BlockOnGoalSuccess replaces PushT's own success rule.
    world = swm.World(
        cfg.world.env_name,
        num_envs=1,
        max_episode_steps=BUDGET,
        image_shape=(224, 224),
        extra_wrappers=[ev.BlockOnGoalSuccess],
    )
    env = world.envs.envs[0].unwrapped  # the raw PushT env, to read its true state

    # A second copy of the env, used only to try out each new plan: its actions
    # are played there to see where they would leave the T (see the run loop).
    sim_world = swm.World(cfg.world.env_name, num_envs=1, max_episode_steps=BUDGET, image_shape=(224, 224))
    sim_env = sim_world.envs.envs[0].unwrapped
    return env, sim_env, sim_world, world


@app.cell
def _(cfg, mo):
    # eval.py runs num_eval (8) envs at once with seed 42; env k gets seed 42 + k.
    # So "episode k" here has the same start as episode k of an eval.py run.
    pick = mo.ui.number(start=0, stop=cfg.eval.num_eval - 1, value=0, step=1, label="eval.py episode")
    pick
    return (pick,)


@app.cell
def _(cfg, env, np, pick, sim_world, world):
    # The goal state (7 numbers): agent xy, block xy, block angle, agent velocity xy.
    # The block part is the green T; the agent part is a fixed parking spot.
    seed = cfg.seed + pick.value
    world.reset(seed=seed)  # a first reset just to read the green-T pose
    goal_state = np.concatenate([cfg.eval.goal_agent_pos, env.goal_pose, [0.0, 0.0]])

    def reset_episode() -> None:
        """Random start from `seed`; the goal image is rendered from `goal_state`."""
        world.reset(seed=seed, options={"goal_state": goal_state})
        sim_world.reset(seed=seed, options={"goal_state": goal_state})  # same physics settings

    reset_episode()
    start_image = env.render()
    goal_image = env._goal  # the image the policy is asked to reach
    return goal_image, goal_state, reset_episode, seed, start_image


@app.cell
def _(goal_image, goal_state, mo, np, plt, seed, start_image):
    _fig, _ax = plt.subplots(1, 2, figsize=(7, 3.6))
    _ax[0].imshow(start_image)
    _ax[0].set_title(f"start (seed {seed})", fontsize=10)
    _ax[1].imshow(goal_image)
    _ax[1].set_title("goal image", fontsize=10)
    for _a in _ax:
        _a.axis("off")
    _fig.tight_layout()
    mo.vstack([
        mo.md(
            f"Goal: block at ({goal_state[2]:.0f}, {goal_state[3]:.0f}) px, "
            f"angle {np.degrees(goal_state[4]):.0f}°; agent parked at "
            f"({goal_state[0]:.0f}, {goal_state[1]:.0f})."
        ),
        _fig,
    ])
    return


@app.cell
def _(np):
    # Drawing for the live view. Each step is drawn as an inline SVG built from
    # the env's true state (no image files), so the page can swap one frame for
    # the next without a blank flash. Coordinates are the env's own 512 x 512 px.
    import time
    from typing import Any

    FPS = 15  # playback speed of the live view, in env steps per second
    GREEN = "rgb(144,238,144)"  # PushT's goal T color (LightGreen)

    def _rgb(color: Any) -> str:
        return f"rgb({int(color[0])},{int(color[1])},{int(color[2])})"

    def _poly_points(env: Any, pose: np.ndarray) -> list[str]:
        """The T's two boxes, placed at pose = (x, y, angle), as SVG point lists."""
        c, s = np.cos(pose[2]), np.sin(pose[2])
        out = []
        for shape in env.block.shapes:  # vertices in the block's own frame
            pts = [(pose[0] + v.x * c - v.y * s, pose[1] + v.x * s + v.y * c) for v in shape.get_vertices()]
            out.append(" ".join(f"{x:.1f},{y:.1f}" for x, y in pts))
        return out

    def _tee_outline(env: Any, pose: np.ndarray) -> str:
        """The T's outline (one 8-corner shape) placed at pose = (x, y, angle)."""
        boxes = [np.array([(v.x, v.y) for v in sh.get_vertices()]) for sh in env.block.shapes]
        bar, stem = sorted(boxes, key=lambda b: -np.ptp(b[:, 0]))  # the bar is the wider box
        (bx0, by0), (bx1, by1) = bar.min(0), bar.max(0)
        (sx0, _), (sx1, sy1) = stem.min(0), stem.max(0)
        local = [(bx0, by0), (bx1, by0), (bx1, by1), (sx1, by1), (sx1, sy1), (sx0, sy1), (sx0, by1), (bx0, by1)]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return " ".join(f"{pose[0] + x * c - y * s:.1f},{pose[1] + x * s + y * c:.1f}" for x, y in local)

    def _line(xy: list[np.ndarray] | np.ndarray, color: str) -> str:
        if len(xy) < 2:
            return ""
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'

    def scene_svg(
        env: Any,
        agent_xy: np.ndarray,
        block_pose: np.ndarray,
        agent_trail: list[np.ndarray] = (),
        block_trail: list[np.ndarray] = (),
        plan_block: np.ndarray | None = None,
        push: tuple[np.ndarray, np.ndarray] | None = None,
        size: int = 380,
    ) -> str:
        """One PushT picture: green goal T, gray block, blue agent, plus the
        paths so far (blue = agent, orange = block), the current plan's target
        for the T (dotted purple T: where the block ends up if the whole plan is
        carried out) and this step's push (red arrow to the point the agent is
        pulled to)."""
        agent = env.agent.shapes
        radius = next(iter(agent)).radius
        parts = [f'<rect x="0" y="0" width="512" height="512" fill="white" stroke="#ccc" stroke-width="4"/>']
        parts += [f'<polygon points="{p}" fill="{GREEN}"/>' for p in _poly_points(env, env.goal_pose)]
        parts += [f'<polygon points="{p}" fill="{_rgb(sh.color)}"/>' for sh, p in zip(env.block.shapes, _poly_points(env, block_pose))]
        parts.append(f'<circle cx="{agent_xy[0]:.1f}" cy="{agent_xy[1]:.1f}" r="{radius:.1f}" fill="{_rgb(next(iter(agent)).color)}"/>')
        parts.append(_line(block_trail, "rgb(235,104,52)"))
        parts.append(_line(agent_trail, "rgb(42,120,214)"))
        if plan_block is not None:
            parts.append(
                f'<polygon points="{_tee_outline(env, plan_block)}" fill="none" '
                'stroke="rgb(140,60,200)" stroke-width="3" stroke-dasharray="8 6"/>'
            )
        if push is not None:
            (x0, y0), (x1, y1) = push
            parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="rgb(200,30,30)" stroke-width="4"/>')
            parts.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="6" fill="none" stroke="rgb(200,30,30)" stroke-width="3"/>')
        return f'<svg viewBox="0 0 512 512" width="{size}" height="{size}" style="overflow:hidden">{"".join(parts)}</svg>'

    return FPS, scene_svg, time


@app.cell
def _(
    BUDGET,
    FPS,
    env,
    ev,
    goal_state,
    mo,
    np,
    policy,
    reset_episode,
    run_btn,
    scene_svg,
    sim_env,
    time,
    world,
):
    # The evaluation loop, written out. This is what world.evaluate does for
    # each env (see World._run_iter in stable_worldmodel).
    mo.stop(not run_btn.value, mo.md("Press **Run this episode** to start."))

    ACTION_SCALE = env.action_scale  # PushT actions are moves of action * 100 px from the agent
    _goal_svg = scene_svg(env, goal_state[:2], goal_state[2:5])
    _legend = (
        "blue = agent path · orange = block path · dotted purple T = the current plan's target "
        "for the T (where the plan leaves the block, tried out in a copy of the env) · "
        "red arrow = this step's push"
    )

    def try_plan(plan_actions: np.ndarray) -> np.ndarray:
        """Play a plan's actions in the copy env, starting from the real env's
        current state, and return where the block ends up: (x, y, angle)."""
        sim_env.agent.position = env.agent.position
        sim_env.agent.velocity = env.agent.velocity
        sim_env.block.position = env.block.position
        sim_env.block.angle = env.block.angle
        sim_env.block.velocity = env.block.velocity
        sim_env.block.angular_velocity = env.block.angular_velocity
        for a in plan_actions:
            sim_env.step(a)
        return np.array([*sim_env.block.position, sim_env.block.angle])

    def live_view(step: int, status: str, push: tuple[np.ndarray, np.ndarray] | None) -> mo.Html:
        """The env right now (left) next to the goal (right), with a status line."""
        pos, ang = block_err[-1]
        now = scene_svg(env, np.array(env.agent.position), np.array([*env.block.position, env.block.angle]), agent_trail, block_trail, plan_block, push)
        return mo.Html(
            f"<div><b>{status}</b> · step {step} / {BUDGET} · plans made: {len(replanned)} · "
            f"block error {pos:.0f} px, {np.degrees(ang):.0f}° (needs &lt; 20 px and &lt; 20°)</div>"
            f'<div style="display:flex;gap:12px;margin:6px 0">{now}{_goal_svg}</div>'
            f"<small>{_legend}</small>"
        )

    reset_episode()
    world.set_policy(policy)  # also empties the policy's queue of planned actions

    frames = [env.render()]
    block_err = [ev.block_goal_error(env)]  # (position px, angle rad) after each step
    agent_trail = [np.array(env.agent.position)]
    block_trail = [np.array(env.block.position)]
    replanned = []  # env steps where the policy ran CEM for a new plan
    plan_block = None  # where the current plan would leave the block: (x, y, angle)
    success = False
    mo.output.replace(live_view(0, "Starting", None))

    _last_shown = time.perf_counter()
    for step in range(1, BUDGET + 1):
        agent_before = np.array(env.agent.position)

        # The policy keeps a queue of planned actions. When it is empty, it
        # plans again (CEM) from the current image toward the goal image.
        new_plan = len(policy._action_buffer[0]) == 0
        if new_plan:
            replanned.append(step)
            mo.output.replace(live_view(step - 1, "Planning (CEM)...", None))
        actions = policy.get_action(world.infos)  # shape (1, 2), in env units

        if new_plan:
            # The whole new plan = this action + the ones still queued (the queue
            # holds normalized actions, so undo the normalization to get env units).
            _queued = [a.numpy() for a in policy._action_buffer[0]]
            _rest = policy.process["action"].inverse_transform(np.array(_queued)) if _queued else np.zeros((0, 2))
            plan_block = try_plan(np.vstack([actions.reshape(1, -1), _rest]))

        # One env step. `terminated` comes from BlockOnGoalSuccess;
        # `truncated` becomes True when BUDGET steps are used up.
        _, _, terminated, truncated, world.infos = world.envs.step(actions)

        frames.append(env.render())
        block_err.append(ev.block_goal_error(env))
        agent_trail.append(np.array(env.agent.position))
        block_trail.append(np.array(env.block.position))

        # Show this step. Steps between plans are very fast, so wait until
        # 1 / FPS s have passed since the last frame: the run plays like a video.
        _push = (agent_before, agent_before + actions.reshape(-1) * ACTION_SCALE)
        _view = live_view(step, "Running", _push)
        time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - _last_shown)))
        mo.output.replace(_view)
        _last_shown = time.perf_counter()

        if terminated[0]:
            success = True
            break
        if truncated[0]:
            break

    frames = np.stack(frames)
    pos_err = np.array([e[0] for e in block_err])
    ang_err = np.degrees([e[1] for e in block_err])
    steps_used = step
    live_view(steps_used, "Done: success" if success else "Done: failed", None)
    return ang_err, frames, pos_err, replanned, steps_used, success


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run this episode")
    run_btn
    return (run_btn,)


@app.cell
def _(BUDGET, ang_err, mo, pos_err, replanned, steps_used, success):
    mo.md(f"""
    ## Result: {"✅ success" if success else "❌ failed"}

    | | Value |
    |---|---|
    | steps used | {steps_used} of {BUDGET} (expert demos take ~120) |
    | times the policy planned | {len(replanned)} |
    | final block position error | {pos_err[-1]:.0f} px (needs < 20) |
    | final block angle error | {ang_err[-1]:.0f}° (needs < 20°) |
    """)
    return


@app.cell
def _(frames, mo):
    # Slider to scrub through the run, one env step at a time (0 = start).
    frame = mo.ui.slider(start=0, stop=len(frames) - 1, value=len(frames) - 1, step=1, label="Step", show_value=True, full_width=True)
    frame
    return (frame,)


@app.cell
def _(frame, frames, goal_image, plt):
    # Left: the run at the chosen step. Right: the goal image.
    _fig, _ax = plt.subplots(1, 2, figsize=(8, 4))
    _ax[0].imshow(frames[frame.value])
    _ax[0].set_title(f"run, step {frame.value}", fontsize=10)
    _ax[1].imshow(goal_image)
    _ax[1].set_title("goal", fontsize=10)
    for _a in _ax:
        _a.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(ang_err, plt, pos_err, replanned):
    # Block distance to the green T over time. The run succeeds the first step
    # both curves are under their dotted line. Gray ticks = the policy planned.
    _fig, _ax = plt.subplots(1, 2, figsize=(9, 3))
    for _a, _err, _t in ((_ax[0], pos_err, "block position error (px)"), (_ax[1], ang_err, "block angle error (°)")):
        for _s in replanned:
            _a.axvline(_s - 1, color="#bbbbbb", lw=0.8)
        _a.plot(range(len(_err)), _err, color="#2a78d6", lw=2)
        _a.axhline(20, color="#c0392b", lw=1, ls=":", label="success line")
        _a.set_title(_t, fontsize=10)
        _a.set_xlabel("env step")
        _a.spines[["top", "right"]].set_visible(False)
    _ax[0].legend(fontsize=8, frameon=False)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
