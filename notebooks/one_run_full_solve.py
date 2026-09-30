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
def _(mo):
    # Two changes to try (see the checks at the end of the notebook for why).
    replan_switch = mo.ui.switch(label="Replan every 5 steps instead of every 25 (receding_horizon = 1)")
    multi_goal_switch = mo.ui.switch(
        label="Score plans against several goal images (agent at different spots), keep the closest"
    )
    mo.vstack([mo.md("### Planner changes to try"), replan_switch, multi_goal_switch])
    return multi_goal_switch, replan_switch


@app.cell
def _(cfg, ev):
    # Load the expert dataset (only used to fit the input scalers). Takes a little while.
    dataset = ev.get_dataset(cfg, cfg.eval.dataset_name)
    process = ev.fit_process(cfg, dataset)
    return (process,)


@app.cell
def _(cfg, ev, process, replan_switch):
    # Build the CEM policy (the checkpoint is loaded onto the GPU here).
    # Switch 1: CEM still plans 25 steps ahead, but only the first 5 are carried
    # out before planning again from the new image. The rest of the old plan is
    # the starting guess for the next one (warm start).
    from omegaconf import OmegaConf

    policy_cfg = OmegaConf.merge(cfg, {"plan_config": {"receding_horizon": 1}}) if replan_switch.value else cfg
    policy = ev.build_policy(policy_cfg, process)
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
def _(np, sim_env):
    # Save a moment of the real env, and replay actions from it in the copy env.
    # The replay matches the real env exactly (same physics, same settings).
    import copy

    def snapshot(e: object) -> dict[str, np.ndarray]:
        """Everything that moves: agent and block positions and speeds."""
        return {
            "agent_pos": np.array(e.agent.position), "agent_vel": np.array(e.agent.velocity),
            "block_pos": np.array(e.block.position), "block_angle": float(e.block.angle),
            "block_vel": np.array(e.block.velocity), "block_angvel": float(e.block.angular_velocity),
        }

    def play_in_sim(state: dict[str, np.ndarray], actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Put the copy env in `state`, play `actions` (env units, shape (n, 2)),
        and return the block's final pose (x, y, angle) and the final image."""
        sim_env.agent.position = tuple(state["agent_pos"])
        sim_env.agent.velocity = tuple(state["agent_vel"])
        # Angle before position: pymunk turns a body around its center of mass,
        # so setting the angle last would shift the block.
        sim_env.block.angle = state["block_angle"]
        sim_env.block.position = tuple(state["block_pos"])
        sim_env.block.velocity = tuple(state["block_vel"])
        sim_env.block.angular_velocity = state["block_angvel"]
        for a in actions:
            sim_env.step(a)
        return np.array([*sim_env.block.position, sim_env.block.angle]), sim_env.render()

    return copy, play_in_sim, snapshot


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
def _(env, goal_image, goal_state, mo, multi_goal_switch, np, plt, policy, sim_env):
    # How plans are scored. Normally: distance from the plan's predicted latent to
    # the goal image's latent. That goal image has the agent parked at one spot,
    # so the planner also gets rewarded for moving the agent there.
    # Switch 2: render the goal T with the agent at several spots, and score a plan
    # by its distance to the *closest* of those goal images.
    import torch

    model = policy.solver.model
    DEVICE = next(model.parameters()).device

    def encode_images(images: list[np.ndarray]) -> torch.Tensor:
        """Latent of each image, prepared the way the policy prepares its goal: (K, D)."""
        prepared = policy._prepare_info({"goal": np.stack(images)[:, None]})
        with torch.no_grad():
            return model.encode({"pixels": prepared["goal"].to(DEVICE)})["emb"][:, -1]

    def goal_images_with_agent_around() -> list[np.ndarray]:
        """The goal T with the agent at the usual spot, then at 8 spots on a ring
        100 px from the T's center of mass (every part of the T is within ~77 px of
        it, so the agent never overlaps the T), skipping spots off the board."""
        def render_goal(agent_xy: np.ndarray) -> np.ndarray:
            # Same path as PushT's reset uses for its goal image: _set_state (which
            # runs one tiny physics step, so the drawn shapes move too), then render.
            sim_env.block.velocity, sim_env.block.angular_velocity = (0.0, 0.0), 0.0
            sim_env._set_state(np.concatenate([agent_xy, env.goal_pose, [0.0, 0.0]]))
            return sim_env.render()

        images = [render_goal(np.asarray(goal_state[:2]))]
        center = np.array(sim_env.block.local_to_world(sim_env.block.center_of_gravity))
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            p = center + 100 * np.array([np.cos(a), np.sin(a)])
            if np.all((p > 20) & (p < 492)):
                images.append(render_goal(p))
        return images

    if multi_goal_switch.value:
        goal_images = goal_images_with_agent_around()
        goal_embs = encode_images(goal_images)

        def closest_goal_cost(info_dict: dict) -> torch.Tensor:
            """Squared latent distance from the plan's final prediction to the closest goal image: (B, S)."""
            pred = info_dict["predicted_emb"][..., -1, :]  # (B, S, D)
            return ((pred[:, :, None, :] - goal_embs) ** 2).sum(-1).min(-1).values

        model.criterion = closest_goal_cost  # used by model.get_cost, so by CEM
        goal_scoring = f"closest of {len(goal_images)} goal images"
        _fig, _ax = plt.subplots(1, len(goal_images), figsize=(1.3 * len(goal_images), 1.5))
        for _a, _img in zip(_ax, goal_images):
            _a.imshow(_img)
            _a.axis("off")
        _fig.tight_layout()
        _out = mo.vstack([mo.md(f"Plans are scored against the **{goal_scoring}**:"), _fig])
    else:
        goal_embs = encode_images([goal_image])
        model.__dict__.pop("criterion", None)  # back to the model's own criterion
        goal_scoring = "one goal image"
        _out = mo.md("Plans are scored against **one goal image** (the usual one).")
    _out
    return DEVICE, goal_embs, goal_scoring, model, torch


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
        other_block: np.ndarray | None = None,
        size: int = 380,
    ) -> str:
        """One PushT picture: green goal T, gray block, blue agent, plus the
        paths so far (blue = agent, orange = block), the current plan's target
        for the T (dotted purple T: where the block ends up if the whole plan is
        carried out) and this step's push (red arrow to the point the agent is
        pulled to). `other_block` adds a second dotted T, in teal."""
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
        if other_block is not None:
            parts.append(
                f'<polygon points="{_tee_outline(env, other_block)}" fill="none" '
                'stroke="rgb(0,140,140)" stroke-width="3" stroke-dasharray="3 4"/>'
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
    copy,
    env,
    ev,
    goal_scoring,
    goal_state,
    mo,
    np,
    play_in_sim,
    policy,
    reset_episode,
    run_btn,
    scene_svg,
    snapshot,
    time,
    world,
):
    # The evaluation loop, written out. This is what world.evaluate does for
    # each env (see World._run_iter in stable_worldmodel).
    mo.stop(not run_btn.value, mo.md("Press **Run this episode** to start."))
    goal_scoring  # the goal-scoring rule (switch 2) must be set before the run

    ACTION_SCALE = env.action_scale  # PushT actions are moves of action * 100 px from the agent
    _goal_svg = scene_svg(env, goal_state[:2], goal_state[2:5])
    _legend = (
        "blue = agent path · orange = block path · dotted purple T = the current plan's target "
        "for the T (where the whole 25-step plan would leave the block, tried out in a copy of the env) · "
        "red arrow = this step's push"
    )

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
    plan_log = []  # one entry per plan, for the checks at the end of the notebook
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
            _moment = {"step": step, "state": snapshot(env), "infos": copy.deepcopy(world.infos)}
            mo.output.replace(live_view(step - 1, "Planning (CEM)...", None))
        actions = policy.get_action(world.infos)  # shape (1, 2), in env units

        if new_plan:
            # The whole new plan = this action + the ones still queued + (with switch 1)
            # the part that won't be carried out, kept as the next plan's starting
            # guess. Those are normalized actions, so undo the normalization.
            _queued = [a.numpy() for a in policy._action_buffer[0]]
            if policy.cfg.receding_horizon < policy.cfg.horizon:
                _queued += list(policy._next_init[0].reshape(-1, 2).numpy())
            _rest = policy.process["action"].inverse_transform(np.array(_queued)) if _queued else np.zeros((0, 2))
            _plan = np.vstack([actions.reshape(1, -1), _rest])  # (25, 2), env units
            plan_block, _ = play_in_sim(_moment["state"], _plan)
            plan_log.append({**_moment, "actions": _plan, "plan_block": plan_block})

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
    return ang_err, frames, plan_log, pos_err, replanned, steps_used, success


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run this episode")
    run_btn
    return (run_btn,)


@app.cell
def _(BUDGET, ang_err, goal_scoring, mo, policy, pos_err, replanned, steps_used, success):
    mo.md(f"""
    ## Result: {"✅ success" if success else "❌ failed"}

    | | Value |
    |---|---|
    | settings | replan every {policy.cfg.receding_horizon * policy.cfg.action_block} steps · scored against {goal_scoring} |
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Is it CEM? Two checks

    CEM picks the plan whose **imagined** result is closest to the goal: the world model
    predicts the latent (embedding) after the plan's 25 steps, and the cost is the squared
    distance between that and the goal image's latent. Lower is better.

    - **Check 1: does the model imagine the right result?** For every plan of the run,
      compare the cost the model imagined for CEM's plan with the cost of what really
      happened (the image after playing the plan in the copy env, run through the same
      encoder). A big gap means the model's predictions are off.
    - **Check 2: is there a better plan CEM missed?** At one planning moment, search for a
      plan using the real physics (CEM again, but scored by the true block error), then
      ask the model to score both plans. If the model prefers the better plan, CEM's search
      is the problem. If it prefers CEM's plan, the cost or the model is the problem.
    """)
    return


@app.cell
def _(DEVICE, cfg, goal_embs, model, np, policy, torch):
    # Score plans with the model, the same way CEM does (model.get_cost), and
    # score real images with the same encoder and the same goal rule.
    HORIZON = cfg.plan_config.horizon  # 5 action blocks
    BLOCK = cfg.plan_config.action_block  # 5 env steps per block

    def to_plan_tensor(actions: np.ndarray) -> torch.Tensor:
        """(25, 2) env-unit actions -> (5, 10) normalized, the shape CEM works in."""
        norm = policy.process["action"].transform(actions)
        return torch.as_tensor(norm, dtype=torch.float32).reshape(HORIZON, BLOCK * norm.shape[-1])

    def model_costs(infos: dict, plans: list[np.ndarray]) -> np.ndarray:
        """Imagined cost of each plan (env-unit actions), from the moment `infos`."""
        cand = torch.stack([to_plan_tensor(p) for p in plans])[None].to(DEVICE)  # (1, N, 5, 10)
        prepared = policy._prepare_info(infos)
        batch = {}
        for k, v in prepared.items():  # repeat the moment once per plan, like CEM does
            if torch.is_tensor(v):
                v = v.to(DEVICE, dtype=torch.float32 if v.is_floating_point() else None)
                batch[k] = v.unsqueeze(1).expand(1, len(plans), *v.shape[1:])
            elif isinstance(v, np.ndarray):
                batch[k] = np.repeat(v[:, None], len(plans), axis=1)
            else:
                batch[k] = v
        with torch.no_grad():
            return model.get_cost(batch, cand)[0].cpu().numpy()

    def real_cost(infos: dict, image: np.ndarray) -> float:
        """Cost of a real image: squared latent distance to the (closest) goal image."""
        prepared = policy._prepare_info({**infos, "pixels": image[None, None]})
        with torch.no_grad():
            emb = model.encode({"pixels": prepared["pixels"].to(DEVICE)})["emb"][:, -1]
        return float(((emb - goal_embs) ** 2).sum(-1).min())

    return model_costs, real_cost


@app.cell
def _(env, ev, mo, model_costs, np, plan_log, play_in_sim, plt, real_cost):
    # Check 1, for every plan of the run:
    #   now      = cost of the image at the planning moment (the starting point)
    #   imagined = cost the model predicts for CEM's plan
    #   real     = cost of the image the plan really produces
    import types

    def block_error(pose: np.ndarray) -> tuple[float, float]:
        """(position px, angle deg) of a block pose (x, y, angle) from the green T."""
        _fake_env = types.SimpleNamespace(block=types.SimpleNamespace(position=pose[:2], angle=pose[2]), goal_pose=env.goal_pose)
        pos, ang = ev.block_goal_error(_fake_env)
        return pos, np.degrees(ang)

    check1 = []
    for _p in plan_log:
        _, _img = play_in_sim(_p["state"], _p["actions"])
        check1.append({
            "step": _p["step"],
            "now": real_cost(_p["infos"], _p["infos"]["pixels"][0, -1]),
            "imagined": float(model_costs(_p["infos"], [_p["actions"]])[0]),
            "real": real_cost(_p["infos"], _img),
            "err_before": block_error(np.array([*_p["state"]["block_pos"], _p["state"]["block_angle"]])),
            "err_after": block_error(_p["plan_block"]),
        })

    _fig, _ax = plt.subplots(figsize=(7, 3))
    _x = np.arange(1, len(check1) + 1)
    for _key, _c, _ls in (("now", "#888888", ":"), ("imagined", "#8c3cc8", "-"), ("real", "#2a78d6", "-")):
        _ax.plot(_x, [r[_key] for r in check1], color=_c, ls=_ls, marker="o", lw=2, label=_key)
    _ax.set_xlabel("plan")
    _ax.set_ylabel("cost (latent distance to goal)")
    _ax.set_xticks(_x)
    _ax.legend(fontsize=8, frameon=False)
    _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()

    _rows = "\n".join(
        f"| {i} | {r['step']} | {r['now']:.1f} | {r['imagined']:.1f} | {r['real']:.1f} | "
        f"{r['err_before'][0]:.0f} px, {r['err_before'][1]:.0f}° → {r['err_after'][0]:.0f} px, {r['err_after'][1]:.0f}° |"
        for i, r in enumerate(check1, 1)
    )
    mo.vstack([
        mo.md("### Check 1: imagined vs real cost of each plan"),
        _fig,
        mo.md(
            "| plan | at step | now | imagined | real | block error before → after |\n"
            "|---|---|---|---|---|---|\n" + _rows + "\n\n"
            "How to read it: **imagined well below real** means the model is too optimistic about "
            "its plans (a model problem). **real close to now** means the plan barely changed "
            "anything, as far as the cost can tell."
        ),
    ])
    return (block_error,)


@app.cell
def _(mo, plan_log):
    # Check 2 settings: which plan to test, and a button (it takes a little while).
    check2_pick = mo.ui.number(start=1, stop=len(plan_log), value=1, step=1, label="plan to test")
    check2_btn = mo.ui.run_button(label="Search the real physics for a better plan")
    mo.vstack([mo.md("### Check 2: is there a better plan CEM missed?"), mo.hstack([check2_pick, check2_btn], justify="start")])
    return check2_btn, check2_pick


@app.cell
def _(
    block_error,
    check2_btn,
    check2_pick,
    env,
    mo,
    model_costs,
    np,
    plan_log,
    play_in_sim,
    policy,
    real_cost,
    scene_svg,
):
    # Check 2. Physics search = CEM, but each candidate plan is played in the copy
    # env and scored by the true block error (position / 20 px + angle / 20 deg,
    # so 2 = both exactly on the success line). It starts from CEM's own plan, so
    # it can only do better. The agent's position is ignored here.
    mo.stop(not check2_btn.value, mo.md("Pick a plan and press the button."))

    SAMPLES, ITERS, ELITES = 64, 10, 8
    moment = plan_log[check2_pick.value - 1]

    def physics_cost(actions: np.ndarray) -> float:
        pos, ang = block_error(play_in_sim(moment["state"], actions)[0])
        return pos / 20 + ang / 20

    _rng = np.random.default_rng(0)
    _mean = moment["actions"].copy()
    _std = np.broadcast_to(policy.process["action"].scale_, _mean.shape).copy()  # the demos' action spread
    best_actions, best_cost = _mean.copy(), physics_cost(_mean)
    with mo.status.progress_bar(total=ITERS, title="Searching the real physics") as _bar:
        for _ in range(ITERS):
            _cands = _mean + _std * _rng.standard_normal((SAMPLES, *_mean.shape))
            _cands[0] = _mean
            _costs = np.array([physics_cost(c) for c in _cands])
            _elite = _cands[np.argsort(_costs)[:ELITES]]
            _mean, _std = _elite.mean(0), _elite.std(0) + 1e-3
            if _costs.min() < best_cost:
                best_actions, best_cost = _cands[_costs.argmin()].copy(), float(_costs.min())
            _bar.update()

    _plans = {"CEM's plan": moment["actions"], "physics plan": best_actions}
    _imagined = model_costs(moment["infos"], list(_plans.values()))
    _results = {}
    for (_name, _acts), _im in zip(_plans.items(), _imagined):
        _pose, _img = play_in_sim(moment["state"], _acts)
        _results[_name] = {"pose": _pose, "err": block_error(_pose), "imagined": float(_im), "real": real_cost(moment["infos"], _img)}
    _cem, _phys = _results["CEM's plan"], _results["physics plan"]

    _cem_score = _cem["err"][0] / 20 + _cem["err"][1] / 20
    if best_cost > 0.9 * _cem_score:
        verdict = ("Even searching the real physics finds no plan that moves the T much closer in 25 steps. "
                   "The limit here is the short plan (25 steps), not CEM.")
    elif _phys["imagined"] < _cem["imagined"]:
        verdict = ("The model rates the physics plan as better, but CEM didn't find it. "
                   "This points to **CEM's search** (try more samples or rounds).")
    elif _phys["real"] < _cem["real"]:
        verdict = ("The model *imagines* CEM's plan as better, but on the real images the physics plan scores "
                   "better (lower real cost). CEM did its job and the cost is fine here: the **model's "
                   "predictions** are off.")
    else:
        verdict = ("Even on the real images the cost prefers CEM's plan, though the physics plan moves the T "
                   "closer. CEM did its job: the **cost** is steering it wrong (it also counts the agent's "
                   "position, and the goal image has the agent parked in a fixed spot).")

    _svg = scene_svg(
        env, moment["state"]["agent_pos"], np.array([*moment["state"]["block_pos"], moment["state"]["block_angle"]]),
        plan_block=_cem["pose"], other_block=_phys["pose"], size=320,
    )
    _rows = "\n".join(
        f"| {n} | {r['err'][0]:.0f} px, {r['err'][1]:.0f}° | {r['imagined']:.1f} | {r['real']:.1f} |"
        for n, r in _results.items()
    )
    mo.hstack([
        mo.Html(f'<div style="flex:none;width:320px">{_svg}<br><small>purple dotted T = CEM\'s plan · teal dotted T = physics plan</small></div>'),
        mo.md(
            f"Plan {check2_pick.value} (made at step {moment['step']})\n\n"
            "| | block error after 25 steps | model's imagined cost | real cost |\n"
            "|---|---|---|---|\n" + _rows + f"\n\n**Verdict:** {verdict}"
        ),
    ], justify="start", align="start")
    return


if __name__ == "__main__":
    app.run()
