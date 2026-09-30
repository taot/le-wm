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
    import torch
    from hydra import compose, initialize_config_dir

    return ROOT, compose, ev, initialize_config_dir, mo, np, plt, swm, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Push the T from any start to any goal

    You choose where the T starts (position and angle), where the agent starts, and where
    the T should end up. The agent's final position doesn't matter.

    The world model only plans well about 25 steps ahead, so the planner is not given the
    final goal directly. Instead it follows a chain of **waypoints**:

    1. **Reach the T.** While the agent is far from the T, the goal image keeps the T where it
       is and puts the agent one hop (120 px) closer to it.
    2. **Push.** Once the agent is next to the T, each time it plans, take where the T
       **actually** is and move it one short hop toward your goal (at most *hop px* and
       *hop °*). That pose is the waypoint. Once the goal is within one hop, the waypoint is
       the goal itself. The T at the waypoint is rendered with the agent at 8 different spots
       around it, and each plan is scored by its distance to the **closest** of those images,
       so the planner is not rewarded for parking the agent anywhere in particular.
    3. Plan (CEM), carry out the first 5 steps, and repeat, until the T is within 20 px and
       20° of your goal or the step budget runs out.

    **Note:** the model was trained on images that always show a green T at one fixed spot
    (256, 256, 45°). The env keeps drawing it there, so the model sees familiar pictures.
    Your goal only exists in the waypoint images and in the drawings below (dashed green T).

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
    MAX_STEPS = 2000  # env time limit; the step budget below must stay under it

    with initialize_config_dir(config_dir=str(ROOT / "config/eval"), version_base=None):
        cfg = compose("pusht_full", [f"policy={CHECKPOINT}", f"eval.img_size={IMG_SIZE}"])
    return MAX_STEPS, cfg


@app.cell
def _(cfg, ev):
    # Load the expert dataset (only used to fit the input scalers). Takes a little while.
    dataset = ev.get_dataset(cfg, cfg.eval.dataset_name)
    process = ev.fit_process(cfg, dataset)
    return (process,)


@app.cell
def _(mo):
    replan_25 = mo.ui.switch(label="Replan every 25 steps instead of every 5")
    replan_25
    return (replan_25,)


@app.cell
def _(cfg, ev, process, replan_25):
    # Build the CEM policy (the checkpoint is loaded onto the GPU here). CEM always
    # plans 25 steps ahead; by default only the first 5 are carried out before it
    # plans again from the new image (receding_horizon = 1 action block).
    from omegaconf import OmegaConf

    policy_cfg = cfg if replan_25.value else OmegaConf.merge(cfg, {"plan_config": {"receding_horizon": 1}})
    policy = ev.build_policy(policy_cfg, process)
    model = policy.solver.model
    return model, policy


@app.cell
def _(MAX_STEPS, cfg, swm):
    # The real env, and a copy used to render waypoint images and to try out plans.
    # No success wrapper: success is checked against your goal in the run loop.
    world = swm.World(cfg.world.env_name, num_envs=1, max_episode_steps=MAX_STEPS, image_shape=(224, 224))
    env = world.envs.envs[0].unwrapped
    sim_world = swm.World(cfg.world.env_name, num_envs=1, max_episode_steps=MAX_STEPS, image_shape=(224, 224))
    sim_env = sim_world.envs.envs[0].unwrapped
    world.reset(seed=cfg.seed)
    sim_world.reset(seed=cfg.seed)  # same seed = same physics settings
    return env, sim_env, world


@app.cell
def _(np):
    # Drawing: each picture is an inline SVG built from the env's true state, in
    # the env's own 512 x 512 px coordinates (no image files, so no flicker).
    import time
    from typing import Any

    FPS = 15  # playback speed of the live view, in env steps per second

    def _rgb(color: Any) -> str:
        return f"rgb({int(color[0])},{int(color[1])},{int(color[2])})"

    def _place(pts: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Points in the T's own frame -> board coordinates for pose = (x, y, angle)."""
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return np.stack([pose[0] + pts[:, 0] * c - pts[:, 1] * s, pose[1] + pts[:, 0] * s + pts[:, 1] * c], 1)

    def _svg_pts(pts: np.ndarray) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    def tee_corners(env: Any, pose: np.ndarray) -> np.ndarray:
        """The T's outline (8 corners, board coordinates) at pose = (x, y, angle)."""
        boxes = [np.array([(v.x, v.y) for v in sh.get_vertices()]) for sh in env.block.shapes]
        bar, stem = sorted(boxes, key=lambda b: -np.ptp(b[:, 0]))  # the bar is the wider box
        (bx0, by0), (bx1, by1) = bar.min(0), bar.max(0)
        (sx0, _), (sx1, sy1) = stem.min(0), stem.max(0)
        local = np.array([(bx0, by0), (bx1, by0), (bx1, by1), (sx1, by1), (sx1, sy1), (sx0, sy1), (sx0, by1), (bx0, by1)])
        return _place(local, pose)

    def _line(xy: list[np.ndarray], color: str) -> str:
        if len(xy) < 2:
            return ""
        return f'<polyline points="{_svg_pts(xy)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'

    def _dotted_tee(env: Any, pose: np.ndarray, color: str, dash: str) -> str:
        return f'<polygon points="{_svg_pts(tee_corners(env, pose))}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>'

    def scene_svg(
        env: Any,
        agent_xy: np.ndarray,
        block_pose: np.ndarray,
        target: np.ndarray | None = None,
        waypoint: np.ndarray | None = None,
        plan_block: np.ndarray | None = None,
        agent_trail: list[np.ndarray] = (),
        block_trail: list[np.ndarray] = (),
        push: tuple[np.ndarray, np.ndarray] | None = None,
        agent_goal: np.ndarray | None = None,
        size: int = 380,
    ) -> str:
        """One picture: your goal (dashed green T), the gray T, the blue agent, the
        current waypoint (dotted purple T), where the current plan would leave the T
        (dotted teal T), the paths so far (blue = agent, orange = T) and this step's
        push (red arrow to the point the agent is pulled toward). `agent_goal` is
        drawn as a purple ring: where the agent is asked to go next."""
        agent = next(iter(env.agent.shapes))
        parts = ['<rect x="0" y="0" width="512" height="512" fill="white" stroke="#ccc" stroke-width="4"/>']
        if target is not None:
            parts.append(
                f'<polygon points="{_svg_pts(tee_corners(env, target))}" fill="rgba(144,238,144,0.5)" '
                'stroke="rgb(40,150,40)" stroke-width="3" stroke-dasharray="10 5"/>'
            )
        for sh in env.block.shapes:
            box = np.array([(v.x, v.y) for v in sh.get_vertices()])
            parts.append(f'<polygon points="{_svg_pts(_place(box, block_pose))}" fill="{_rgb(sh.color)}"/>')
        parts.append(f'<circle cx="{agent_xy[0]:.1f}" cy="{agent_xy[1]:.1f}" r="{agent.radius:.1f}" fill="{_rgb(agent.color)}"/>')
        parts.append(_line(block_trail, "rgb(235,104,52)"))
        parts.append(_line(agent_trail, "rgb(42,120,214)"))
        if waypoint is not None:
            parts.append(_dotted_tee(env, waypoint, "rgb(140,60,200)", "8 6"))
        if plan_block is not None:
            parts.append(_dotted_tee(env, plan_block, "rgb(0,140,140)", "3 4"))
        if agent_goal is not None:
            parts.append(
                f'<circle cx="{agent_goal[0]:.1f}" cy="{agent_goal[1]:.1f}" r="{agent.radius:.1f}" fill="none" '
                'stroke="rgb(140,60,200)" stroke-width="3" stroke-dasharray="4 3"/>'
            )
        if push is not None:
            (x0, y0), (x1, y1) = push
            parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="rgb(200,30,30)" stroke-width="4"/>')
            parts.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="6" fill="none" stroke="rgb(200,30,30)" stroke-width="3"/>')
        return f'<svg viewBox="0 0 512 512" width="{size}" height="{size}" style="overflow:hidden">{"".join(parts)}</svg>'

    LEGEND = (
        "dashed green T = your goal · dotted purple T = current waypoint for the T · dotted purple "
        "circle = where the agent is asked to go (while it is still far from the T) · dotted teal T = where the "
        "current plan would leave the T (tried out in a copy of the env) · blue = agent path · "
        "orange = T path · red arrow = this step's push"
    )
    return FPS, LEGEND, scene_svg, tee_corners, time


@app.cell
def _(mo):
    # Your start and goal. Angles in degrees; the board is 512 x 512 px, y points down.
    inputs = mo.ui.dictionary({
        "t_start_x": mo.ui.number(start=0, stop=512, value=150, label="x"),
        "t_start_y": mo.ui.number(start=0, stop=512, value=370, label="y"),
        "t_start_deg": mo.ui.number(start=-180, stop=180, value=0, label="angle °"),
        "agent_x": mo.ui.number(start=0, stop=512, value=420, label="x"),
        "agent_y": mo.ui.number(start=0, stop=512, value=100, label="y"),
        "t_goal_x": mo.ui.number(start=0, stop=512, value=330, label="x"),
        "t_goal_y": mo.ui.number(start=0, stop=512, value=190, label="y"),
        "t_goal_deg": mo.ui.number(start=-180, stop=180, value=90, label="angle °"),
        "hop_px": mo.ui.number(start=5, stop=200, value=40, label="hop px"),
        "hop_deg": mo.ui.number(start=5, stop=180, value=25, label="hop °"),
        "budget": mo.ui.number(start=25, stop=1500, value=600, step=25, label="step budget"),
    })
    mo.vstack([
        mo.md("### Start and goal"),
        mo.hstack([mo.md("**T start**"), inputs["t_start_x"], inputs["t_start_y"], inputs["t_start_deg"]], justify="start"),
        mo.hstack([mo.md("**Agent start**"), inputs["agent_x"], inputs["agent_y"]], justify="start"),
        mo.hstack([mo.md("**T goal**"), inputs["t_goal_x"], inputs["t_goal_y"], inputs["t_goal_deg"]], justify="start"),
        mo.hstack([mo.md("**Waypoints and budget**"), inputs["hop_px"], inputs["hop_deg"], inputs["budget"]], justify="start"),
    ])
    return (inputs,)


@app.cell
def _(env, inputs, mo, np, scene_svg, sim_env, tee_corners):
    # Turn the inputs into env states, and check them before running.
    _v = inputs.value
    start_state = np.array([
        _v["agent_x"], _v["agent_y"], _v["t_start_x"], _v["t_start_y"], np.radians(_v["t_start_deg"]), 0.0, 0.0,
    ])  # agent xy, T xy, T angle, agent velocity xy: PushT's 7-number state
    target = np.array([_v["t_goal_x"], _v["t_goal_y"], np.radians(_v["t_goal_deg"])])

    problems = []
    for _name, _pose in (("start T", start_state[2:5]), ("goal T", target)):
        if not np.all((tee_corners(env, _pose) > 5) & (tee_corners(env, _pose) < 507)):
            problems.append(f"The {_name} sticks out of the board.")
    # Agent overlap: put the copy env in the start state and ask pymunk how far the
    # agent's center is from each part of the T.
    sim_env.block.velocity, sim_env.block.angular_velocity = (0.0, 0.0), 0.0
    sim_env._set_state(start_state)
    _radius = next(iter(sim_env.agent.shapes)).radius
    if min(sh.point_query(tuple(start_state[:2])).distance for sh in sim_env.block.shapes) < _radius:
        problems.append("The agent overlaps the start T.")

    mo.hstack([
        mo.Html(scene_svg(env, start_state[:2], start_state[2:5], target=target, size=300)),
        mo.md("\n\n".join(f"⚠️ {p}" for p in problems) if problems else "Start and goal look fine."),
    ], justify="start", align="start")
    return problems, start_state, target


@app.cell
def _(model, np, policy, sim_env, torch):
    # Waypoints, the images that stand for them, and how plans are scored.
    DEVICE = next(model.parameters()).device

    def pose_error(pose: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        """(position px, angle deg) between two T poses; the angle goes the short way round."""
        dang = (target[2] - pose[2] + np.pi) % (2 * np.pi) - np.pi
        return float(np.linalg.norm(target[:2] - pose[:2])), float(np.degrees(abs(dang)))

    def next_waypoint(pose: np.ndarray, target: np.ndarray, hop_px: float, hop_deg: float) -> np.ndarray:
        """One hop from `pose` toward `target`: at most hop_px straight toward it,
        and at most hop_deg of turning, the short way round."""
        d = target[:2] - pose[:2]
        dist = np.linalg.norm(d)
        xy = target[:2] if dist <= hop_px else pose[:2] + d / dist * hop_px
        dang = (target[2] - pose[2] + np.pi) % (2 * np.pi) - np.pi
        ang = pose[2] + np.clip(dang, -np.radians(hop_deg), np.radians(hop_deg))
        return np.array([*xy, ang])

    RING = 100  # px from the T's center of mass; every part of the T is within ~77 px of it
    AGENT_HOP = 120  # px the agent is asked to move per plan while it is still far from the T
    # The agent counts as "far" beyond this distance from the T's center. It is well
    # outside the ring on purpose: the planner only brings the agent to within ~50 px
    # of a spot it is asked to reach, so a tighter limit would never switch to pushing.
    FAR = RING + 70

    def t_center(pose: np.ndarray) -> np.ndarray:
        """The T's center of mass on the board, for the T at pose = (x, y, angle)."""
        c, s = np.cos(pose[2]), np.sin(pose[2])
        cog = np.array(sim_env.block.center_of_gravity)
        return pose[:2] + np.array([cog[0] * c - cog[1] * s, cog[0] * s + cog[1] * c])

    def ring_spots(pose: np.ndarray) -> list[np.ndarray]:
        """Up to 8 agent spots on a ring around the T at `pose`, clear of the T and
        on the board."""
        center = t_center(pose)
        spots = [center + RING * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        return [p for p in spots if np.all((p > 20) & (p < 492))]

    def render_goal(agent_xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Image of the agent at `agent_xy` and the T at `pose`, rendered the same way
        PushT renders its own goal image: _set_state (one tiny physics step, so the
        drawn shapes move too), then render."""
        sim_env.block.velocity, sim_env.block.angular_velocity = (0.0, 0.0), 0.0
        sim_env._set_state(np.concatenate([agent_xy, pose, [0.0, 0.0]]))
        return sim_env.render()

    def next_goal(
        t_pose: np.ndarray, agent_xy: np.ndarray, target: np.ndarray, hop_px: float, hop_deg: float
    ) -> tuple[str, np.ndarray, np.ndarray | None, list[np.ndarray]]:
        """What the next plan should aim for: (phase, T waypoint, agent goal or None, goal images).

        - "reach the T": the agent is farther than FAR from the T. The T stays
          put and the agent is asked to move one hop (AGENT_HOP) toward the nearest
          ring spot. One goal image.
        - "push": the agent is within FAR. The T is asked to move one hop
          toward your goal, with the agent anywhere on the ring around it (the
          closest of those goal images counts).
        """
        if np.linalg.norm(agent_xy - t_center(t_pose)) > FAR:
            nearest = min(ring_spots(t_pose), key=lambda p: np.linalg.norm(p - agent_xy))
            d = nearest - agent_xy
            agent_goal = nearest if np.linalg.norm(d) <= AGENT_HOP else agent_xy + d / np.linalg.norm(d) * AGENT_HOP
            return "reach the T", t_pose, agent_goal, [render_goal(agent_goal, t_pose)]
        waypoint = next_waypoint(t_pose, target, hop_px, hop_deg)
        return "push", waypoint, None, [render_goal(p, waypoint) for p in ring_spots(waypoint)]

    def encode_images(images: list[np.ndarray]) -> torch.Tensor:
        """Latent of each image, prepared the way the policy prepares its goal: (K, D)."""
        prepared = policy._prepare_info({"goal": np.stack(images)[:, None]})
        with torch.no_grad():
            return model.encode({"pixels": prepared["goal"].to(DEVICE)})["emb"][:, -1]

    # Plans are scored by the squared latent distance from the model's predicted
    # final image to the closest waypoint image. The run loop sets the images.
    goal_embs = {"current": None}

    def closest_goal_cost(info_dict: dict) -> torch.Tensor:
        pred = info_dict["predicted_emb"][..., -1, :]  # (B, S, D)
        return ((pred[:, :, None, :] - goal_embs["current"]) ** 2).sum(-1).min(-1).values

    model.criterion = closest_goal_cost  # used by model.get_cost, so by CEM

    def snapshot(e: object) -> dict[str, np.ndarray]:
        """Everything that moves: agent and T positions and speeds."""
        return {
            "agent_pos": np.array(e.agent.position), "agent_vel": np.array(e.agent.velocity),
            "block_pos": np.array(e.block.position), "block_angle": float(e.block.angle),
            "block_vel": np.array(e.block.velocity), "block_angvel": float(e.block.angular_velocity),
        }

    def play_in_sim(state: dict[str, np.ndarray], actions: np.ndarray) -> np.ndarray:
        """Put the copy env in `state`, play `actions` (env units, (n, 2)) and
        return the T's final pose (x, y, angle)."""
        sim_env.agent.position = tuple(state["agent_pos"])
        sim_env.agent.velocity = tuple(state["agent_vel"])
        # Angle before position: pymunk turns a body around its center of mass,
        # so setting the angle last would shift the T.
        sim_env.block.angle = state["block_angle"]
        sim_env.block.position = tuple(state["block_pos"])
        sim_env.block.velocity = tuple(state["block_vel"])
        sim_env.block.angular_velocity = state["block_angvel"]
        for a in actions:
            sim_env.step(a)
        return np.array([*sim_env.block.position, sim_env.block.angle])

    return encode_images, goal_embs, next_goal, play_in_sim, pose_error, render_goal, snapshot


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run")
    run_btn
    return (run_btn,)


@app.cell
def _(
    FPS,
    LEGEND,
    cfg,
    encode_images,
    env,
    goal_embs,
    inputs,
    mo,
    next_goal,
    np,
    play_in_sim,
    policy,
    pose_error,
    problems,
    run_btn,
    scene_svg,
    snapshot,
    start_state,
    target,
    time,
    world,
):
    # The run: plan toward the next waypoint, carry out part of the plan, repeat.
    mo.stop(not run_btn.value, mo.md("Set the start and goal, then press **Run**."))
    mo.stop(bool(problems), mo.md("Fix the start / goal first (see the warning above)."))

    BUDGET = int(inputs.value["budget"])
    HOP_PX, HOP_DEG = inputs.value["hop_px"], inputs.value["hop_deg"]
    ACTION_SCALE = env.action_scale  # PushT actions are moves of action * 100 px from the agent

    def t_pose() -> np.ndarray:
        return np.array([*env.block.position, env.block.angle])

    def live_view(step: int, status: str, push: tuple[np.ndarray, np.ndarray] | None) -> mo.Html:
        pos, ang = errors[-1]
        svg = scene_svg(env, np.array(env.agent.position), t_pose(), target, waypoint, plan_block, agent_trail, block_trail, push, agent_goal)
        return mo.Html(
            f"<div><b>{status}</b> · step {step} / {BUDGET} · plans made: {len(plan_log)} · phase: {phase or '-'} · "
            f"distance to your goal {pos:.0f} px, {ang:.0f}° (needs &lt; 20 px and &lt; 20°)</div>"
            f'<div style="margin:6px 0">{svg}</div><small>{LEGEND}</small>'
        )

    # Start state from your inputs. The goal_state option only sets PushT's own
    # goal image, which is replaced by the waypoint images below.
    world.reset(seed=cfg.seed, options={"state": start_state, "goal_state": start_state})
    world.set_policy(policy)  # also empties the policy's queue of planned actions

    frames = [env.render()]
    errors = [pose_error(t_pose(), target)]  # distance to your goal after each step
    agent_trail = [np.array(env.agent.position)]
    block_trail = [np.array(env.block.position)]
    plan_log = []  # one entry per plan
    waypoint = plan_block = agent_goal = None
    phase = ""
    success = errors[0][0] < 20 and errors[0][1] < 20
    mo.output.replace(live_view(0, "Starting", None))

    _last_shown = time.perf_counter()
    step = 0
    while not success and step < BUDGET:
        step += 1
        agent_before = np.array(env.agent.position)

        # A plan is due when the policy's queue of actions is empty. Aim it one
        # hop further: the agent toward the T, or (once it is there) the T toward your goal.
        new_plan = len(policy._action_buffer[0]) == 0
        if new_plan:
            phase, waypoint, agent_goal, _images = next_goal(t_pose(), np.array(env.agent.position), target, HOP_PX, HOP_DEG)
            goal_embs["current"] = encode_images(_images)
            world.infos["goal"][0, -1] = _images[0]  # what the policy's input shows as the goal
            _moment = {"step": step, "state": snapshot(env), "t_pose": t_pose(), "waypoint": waypoint, "phase": phase}
            mo.output.replace(live_view(step - 1, "Planning (CEM)...", None))

        # PushT flags `terminated` when it thinks its own goal is reached; the
        # policy would then stop acting. Success here is our own check instead.
        world.infos["terminated"][:] = False
        actions = policy.get_action(world.infos)  # shape (1, 2), in env units

        if new_plan:
            # The whole 25-step plan = this action + the queued ones + (when replanning
            # every 5 steps) the part kept as the next plan's starting guess. Those
            # are normalized actions, so undo the normalization.
            _queued = [a.numpy() for a in policy._action_buffer[0]]
            if policy.cfg.receding_horizon < policy.cfg.horizon:
                _queued += list(policy._next_init[0].reshape(-1, 2).numpy())
            _rest = policy.process["action"].inverse_transform(np.array(_queued)) if _queued else np.zeros((0, 2))
            _plan = np.vstack([actions.reshape(1, -1), _rest])
            plan_block = play_in_sim(_moment["state"], _plan)
            plan_log.append({**_moment, "plan_block": plan_block})

        _, _, _, _, world.infos = world.envs.step(actions)

        frames.append(env.render())
        errors.append(pose_error(t_pose(), target))
        agent_trail.append(np.array(env.agent.position))
        block_trail.append(np.array(env.block.position))
        success = errors[-1][0] < 20 and errors[-1][1] < 20

        # Show this step, paced to FPS so the run plays like a video.
        _push = (agent_before, agent_before + actions.reshape(-1) * ACTION_SCALE)
        _view = live_view(step, "Running", _push)
        time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - _last_shown)))
        mo.output.replace(_view)
        _last_shown = time.perf_counter()

    frames = np.stack(frames)
    live_view(step, "Done: reached the goal" if success else "Done: budget used up", None)
    return errors, frames, plan_log, step, success


@app.cell
def _(errors, inputs, mo, plan_log, policy, step, success):
    mo.md(f"""
    ## Result: {"✅ reached the goal" if success else "❌ did not reach the goal"}

    | | Value |
    |---|---|
    | settings | replan every {policy.cfg.receding_horizon * policy.cfg.action_block} steps · hop {inputs.value["hop_px"]} px, {inputs.value["hop_deg"]}° |
    | steps used | {step} of {inputs.value["budget"]} |
    | plans made | {len(plan_log)} |
    | distance to your goal at the start | {errors[0][0]:.0f} px, {errors[0][1]:.0f}° |
    | distance to your goal at the end | {errors[-1][0]:.0f} px, {errors[-1][1]:.0f}° (needs < 20 px and < 20°) |
    | closest it got | {min(e[0] for e in errors):.0f} px · {min(e[1] for e in errors):.0f}° (not necessarily at the same step) |
    """)
    return


@app.cell
def _(errors, plan_log, plt, pose_error, target):
    # Does each plan get closer? Line: the T's distance to your goal over time.
    # Purple dots: at each plan, where that plan would leave the T after its full
    # 25 steps (tried out in the copy env), drawn 25 steps after the plan starts.
    # Gray ticks: when a plan was made.
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for _i, (_a, _t) in enumerate(((_ax[0], "distance to your goal (px)"), (_ax[1], "angle to your goal (°)"))):
        for _p in plan_log:
            _a.axvline(_p["step"] - 1, color="#dddddd", lw=0.8)
        _a.plot(range(len(errors)), [e[_i] for e in errors], color="#2a78d6", lw=2, label="T")
        _a.scatter(
            [_p["step"] - 1 + 25 for _p in plan_log], [pose_error(_p["plan_block"], target)[_i] for _p in plan_log],
            color="#8c3cc8", s=14, zorder=3, label="plan's end point",
        )
        _a.axhline(20, color="#c0392b", lw=1, ls=":", label="success line")
        _a.set_title(_t, fontsize=10)
        _a.set_xlabel("env step")
        _a.spines[["top", "right"]].set_visible(False)
    _ax[0].legend(fontsize=8, frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(frames, mo):
    # Scrub through the run. These are the env's own images, i.e. what the model
    # sees: the green T in them is the fixed one, not your goal.
    frame = mo.ui.slider(start=0, stop=len(frames) - 1, value=len(frames) - 1, step=1, label="Step", show_value=True, full_width=True)
    frame
    return (frame,)


@app.cell
def _(frame, frames, plt):
    _fig, _ax = plt.subplots(figsize=(4, 4))
    _ax.imshow(frames[frame.value])
    _ax.set_title(f"env image, step {frame.value}", fontsize=10)
    _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
