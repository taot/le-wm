import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Imports for the whole notebook.
    # Importing stable_worldmodel registers the "swm/PushT-v1" env with gymnasium,
    # so gym.make() can find it later. Nothing from it is used directly.
    import io

    import gymnasium as gym
    import lance
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import stable_worldmodel  # noqa: F401  registers swm/PushT-v1
    from PIL import Image

    return Image, gym, io, lance, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    # Intro text: what the two eval modes are and a table of how they differ
    # (start state, goal, success rule, step budget, number of episodes).
    mo.md(r"""
    # 25-step episodes vs full-solve episodes

    `eval.py` can evaluate a PushT policy in two ways. They look alike (same env,
    same planner, same images) but they ask very different questions.
    This notebook puts them side by side, first on single examples, then on numbers
    over the whole dataset. No GPU or checkpoint needed.

    | | **25-step** (`config/eval/pusht.yaml`) | **Full solve** (`config/eval/pusht_full.yaml`) |
    |---|---|---|
    | Start state | a state in the middle of an expert demo | random: agent anywhere in 50–450, block anywhere in 100–400, any angle |
    | Goal | the expert's own state **25 steps later** (its image from the dataset) | the block on the **green T** (256, 256, 45°); agent drawn at (153, 356) |
    | Success rule | agent **and** block near the goal: ‖(agent, block) error‖ < 20 and angle < 20° | **block only** within 20 px and 20° of the green T |
    | Step budget | 50 (twice the 25 the expert needed) | 300 (expert demos take ~120) |
    | Episodes | 50 | 8 |
    | Code | `world.evaluate(dataset=...)` | `eval.py:evaluate_full_solve` + `BlockOnGoalSuccess` |

    In short: **25-step** tests "can you copy a short piece of what the expert did",
    **full solve** tests "can you actually do the task".
    """)
    return


@app.cell
def _(lance):
    # Settings copied from the two eval configs (goal offset, step budgets,
    # green-T pose, where the agent sits in the full-solve goal image, success
    # tolerances), plus the path to the expert dataset, which is opened here.
    # Change DATASET_PATH if the dataset lives somewhere else on your machine.
    DATASET_PATH = "/home/taot/data/ml_data/manual_download/stable_worldmodel/datasets/librakevin--lewm-pusht/pusht_expert_train.lance"
    WORLD_SIZE = 512  # env world is 512x512; images are 224x224, no flip
    OFFSET = 25  # eval.goal_offset_steps
    BUDGET_DS, BUDGET_FULL = 50, 300  # eval.eval_budget in the two configs
    GOAL_POSE = (256.0, 256.0, 0.7853981633974483)  # green T: x, y, angle (pi/4)
    GOAL_AGENT = (153.0, 356.0)  # pusht_full.yaml eval.goal_agent_pos
    POS_TOL, ANGLE_TOL = 20.0, 0.3490658503988659  # 20 px, pi/9 = 20 deg
    ds = lance.dataset(DATASET_PATH)
    return (
        ANGLE_TOL,
        BUDGET_DS,
        BUDGET_FULL,
        GOAL_AGENT,
        GOAL_POSE,
        OFFSET,
        POS_TOL,
        WORLD_SIZE,
        ds,
    )


@app.cell
def _(ds, np):
    # Load the state of every step of every expert demo into one big array.
    # Then work out, for each demo, which row it starts at and how many steps it has,
    # so later cells can pull out one demo or one 25-step window quickly.
    # state = [agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]
    # Rows are sorted by episode, then step. Loads ~2.3M rows, takes a few seconds.
    _tbl = ds.to_table(columns=["episode_idx", "state"])
    ep_of_row = _tbl.column("episode_idx").to_numpy()
    states = np.stack(_tbl.column("state").to_numpy(zero_copy_only=False)).astype(np.float64)
    ep_ids, ep_start_row, ep_len = np.unique(ep_of_row, return_index=True, return_counts=True)
    return ep_len, ep_start_row, states


@app.cell
def _(ANGLE_TOL, GOAL_POSE, POS_TOL, np):
    # Helper functions shared by the rest of the notebook:
    # - angle_diff: smallest gap between two angles (handles wrap-around at 360°).
    # - ds_rule: the 25-step success check (agent AND block close to the goal state).
    # - full_rule: the full-solve success check (only the block vs the green T).
    # Each rule returns (success?, position error, angle error).
    def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Smallest absolute difference between two angles (radians)."""
        d = np.abs(np.asarray(a) - np.asarray(b)) % (2 * np.pi)
        return np.minimum(d, 2 * np.pi - d)

    def ds_rule(state: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """25-step success (PushT.eval_state): one 4-d distance over agent AND block."""
        pos = np.linalg.norm(np.asarray(state)[..., :4] - np.asarray(goal)[..., :4], axis=-1)
        ang = angle_diff(np.asarray(state)[..., 4], np.asarray(goal)[..., 4])
        return (pos < POS_TOL) & (ang < ANGLE_TOL), pos, ang

    def full_rule(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full-solve success (BlockOnGoalSuccess): block vs green T, agent ignored."""
        pos = np.linalg.norm(np.asarray(state)[..., 2:4] - np.asarray(GOAL_POSE[:2]), axis=-1)
        ang = angle_diff(np.asarray(state)[..., 4], GOAL_POSE[2])
        return (pos < POS_TOL) & (ang < ANGLE_TOL), pos, ang

    return angle_diff, ds_rule, full_rule


@app.cell
def _(gym):
    # Create one PushT env that runs on the CPU. It is used to render images
    # and to draw random full-solve start states the same way eval.py does.
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224).unwrapped
    env.reset(seed=0)
    return (env,)


@app.cell
def _(mo):
    # Section 1 text: explains the 25-step example shown below and warns that
    # the green T in the images does not matter in this mode.
    mo.md(r"""
    ## 1. A 25-step episode

    Pick an expert demo and a start step. The start state and the goal are both
    real states of that demo, 25 steps apart. The dashed line is what the expert
    did in between; the faint line is the rest of the demo.

    Note the **green T is drawn in every image, but in this mode it means nothing**:
    the goal is wherever the gray block is 25 steps later, which usually is not
    on the green T.
    """)
    return


@app.cell
def _(ep_len, mo):
    # Input box to choose which expert demo (by index) to use for section 1.
    ep_input = mo.ui.number(start=0, stop=len(ep_len) - 1, value=0, step=1, label="Demo")
    ep_input
    return (ep_input,)


@app.cell
def _(OFFSET, ep_input, ep_len, mo):
    # Slider to choose the start step inside the chosen demo. It stops 25 steps
    # before the end so the goal (start + 25) always exists. Defaults to step 40
    # because the first steps of a demo often only move the agent, not the block.
    _stop = int(ep_len[ep_input.value]) - OFFSET - 1
    start_slider = mo.ui.slider(
        start=0,
        stop=_stop,
        value=min(40, _stop),  # early steps often only move the agent
        step=1,
        label="Start step",
        show_value=True,
        full_width=True,
    )
    start_slider
    return (start_slider,)


@app.cell
def _(Image, OFFSET, ds, ep_input, ep_start_row, io, start_slider, states):
    # Pick out the start state and the goal state (25 steps later) of the chosen
    # demo, plus all states in between (the expert's path). Also load the two
    # dataset images, which are exactly what the policy sees during eval.
    row0 = int(ep_start_row[ep_input.value]) + start_slider.value
    ds_start, ds_goal = states[row0], states[row0 + OFFSET]
    ds_window = states[row0 : row0 + OFFSET + 1]
    # the images the policy actually gets are the dataset pixels
    _pix = ds.take([row0, row0 + OFFSET], columns=["pixels"]).column("pixels").to_pylist()
    ds_start_img, ds_goal_img = (Image.open(io.BytesIO(p)) for p in _pix)
    return ds_goal, ds_goal_img, ds_start, ds_start_img, ds_window


@app.cell
def _(ep_input, ep_len, ep_start_row, states):
    # All states of the chosen demo, start to finish. Used to draw the faint
    # line showing the rest of the demo around the 25-step window.
    _a = int(ep_start_row[ep_input.value])
    ds_episode = states[_a : _a + int(ep_len[ep_input.value])]
    return (ds_episode,)


@app.cell
def _(WORLD_SIZE, np, plt):
    # Plot helper used by sections 1 and 2: shows the start image and goal image
    # side by side. Optional paths (in world pixels, 0-512) are scaled down to the
    # 224 px image and drawn on top of the start image.
    def show_pair(start_img, goal_img, start_title: str, goal_title: str, paths=()):
        """Start and goal images; paths = [(xy array in world px, style dict)] drawn on the start."""
        s = 224 / WORLD_SIZE
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, img, title in zip(axes, (start_img, goal_img), (start_title, goal_title)):
            ax.imshow(np.asarray(img))
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        for xy, style in paths:
            axes[0].plot(xy[:, 0] * s, xy[:, 1] * s, **style)
        if any("label" in style for _, style in paths):
            axes[0].legend(loc="upper right", fontsize=8, frameon=False, title="next 25 steps", title_fontsize=8)
        fig.tight_layout()
        return fig

    return (show_pair,)


@app.cell
def _(OFFSET, ds_episode, ds_goal_img, ds_start_img, ds_window, show_pair):
    # Draw the 25-step example: start image and goal image. On the start image,
    # dashed lines show where the expert moved the block (blue) and agent (orange)
    # over the next 25 steps; the faint gray line is the block over the whole demo.
    show_pair(
        ds_start_img,
        ds_goal_img,
        "start (dataset state)",
        f"goal = expert state {OFFSET} steps later",
        paths=[
            (ds_episode[:, 2:4], dict(color="#888888", lw=1, alpha=0.4)),
            (ds_window[:, 2:4], dict(color="#2a78d6", lw=2, ls="--", label="block")),
            (ds_window[:, 0:2], dict(color="#eb6834", lw=2, ls="--", label="agent")),
        ],
    )
    return


@app.cell
def _(ds_goal, ds_rule, ds_start, full_rule, mo, np):
    # Table for the 25-step example: how far the agent and block must move,
    # how much the block must turn, whether the start already counts as a success,
    # and whether the goal state happens to be on the green T.
    _ok_ds, _pos_ds, _ang_ds = ds_rule(ds_start, ds_goal)
    _ok_full, _pos_full, _ang_full = full_rule(ds_start)
    _ok_goal_full, _, _ = full_rule(ds_goal)
    mo.md(
        f"""
    How far the start is from the goal:

    | | Value |
    |---|---|
    | agent must move | {np.linalg.norm(ds_goal[:2] - ds_start[:2]):.0f} px |
    | block must move | {np.linalg.norm(ds_goal[2:4] - ds_start[2:4]):.0f} px |
    | block must turn | {np.degrees(_ang_ds):.0f}° |
    | 25-step error at start (agent + block) | {_pos_ds:.0f} px → {"already a success" if _ok_ds else "not yet a success"} |
    | is the **goal** state on the green T? | {"yes" if _ok_goal_full else "no"} (so reaching it would {"" if _ok_goal_full else "**not** "}count as a full solve) |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    mo.md(r"\"\"
    ## 2. A full-solve episode

    The start is drawn at random by the env (seed below). The goal image shows the
    block on the green T with the agent parked at (153, 356), where expert demos
    that finish on the T usually leave it. There is no expert path to follow.
    "\"\")
    """)
    return


@app.cell
def _(mo):
    # Input box to choose the random seed; each seed gives a different
    # random full-solve start.
    seed_input = mo.ui.number(start=0, value=42, step=1, label="Seed")
    seed_input
    return (seed_input,)


@app.cell
def _(GOAL_AGENT, GOAL_POSE, env, np, seed_input):
    # Reset the env with the chosen seed to get a random start state, just like
    # evaluate_full_solve does. The goal state is the block on the green T with the
    # agent at its parking spot; the env renders that as the goal image.
    full_goal = np.array([*GOAL_AGENT, *GOAL_POSE, 0.0, 0.0])
    _obs, _info = env.reset(seed=seed_input.value, options={"goal_state": full_goal})
    full_start = np.asarray(_obs["state"], dtype=np.float64)
    full_start_img, full_goal_img = env.render(), _info["goal"]
    return full_goal, full_goal_img, full_start, full_start_img


@app.cell
def _(full_goal_img, full_start_img, show_pair):
    # Draw the full-solve example: random start image and green-T goal image.
    show_pair(full_start_img, full_goal_img, "start (random)", "goal = block on the green T")
    return


@app.cell
def _(full_goal, full_rule, full_start, mo, np):
    # Table for the full-solve example: how far the block must move and turn to
    # reach the green T. The agent's distance is shown too, but success ignores it.
    _, _pos, _ang = full_rule(full_start)
    mo.md(
        f"""
    | | Value |
    |---|---|
    | block must move | {_pos:.0f} px |
    | block must turn | {np.degrees(_ang):.0f}° |
    | agent distance to its spot in the goal image | {np.linalg.norm(full_goal[:2] - full_start[:2]):.0f} px (not checked for success) |
    """
    )
    return


@app.cell
def _(mo):
    # Section 3 text: explains that both rules say "20 px and 20°" but the
    # 25-step rule also counts the agent's error, while full solve only checks the block.
    mo.md(r"""
    ## 3. The two success rules

    Both rules use "20 px and 20°", but they measure different things:

    - **25-step** (`PushT.eval_state`) takes one distance over the agent *and* the
      block together: $\sqrt{\lVert\Delta\text{agent}\rVert^2 + \lVert\Delta\text{block}\rVert^2} < 20$.
      So the agent must also end up where the expert's agent was. A perfect block
      with the agent 20 px off is a failure.
    - **Full solve** (`BlockOnGoalSuccess`) only looks at the block. The agent can
      be anywhere.

    Try some errors:
    """)
    return


@app.cell
def _(mo):
    # Three sliders to try made-up errors: how far the agent is off, how far
    # the block is off, and how much the block's angle is off.
    agent_err = mo.ui.slider(0, 60, value=15, label="agent error (px)", show_value=True)
    block_err = mo.ui.slider(0, 60, value=15, label="block error (px)", show_value=True)
    angle_err = mo.ui.slider(0, 45, value=5, label="block angle error (°)", show_value=True)
    mo.vstack([agent_err, block_err, angle_err])
    return agent_err, angle_err, block_err


@app.cell
def _(ANGLE_TOL, POS_TOL, agent_err, angle_err, block_err, mo, np):
    # Apply both success rules to the slider values and show the result.
    # The 25-step rule combines agent and block errors into one distance
    # (sqrt(agent² + block²)), so a big agent error alone can make it fail.
    _ds_pos = float(np.hypot(agent_err.value, block_err.value))
    _ang_ok = np.radians(angle_err.value) < ANGLE_TOL
    _ds_ok = _ds_pos < POS_TOL and _ang_ok
    _full_ok = block_err.value < POS_TOL and _ang_ok

    def _mark(ok: bool) -> str:
        return "✅ success" if ok else "❌ fail"

    mo.md(
        f"""
    | Rule | Distance it checks | Angle | Result |
    |---|---|---|---|
    | 25-step | √(agent² + block²) = {_ds_pos:.1f} px | {angle_err.value}° | {_mark(_ds_ok)} |
    | Full solve | block = {block_err.value} px | {angle_err.value}° | {_mark(_full_ok)} |
    """
    )
    return


@app.cell
def _(mo):
    # Text: how an episode ends in each mode (any-step success vs stop on
    # success, 50 vs 300 steps), then the intro to section 4's dataset-wide numbers.
    mo.md(r"""
    Two more differences in how an episode ends:

    - **25-step** counts a success if the rule holds at *any* of the 50 steps
      (`episode_successes |= terminated`). Only 50 episodes run, each capped at 50 steps.
    - **Full solve** stops an env as soon as it succeeds (`reset_mode="wait"`), and
      gives up after 300 steps.

    ## 4. Numbers over the whole dataset

    Below: 25-step windows sampled from every demo, the first state of every demo,
    and 2,000 full-solve random starts from the env.
    """)
    return


@app.cell
def _(
    GOAL_POSE,
    OFFSET,
    angle_diff,
    ds_rule,
    env,
    ep_len,
    ep_start_row,
    full_goal,
    np,
    states,
):
    # Dataset-wide numbers, three groups:
    # 1. 200k random 25-step windows: how far the block moves/turns between start and goal.
    # 2. The first state of each demo: how far the block is from the green T.
    # 3. 2,000 random full-solve starts from the env: how far the block is from the green T.
    # Also counts how many distinct start layouts the demos use.
    _rng = np.random.default_rng(0)
    # 25-step windows: every (start, start + 25) pair inside one demo, subsampled
    _valid = np.concatenate(
        [np.arange(a, a + n - OFFSET) for a, n in zip(ep_start_row, ep_len) if n > OFFSET]
    )
    _rows = _rng.choice(_valid, size=200_000, replace=False)
    win_start, win_goal = states[_rows], states[_rows + OFFSET]
    win_block_move = np.linalg.norm(win_goal[:, 2:4] - win_start[:, 2:4], axis=1)
    win_turn = np.degrees(angle_diff(win_goal[:, 4], win_start[:, 4]))
    win_already_ok = ds_rule(win_start, win_goal)[0].mean()

    # first state of each demo, measured against the green T
    _first = states[ep_start_row]
    n_start_layouts = len(np.unique(np.round(_first[:, :5], 2), axis=0))
    first_block_move = np.linalg.norm(_first[:, 2:4] - np.asarray(GOAL_POSE[:2]), axis=1)
    first_turn = np.degrees(angle_diff(_first[:, 4], GOAL_POSE[2]))

    # full-solve random starts, drawn exactly as evaluate_full_solve does
    _starts = []
    for _s in range(2000):
        _obs, _ = env.reset(seed=_s, options={"goal_state": full_goal})
        _starts.append(_obs["state"])
    _starts = np.asarray(_starts, dtype=np.float64)
    rand_block_move = np.linalg.norm(_starts[:, 2:4] - np.asarray(GOAL_POSE[:2]), axis=1)
    rand_turn = np.degrees(angle_diff(_starts[:, 4], GOAL_POSE[2]))
    return (
        first_block_move,
        first_turn,
        n_start_layouts,
        rand_block_move,
        rand_turn,
        win_already_ok,
        win_block_move,
        win_turn,
    )


@app.cell
def _(
    first_block_move,
    first_turn,
    np,
    plt,
    rand_block_move,
    rand_turn,
    win_block_move,
    win_turn,
):
    # Two histograms comparing the three groups above: how far the block must
    # move (left) and turn (right). Shows 25-step goals are much closer than a
    # full solve.
    _series = [
        ("25-step goal", win_block_move, win_turn, "#2a78d6"),
        ("expert demo start → green T", first_block_move, first_turn, "#1baf7a"),
        ("full-solve random start → green T", rand_block_move, rand_turn, "#eb6834"),
    ]
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for _name, _move, _turn, _c in _series:
        _axes[0].hist(_move, bins=np.linspace(0, 400, 61), density=True, histtype="step", lw=2, color=_c, label=_name)
        _axes[1].hist(_turn, bins=np.linspace(0, 180, 46), density=True, histtype="step", lw=2, color=_c, label=_name)
    _axes[0].set_xlabel("how far the block must move (px)")
    _axes[1].set_xlabel("how far the block must turn (°)")
    for _ax in _axes:
        _ax.set_yticks([])
        _ax.spines[["top", "right", "left"]].set_visible(False)
    _axes[0].legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    ep_len,
    first_block_move,
    first_turn,
    mo,
    n_start_layouts,
    np,
    rand_block_move,
    rand_turn,
    win_already_ok,
    win_block_move,
    win_turn,
):
    # Table of the median block move/turn for each group, plus notes on what
    # it means: 25-step goals are small nudges, and full-solve starts are new
    # layouts the model never saw in training.
    def _med(x: np.ndarray) -> str:
        return f"{np.median(x):.0f}"

    mo.md(
        f"""
    | Median | block move (px) | block turn (°) |
    |---|---|---|
    | 25-step goal | {_med(win_block_move)} | {_med(win_turn)} |
    | expert demo start → green T | {_med(first_block_move)} | {_med(first_turn)} |
    | full-solve random start → green T | {_med(rand_block_move)} | {_med(rand_turn)} |

    A 25-step goal is a small nudge: in many windows the block barely moves
    (the expert is often just walking the agent to the next contact point).
    A full solve needs the whole trip, like the start of an expert demo.
    Only {win_already_ok:.1%} of 25-step windows are a success before doing anything.

    The green line is jagged for a reason: the {len(ep_len):,} demos start from only
    **{n_start_layouts} different layouts** (many demos per start). 25-step starts
    are always states the model saw in training. Full-solve starts are fresh random
    layouts, so full solve also tests how well the model generalizes.
    """
    )
    return


@app.cell
def _(BUDGET_DS, BUDGET_FULL, OFFSET, ep_len, np, plt):
    # Histogram of how many steps each expert demo takes, with dashed lines at
    # the 25-step goal offset, the 25-step budget (50) and the full-solve budget (300).
    _fig, _ax = plt.subplots(figsize=(10, 3.2))
    _ax.hist(ep_len, bins=np.arange(0, 330, 5), color="#1baf7a", alpha=0.8)
    for _x, _label, _y in [
        (OFFSET, "25-step goal", 0.95),
        (BUDGET_DS, "25-step budget", 0.8),
        (BUDGET_FULL, "full-solve budget", 0.95),
    ]:
        _ax.axvline(_x, color="#52514e", ls="--", lw=1)
        _ax.text(_x + 3, _ax.get_ylim()[1] * _y, _label, fontsize=8, color="#52514e")
    _ax.set_xlabel("expert demo length (steps)")
    _ax.set_yticks([])
    _ax.spines[["top", "right", "left"]].set_visible(False)
    _ax.set_title("How long the expert takes for the whole task", fontsize=10)
    _fig
    return


@app.cell
def _(ep_len, ep_start_row, full_rule, mo, np, states):
    # Check how many expert demos actually end with the block on the green T,
    # compare demo length to the 300-step budget, and list the final takeaways.
    _last = states[ep_start_row + ep_len - 1]
    _on_t = full_rule(_last)[0].mean()
    mo.md(
        f"""
    Expert demos take a median of **{np.median(ep_len):.0f} steps** (90% within
    {np.percentile(ep_len, 90):.0f}), so the full-solve budget of 300 is about
    2.5× what the expert needs, the same slack as 50 for a 25-step goal.

    Even the expert does not always finish on the T: only **{_on_t:.0%}** of demos
    end with the block within 20 px / 20° of the green T. So "full solve" is a
    strict test, while "25-step" can be passed by imitating short pieces of a
    demo that never solves the task.

    ## Takeaways

    - **25-step** checks short-range, goal-image imitation. The goal is close,
      the path is known to be doable in 25 steps, but the agent's position must match too.
    - **Full solve** checks the real task: long horizon (~120 steps, many replans
      of 25 actions each), random starts, block-only success, fixed green-T goal.
    - A model can score well on 25-step and still fail full solve: small errors
      that don't matter over 25 steps pile up over 120+, and the planner must
      pick good intermediate moves toward a goal it can't reach in one plan.
    - In full solve the goal *image* includes the agent at (153, 356), so the
      planner also tries to move the agent there even though success ignores it.
    """
    )
    return


if __name__ == "__main__":
    app.run()
