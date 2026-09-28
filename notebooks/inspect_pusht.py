import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import io

    import lance
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Polygon
    from PIL import Image

    return Image, Polygon, io, lance, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Inspecting PushT rows

    Each row of `pusht_expert_train.lance` is one timestep of one expert demonstration.
    This notebook inspects the `action`, `proprio` and `state` fields of a row, checked
    against the rendered `pixels`. See `docs/pusht_dataset.md`.
    """)
    return


@app.cell
def _(lance):
    DATASET_PATH = "/home/taot/data/ml_data/manual_download/stable_worldmodel/datasets/librakevin--lewm-pusht/pusht_expert_train.lance"
    ACTION_SCALE = 100  # env.action_scale
    WORLD_SIZE = 512  # env.window_size; pixels are rendered at 224x224
    DT = 0.1  # one env step = 1 / control_hz (10 Hz)
    ds = lance.dataset(DATASET_PATH)
    return ACTION_SCALE, DT, WORLD_SIZE, ds


@app.cell
def _(mo):
    row_input = mo.ui.number(start=0, value=0, step=1, label="Row")
    row_input
    return (row_input,)


@app.cell
def _(Image, ds, io, np, row_input):
    row_idx = row_input.value
    rows = ds.take([row_idx, row_idx + 1]).to_pylist()
    cur, nxt = rows
    # The next row only shows the result of this action if it's in the same episode.
    has_next = nxt["episode_idx"] == cur["episode_idx"]

    action = np.array(cur["action"])
    proprio = np.array(cur["proprio"])
    state = np.array(cur["state"])
    pos = proprio[:2]
    next_pos = np.array(nxt["proprio"][:2])
    frame = Image.open(io.BytesIO(cur["pixels"]))
    next_frame = Image.open(io.BytesIO(nxt["pixels"]))
    return (
        action,
        cur,
        frame,
        has_next,
        next_frame,
        next_pos,
        pos,
        proprio,
        state,
    )


@app.cell
def _(cur, ds, np):
    # Non-pixel columns of the whole episode, for trajectory / time-series context.
    _tbl = ds.to_table(
        filter=f"episode_idx = {cur['episode_idx']}",
        columns=["step_idx", "proprio", "state"],
    )
    _order = np.argsort(_tbl.column("step_idx").to_numpy())
    ep_steps = _tbl.column("step_idx").to_numpy()[_order]
    ep_proprio = np.stack(_tbl.column("proprio").to_numpy(zero_copy_only=False))[_order]
    ep_state = np.stack(_tbl.column("state").to_numpy(zero_copy_only=False))[_order]
    return ep_proprio, ep_state, ep_steps


@app.cell
def _():
    def fmt(v):
        return f"({v[0]:.2f}, {v[1]:.2f})"

    return (fmt,)


@app.cell
def _(mo):
    mo.md(r"""
    ## `action`

    An action is a **relative target offset** in units of 100 px:
    `target = agent_pos + action * 100`. A PD controller then pulls the agent
    toward `target` for one env step (10 physics substeps of 0.01 s).
    """)
    return


@app.cell
def _(ACTION_SCALE, action, cur, fmt, has_next, mo, next_pos, np, pos):
    target = pos + action * ACTION_SCALE
    _offset = target - pos
    _moved = next_pos - pos

    _lines = [
        "| | Value |",
        "|---|---|",
        f"| episode / step | {cur['episode_idx']} / {cur['step_idx']} |",
        f"| `action` (raw) | {fmt(action)} |",
        f"| agent position | {fmt(pos)} |",
        f"| commanded offset = action × {ACTION_SCALE} | {fmt(_offset)} px |",
        f"| target = pos + offset | {fmt(target)} |",
    ]
    if has_next:
        _covered = float(_moved @ _offset / (_offset @ _offset)) if _offset @ _offset > 0 else float("nan")
        _cos = float(_moved @ _offset / (np.linalg.norm(_moved) * np.linalg.norm(_offset) + 1e-9))
        _lines += [
            f"| agent position at next step | {fmt(next_pos)} |",
            f"| actual move | {fmt(_moved)} px |",
            f"| fraction of offset covered | {_covered:.1%} |",
            f"| direction agreement (cosine) | {_cos:.3f} |",
        ]
    else:
        _lines.append("| next step | (last step of episode) |")
    mo.md("\n".join(_lines))
    return (target,)


@app.cell
def _(WORLD_SIZE, frame, has_next, next_frame, next_pos, plt, pos, target):
    _s = frame.size[0] / WORLD_SIZE  # world px -> image px

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))
    for _ax in _axes[:2]:
        _ax.imshow(frame)
        _ax.annotate(
            "", xy=target * _s, xytext=pos * _s,
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )
        _ax.plot(*(target * _s), "rx", ms=10, mew=2, label="target")
        if has_next:
            _ax.plot(*(next_pos * _s), "o", mfc="none", mec="orange", ms=12, mew=2, label="next position")
        _ax.axis("off")
    _axes[0].legend(loc="lower right")
    _axes[0].set_title("This step: action as offset (red)")

    # Zoom on the agent: typical offsets are only ~20 world px (~9 image px).
    _center = pos * _s
    _half = max(20.0, 1.5 * abs(target * _s - _center).max())
    _axes[1].set_xlim(_center[0] - _half, _center[0] + _half)
    _axes[1].set_ylim(_center[1] + _half, _center[1] - _half)
    _axes[1].set_title("Zoomed on agent")

    _axes[2].imshow(next_frame)
    _axes[2].set_title("Next row" + ("" if has_next else " (new episode)"))
    _axes[2].axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## `proprio`

    The agent's own body state, `[agent_x, agent_y, agent_vx, agent_vy]`, in world
    pixels of a 512×512 arena (x right, y **down**, same as the image). It is exactly
    `state[[0, 1, 5, 6]]`. Velocity is the agent's instantaneous velocity at the end of
    the step (px/s). It tracks, but isn't identical to, the average velocity over the
    step (`Δpos / 0.1 s`). Every episode starts at rest, so it's zero at step 0.
    """)
    return


@app.cell
def _(mo, np, proprio, state):
    PROPRIO_NAMES = ["agent_x", "agent_y", "agent_vx", "agent_vy"]
    _units = ["px", "px", "px/s", "px/s"]
    _state_idx = [0, 1, 5, 6]
    _lines = ["| index | name | value | unit | `state` index | matches |", "|---|---|---|---|---|---|"]
    for _i, (_name, _unit, _j) in enumerate(zip(PROPRIO_NAMES, _units, _state_idx)):
        _ok = "✓" if np.isclose(proprio[_i], state[_j]) else "✗"
        _lines.append(f"| {_i} | `{_name}` | {proprio[_i]:.2f} | {_unit} | {_j} | {_ok} |")
    _lines.append(f"\nSpeed: **{np.linalg.norm(proprio[2:]):.1f} px/s**")
    mo.md("\n".join(_lines))
    return


@app.cell
def _(DT, WORLD_SIZE, cur, ep_proprio, ep_steps, frame, np, plt, proprio):
    _s = frame.size[0] / WORLD_SIZE
    _step = cur["step_idx"]

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Velocity drawn as the displacement it would cause over one step (v * DT).
    _ax = _axes[0]
    _ax.imshow(frame)
    _p = proprio[:2] * _s
    _tip = (proprio[:2] + proprio[2:] * DT) * _s
    _ax.plot(*_p, "o", mfc="none", mec="magenta", ms=12, mew=2, label="(agent_x, agent_y)")
    _ax.annotate("", xy=_tip, xytext=_p, arrowprops=dict(arrowstyle="->", color="magenta", lw=2))
    _half = max(20.0, 1.5 * abs(_tip - _p).max())
    _ax.set_xlim(_p[0] - _half, _p[0] + _half)
    _ax.set_ylim(_p[1] + _half, _p[1] - _half)
    _ax.set_title("Agent position and velocity × 0.1 s (zoomed)")
    _ax.legend(loc="lower right")
    _ax.axis("off")

    # Recorded velocity vs. average velocity from consecutive positions.
    _ax = _axes[1]
    _avg_v = np.diff(ep_proprio[:, :2], axis=0) / DT
    for _k, (_name, _color) in enumerate([("vx", "C0"), ("vy", "C1")]):
        _ax.plot(ep_steps, ep_proprio[:, 2 + _k], color=_color, label=f"agent_{_name} (recorded)")
        _ax.plot(ep_steps[1:], _avg_v[:, _k], color=_color, ls="--", alpha=0.6, label=f"Δpos/Δt ({_name})")
    _ax.axvline(_step, color="k", lw=1)
    _ax.set_xlabel("step_idx")
    _ax.set_ylabel("px/s")
    _ax.set_title(f"Agent velocity over episode {cur['episode_idx']}")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## `state`

    The full simulator state,
    `[agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]`.
    Beyond `proprio`, it adds the T block's pose. This is the information the world model
    must infer from pixels alone. `(block_x, block_y)` is the origin of the T's body frame
    (the top edge's midpoint, not its center of mass), and `block_angle` is its
    rotation in radians, wrapped to [0, 2π). The overlay below redraws the T from these
    three numbers using the env's T geometry; it should sit exactly on the grey T.
    """)
    return


@app.cell
def _(mo, np, state):
    STATE_NAMES = ["agent_x", "agent_y", "block_x", "block_y", "block_angle", "agent_vx", "agent_vy"]
    _units = ["px", "px", "px", "px", "rad", "px/s", "px/s"]
    _lines = ["| index | name | value | unit |", "|---|---|---|---|"]
    for _i, (_name, _unit) in enumerate(zip(STATE_NAMES, _units)):
        _val = f"{state[_i]:.3f}"
        if _name == "block_angle":
            _val += f" ({np.degrees(state[_i]):.1f}°)"
        _lines.append(f"| {_i} | `{_name}` | {_val} | {_unit} |")
    mo.md("\n".join(_lines))
    return


@app.cell
def _(np):
    def tee_polygons(x, y, angle, scale=30, length=4):
        """World-space vertices of the two T rectangles (from PushTEnv.add_tee)."""
        _bar = [(-length * scale / 2, scale), (length * scale / 2, scale),
                (length * scale / 2, 0), (-length * scale / 2, 0)]
        _stem = [(-scale / 2, scale), (-scale / 2, length * scale),
                 (scale / 2, length * scale), (scale / 2, scale)]
        _rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return [np.array(_v) @ _rot.T + [x, y] for _v in (_bar, _stem)]

    return (tee_polygons,)


@app.cell
def _(Polygon, WORLD_SIZE, cur, ep_state, ep_steps, frame, np, plt, state, tee_polygons):
    _s = frame.size[0] / WORLD_SIZE
    _step = cur["step_idx"]

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 5))

    # Overlay: T redrawn from (block_x, block_y, block_angle), agent from (agent_x, agent_y).
    _ax = _axes[0]
    _ax.imshow(frame)
    for _poly in tee_polygons(*state[2:5]):
        _ax.add_patch(Polygon(_poly * _s, fill=False, ec="red", lw=1.5, ls="--"))
    _ax.plot(*(state[2:4] * _s), "r+", ms=12, mew=2, label="(block_x, block_y)")
    _ax.plot(*(state[:2] * _s), "o", mfc="none", mec="magenta", ms=12, mew=2, label="(agent_x, agent_y)")
    _ax.legend(loc="lower right", fontsize=8)
    _ax.set_title("state redrawn over pixels")
    _ax.axis("off")

    # Top-down trajectories of agent and block over the episode.
    _ax = _axes[1]
    _ax.plot(ep_state[:, 0], ep_state[:, 1], color="royalblue", lw=1, label="agent")
    _ax.plot(ep_state[:, 2], ep_state[:, 3], color="slategray", lw=2, label="block origin")
    _ax.plot(*ep_state[0, :2], "o", color="royalblue", ms=5)
    _ax.plot(*state[:2], "o", mfc="none", mec="magenta", ms=10, mew=2)
    _ax.plot(*state[2:4], "r+", ms=12, mew=2)
    for _poly in tee_polygons(*ep_state[-1, 2:5]):
        _ax.add_patch(Polygon(_poly, fill=False, ec="slategray", lw=1, ls=":"))
    _ax.set_xlim(0, WORLD_SIZE)
    _ax.set_ylim(WORLD_SIZE, 0)  # y down, like the image
    _ax.set_aspect("equal")
    _ax.set_title(f"Episode {cur['episode_idx']} trajectories (dotted T = final pose)")
    _ax.legend(fontsize=8)

    # Block pose over time.
    _ax = _axes[2]
    _ax.plot(ep_steps, ep_state[:, 2], label="block_x")
    _ax.plot(ep_steps, ep_state[:, 3], label="block_y")
    _ax.set_xlabel("step_idx")
    _ax.set_ylabel("px")
    _ax2 = _ax.twinx()
    _ax2.plot(ep_steps, np.degrees(ep_state[:, 4]), color="C2", label="block_angle")
    _ax2.set_ylabel("block_angle (°)")
    _ax.axvline(_step, color="k", lw=1)
    _h1, _l1 = _ax.get_legend_handles_labels()
    _h2, _l2 = _ax2.get_legend_handles_labels()
    _ax.legend(_h1 + _h2, _l1 + _l2, fontsize=8)
    _ax.set_title("Block pose over episode")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
