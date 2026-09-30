import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from typing import Any

    import gymnasium as gym
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import stable_worldmodel as swm  # importing swm registers the "swm/..." env ids

    return Any, gym, mo, np, plt, swm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 · 环境与 variation space

    swm 的环境就是普通的 **gymnasium 环境**：`gym.make("swm/...")`，然后
    `reset()` / `step(action)`。它在 gym 之上只加了两条约定：

    1. **重要信息放进 `info` 字典**（位置、目标、目标图像……），而不仅仅是 `obs`；
    2. 每个环境有一个 **`variation_space`**：描述环境里"可以变的东西"（颜色、形状、
       大小、起点……），`reset` 时按 seed 从里面抽样。这让同一个环境可以方便地做
       泛化测试（比如换个颜色，模型还行不行）。

    本篇用 PushT 做例子，但这些接口对所有 `swm/` 环境都一样。
    """)
    return


@app.cell
def _(gym):
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    return env, info, obs


@app.cell
def _(env, info, mo, obs):
    def _describe(v: object) -> str:
        shape = getattr(v, "shape", None)
        return f"`{type(v).__name__}` shape={shape}" if shape is not None else f"`{v!r}`"

    _rows = ["| 键 | 内容 |", "|---|---|"]
    _rows += [f"| `obs['{k}']` | {_describe(v)} |" for k, v in obs.items()]
    _rows += [f"| `info['{k}']` | {_describe(v)} |" for k, v in info.items()]

    def _one_line(space: object) -> str:
        # Box reprs of array bounds contain newlines, which break inline code.
        return " ".join(str(space).split())

    mo.md(
        f"""
    ## 1. reset / step 返回什么

    - `env.action_space` = `{_one_line(env.action_space)}`
    - `env.observation_space` = `{_one_line(env.observation_space)}`

    `reset(seed=0)` 返回的 `obs` 和 `info`：

    {chr(10).join(_rows)}

    `obs` 只有低维状态；**图像不在 obs 里**，要用 `env.render()` 拿（下一篇的
    `World` 会自动帮你渲染并塞进 `info['pixels']`）。`info['goal']` 是目标状态渲染出的
    图像，`info['goal_state']` 是目标状态本身。
    """
    )
    return


@app.cell
def _(env, np, plt):
    # One step: actions are in [-1, 1]^2. Push the agent a bit to the right.
    _frame0 = env.render()
    _obs1, _reward, _terminated, _truncated, _info1 = env.step(np.array([0.5, 0.0], dtype=np.float32))
    _frame1 = env.render()

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4))
    _axes[0].imshow(_frame0)
    _axes[0].set_title("before step")
    _axes[1].imshow(_frame1)
    _axes[1].set_title("after step([0.5, 0])")
    _axes[2].imshow(_info1["goal"])
    _axes[2].set_title("info['goal']")
    for _ax in _axes:
        _ax.axis("off")
    _fig.suptitle(f"reward={_reward:.1f}  terminated={_terminated}  truncated={_truncated}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `step` 返回标准的五元组 `(obs, reward, terminated, truncated, info)`。
    PushT 的 `reward` 是"当前状态到目标状态的距离"取负，`terminated=True` 表示成功
    （06 篇细讲）。`truncated` 由 `gym.make(..., max_episode_steps=...)` 控制。

    ## 2. variation space：环境里"可以变的东西"

    `env.unwrapped.variation_space` 是一个嵌套的 `swm.spaces.Dict`。它的叶子是
    `swm.spaces.Box` / `RGBBox` / `Discrete`，和 gym 的 space 一样能 `sample()`，
    但多了两个属性：

    - `init_value`：默认值；
    - `value`：**当前**值（环境真正在用的值）。

    下表是 PushT 的全部可变因素和 `reset(seed=0)` 之后的当前值：
    """)
    return


@app.cell
def _(Any, env, mo, np, swm):
    vspace = env.unwrapped.variation_space

    # What each PushT variation controls (read from the PushT env source).
    descriptions: dict[str, str] = {
        "background.color": "背景颜色（RGB）",
        "goal.angle": "画面上绿色目标轮廓的角度（弧度）。只影响画出来的样子，不影响成功判定",
        "goal.color": "目标轮廓的颜色",
        "goal.position": "目标轮廓的中心位置 (x, y)，同样只影响画面",
        "goal.scale": "目标轮廓的大小",
        "block.angle": "方块的起始角度（弧度）；也用来抽样目标状态里的角度",
        "block.color": "方块颜色",
        "block.scale": "方块大小",
        "block.shape": "方块形状，是 `env.unwrapped.shapes` 的下标（1–7，默认 2 = `T`）",
        "block.start_position": "方块的起始位置 (x, y)；也用来抽样目标状态里方块的位置",
        "agent.angle": "智能体的角度（圆点时看不出区别）",
        "agent.color": "智能体颜色",
        "agent.scale": "智能体大小",
        "agent.shape": "智能体形状，`shapes` 的下标（0–7，默认 0 = 圆点 `o`）",
        "agent.start_position": "智能体的起始位置 (x, y)；也用来抽样目标状态里智能体的位置",
        "agent.velocity": "智能体的初始速度 (vx, vy)",
        "rendering.render_goal": "画面上是否画出目标轮廓（1 画，0 不画）",
    }

    def leaf_rows(space: Any) -> list[str]:
        rows = []
        for path in space.sampling_order:
            leaf = swm.utils.get_in(space, path.split("."))
            if isinstance(leaf, swm.spaces.Dict):
                continue
            val = np.round(np.asarray(leaf.value, dtype=float), 3).tolist()
            init = np.round(np.asarray(leaf.init_value, dtype=float), 3).tolist()
            desc = descriptions.get(path, "")
            rows.append(f"| `{path}` | `{type(leaf).__name__}` | {desc} | {init} | {val} |")
        return rows

    mo.md(
        "\n".join(
            ["| 路径 | 类型 | 说明 | init_value | 当前 value |", "|---|---|---|---|---|", *leaf_rows(vspace)]
        )
    )
    return (vspace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### reset 时哪些东西会被重新抽样？

    `reset(seed, options)` 内部调用 `swm.spaces.reset_variation_space`：

    1. 所有变量先恢复成 `init_value`；
    2. 再把 `options["variation"]` 里列出的变量**重新抽样**（用 `seed` 决定随机数）。
       没给 `options["variation"]` 时，用环境自己的默认列表——PushT 是
       `agent.start_position`、`block.start_position`、`block.angle`，所以默认只有起点在变；
       传 `["all"]` 表示全部重抽；
    3. 最后 `options["variation_values"]` 里给定的值**直接覆盖**（不抽样）。

    注意第 2 步：一旦你自己传了 `options["variation"]`，默认列表就不生效了——比如只写
    `["block.color"]`，起点就会停在 `init_value`。

    下面自己试：选要重抽的变量，拖 seed，看 8 个环境长什么样（第 i 个用 seed + i）。
    """)
    return


@app.cell
def _(mo, vspace):
    _leaves = [p for p in vspace.sampling_order if p.count(".") == 1]
    variation_select = mo.ui.multiselect(
        options=["all", *_leaves],
        value=["agent.start_position", "block.start_position", "block.angle"],
        label="options['variation']",
    )
    seed_slider = mo.ui.slider(0, 1000, value=0, label="seed", show_value=True)
    mo.vstack([variation_select, seed_slider])
    return seed_slider, variation_select


@app.cell
def _(env, plt, seed_slider, variation_select):
    _fig, _axes = plt.subplots(2, 4, figsize=(12, 6))
    for _i, _ax in enumerate(_axes.flat):
        env.reset(seed=seed_slider.value + _i, options={"variation": list(variation_select.value)})
        _ax.imshow(env.render())
        _ax.set_title(f"seed {seed_slider.value + _i}")
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 直接指定值：`variation_values`

    不想随机、想要一个确定的设置，就用 `variation_values`，键是点分路径。比如把方块
    换成别的形状（`block.shape` 是 `self.shapes` 列表的下标：
    `['o', 'L', 'T', 'Z', 'square', 'I', 'small_tee', '+']`，默认 2 = `T`），并把背景染色：
    """)
    return


@app.cell
def _(env, np, plt):
    _shapes = env.unwrapped.shapes
    _fig, _axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for _ax, _shape in zip(_axes, [1, 2, 3, 5]):
        env.reset(
            seed=0,
            options={
                "variation_values": {
                    "block.shape": _shape,
                    "background.color": np.array([235, 225, 200], dtype=np.uint8),
                }
            },
        )
        _ax.imshow(env.render())
        _ax.set_title(f"block.shape={_shape} ('{_shapes[_shape]}')")
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 环境自己的 reset 选项

    除了 variation，各个环境还可以支持自己的 `options`。PushT 支持：

    - `options["state"]`：直接指定起始状态（7 维，06 篇讲含义）；
    - `options["goal_state"]`：直接指定目标状态。

    这两个是 `eval.py` 做"从数据集的某一帧出发、去到 25 步之后那一帧"评估的基础
    （实际上 eval 用的是 `_set_state` / `_set_goal_state` 这两个方法，效果一样）。
    """)
    return


@app.cell
def _(env, np, plt):
    _start = np.array([100.0, 400.0, 256.0, 300.0, 0.0, 0.0, 0.0])
    _goal = np.array([400.0, 100.0, 256.0, 256.0, np.pi / 2, 0.0, 0.0])
    _obs, _info = env.reset(seed=0, options={"state": _start, "goal_state": _goal})
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 4))
    _axes[0].imshow(env.render())
    _axes[0].set_title(f"start: state={_obs['state'].round(1).tolist()[:5]}")
    _axes[1].imshow(_info["goal"])
    _axes[1].set_title("goal image (rendered from goal_state)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 同样的接口，别的环境

    换一个环境，代码不用改。下面是 `swm/TwoRoom-v1`（两个房间，点要穿过门走到目标）
    和它的 variation space 顶层键。
    """)
    return


@app.cell
def _(gym, mo, plt):
    try:
        _env2 = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
        _obs2, _info2 = _env2.reset(seed=0)
        _fig, _axes = plt.subplots(1, 2, figsize=(8, 4))
        _axes[0].imshow(_env2.render())
        _axes[0].set_title("TwoRoom: current")
        if "goal" in _info2:
            _axes[1].imshow(_info2["goal"])
            _axes[1].set_title("TwoRoom: goal")
        for _ax in _axes:
            _ax.axis("off")
        _fig.tight_layout()
        _keys = list(_env2.unwrapped.variation_space.spaces.keys())
        _out = mo.vstack(
            [
                mo.md(
                    f"obs 键：`{list(_obs2.keys()) if isinstance(_obs2, dict) else type(_obs2).__name__}`，"
                    f"variation 顶层键：`{_keys}`"
                ),
                _fig,
            ]
        )
    except Exception as _e:  # optional deps may be missing on some machines
        _out = mo.md(f"TwoRoom 在这台机器上不能创建：`{_e!r}`")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 对照：`gym.make` vs `swm.World`

    `swm.World` 内部也是用 `gym.make` 建环境，只是每个环境都会套上 `MegaWrapper`
    （负责渲染、缩放图像，并把所有数据挪进 `info`），再把 N 个环境打包成一批。
    下面用 `World(num_envs=2, image_shape=(64, 64))` 做一次 `reset(seed=0)`，
    把它的 `info` 和上面 `gym.make` 得到的 `obs` / `info` 并排放在一起看。
    """)
    return


@app.cell
def _(env, info, mo, obs, swm):
    def _shape(v: object) -> str:
        shape = getattr(v, "shape", None)
        return f"`{tuple(shape)}`" if shape is not None else f"`{type(v).__name__}`"

    # What gym.make gives for each key (pixels only via env.render()).
    _gym_side: dict[str, str] = {k: f"obs: {_shape(v)}" for k, v in obs.items()}
    _gym_side |= {k: f"info: {_shape(v)}" for k, v in info.items() if k not in _gym_side}
    _gym_side["pixels"] = f"没有，要调用 `env.render()` → {_shape(env.render())}"

    _world = swm.World("swm/PushT-v1", num_envs=2, image_shape=(64, 64))
    _world.reset(seed=0)
    _world_infos = _world.infos
    _world.close()

    _keys = list(dict.fromkeys([*_world_infos.keys(), *_gym_side.keys()]))
    _rows = ["| 键 | `gym.make` | `World`（`world.infos`） |", "|---|---|---|"]
    _rows += [
        f"| `{k}` | {_gym_side.get(k, '—')} | {_shape(_world_infos[k]) if k in _world_infos else '—'} |"
        for k in _keys
    ]
    mo.md(
        "\n".join(_rows)
        + "\n\n- `World` 这一列的数组形状都是 `(num_envs, 1, ...)`，图像已经缩放到 64×64；"
        "\n- 原来分开放的 `obs` 和 `info` 都合并进了 `world.infos`（`reset` / `step` 返回的 obs 是 `None`），"
        "另外多了 `reward`、`terminated`、`action`、`step_idx` 等键，方便记录数据；"
        "\n- 第 i 个环境用的 seed 是 `seed + i`。"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结

    - swm 环境 = gym 环境 + 丰富的 `info` + `variation_space`；
    - `reset(seed, options={"variation": [...], "variation_values": {...}})` 控制环境外观/起点；
    - 环境还可以有自己的 reset 选项（PushT：`state`、`goal_state`）；
    - `swm.World` = 多个 `gym.make` 环境 + `MegaWrapper`（自动渲染和缩放，数据都放进 `info`）+ 批处理。

    下一篇：用 `World` 同时跑很多个环境，并接上 policy。
    """)
    return


if __name__ == "__main__":
    app.run()
