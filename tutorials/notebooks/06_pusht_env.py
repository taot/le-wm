import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import gymnasium as gym
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import stable_worldmodel as swm  # noqa: F401  (registers swm/PushT-v1)

    return gym, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 06 · PushT：物理、状态、动作、成功条件

    从这篇开始专门讲 PushT（`swm/envs/pusht/env.py`）。任务：一个蓝色圆形的 **agent**
    （可以理解成机器人的指尖）在平面上移动，去**推**一个灰色的 **T 形方块**，把它推到
    指定的位置和角度。agent 不能抓，只能推，所以要绕到合适的一侧再推，这是它难的地方。

    物理用 pymunk（2D 刚体引擎），没有重力，地面有摩擦。
    """)
    return


@app.cell
def _(gym):
    env = gym.make("swm/PushT-v1", render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    return (env,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 坐标和状态

    世界是 **512 × 512 像素**（`env.window_size`），x 向右，**y 向下**（和图片一样）。
    渲染出的图是 224 × 224（`env.render_size`），所以图上 1 像素 ≈ 世界 2.3 像素。

    `state` 是 7 维：

    | 下标 | 含义 | 单位 |
    |---|---|---|
    | 0, 1 | agent 位置 x, y | 世界像素 |
    | 2, 3 | 方块位置 x, y（刚体原点，不一定是 T 的视觉中心） | 世界像素 |
    | 4 | 方块角度 | 弧度，取模 2π |
    | 5, 6 | agent 速度 vx, vy | 像素/秒 |

    `proprio`（本体感觉）= `state[[0, 1, 5, 6]]`，只有 agent 自己的位置和速度。
    注意状态里**没有方块速度**——方块的运动只能从连续几帧里推断出来。

    用滑块直接摆一个状态看看（通过 `reset(options={"state": ...})`）：
    """)
    return


@app.cell
def _(mo, np):
    ax_slider = mo.ui.slider(20, 492, value=150, label="agent x", show_value=True)
    ay_slider = mo.ui.slider(20, 492, value=400, label="agent y", show_value=True)
    bx_slider = mo.ui.slider(60, 452, value=300, label="block x", show_value=True)
    by_slider = mo.ui.slider(60, 452, value=300, label="block y", show_value=True)
    bang_slider = mo.ui.slider(0.0, float(2 * np.pi), step=0.05, value=1.0, label="block angle (rad)", show_value=True)
    mo.hstack([mo.vstack([ax_slider, ay_slider]), mo.vstack([bx_slider, by_slider, bang_slider])])
    return ax_slider, ay_slider, bang_slider, bx_slider, by_slider


@app.cell
def _(ax_slider, ay_slider, bang_slider, bx_slider, by_slider, env, np):
    manual_state = np.array(
        [ax_slider.value, ay_slider.value, bx_slider.value, by_slider.value, bang_slider.value, 0.0, 0.0]
    )
    _obs, manual_info = env.reset(seed=0, options={"state": manual_state})
    manual_frame = env.render()
    manual_obs_state = _obs["state"]
    return manual_frame, manual_info, manual_obs_state


@app.cell
def _(env, manual_frame, manual_info, manual_obs_state, np, plt):
    _s = env.render_size / env.window_size  # world px -> image px
    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    _ax.imshow(manual_frame)
    _ax.plot(*(manual_obs_state[:2] * _s), "wx", ms=10, mew=2)
    _ax.plot(*(manual_obs_state[2:4] * _s), "r+", ms=14, mew=2, label="block body origin")
    _gp = manual_info["goal_pose"]
    _ax.plot(*(_gp[:2] * _s), "g+", ms=14, mew=2, label=f"goal_pose {np.round(_gp, 2).tolist()}")
    _ax.legend(loc="upper left", fontsize=8)
    _ax.set_title(f"state = {np.round(manual_obs_state, 2).tolist()}", fontsize=9)
    _ax.axis("off")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 动作：相对位移 + PD 控制器

    动作 `a ∈ [-1, 1]²` 不是速度也不是力，而是一个**相对目标点**：

    ```text
    target = agent_pos + a * 100        # env.action_scale = 100，所以一步最多指向 100 像素外
    ```

    然后一个 PD 控制器把 agent 往 `target` 拉：加速度 = `k_p·(target − pos) − k_v·vel`
    （`k_p = 100`，`k_v = 20`），每个环境步内跑 **10 个物理子步**（`dt = 0.01 s`，
    控制频率 10 Hz）。所以一步之内 agent **走不到** target，只走一部分；下一步 target
    又按新位置重新算。

    下面从静止开始，连续 15 步都给同一个动作 `(0.5, 0)`：
    """)
    return


@app.cell
def _(env, np, plt):
    env.reset(seed=0, options={"state": np.array([100.0, 100.0, 400.0, 400.0, 0.0, 0.0, 0.0])})
    _xs, _vs = [env.agent.position.x], [env.agent.velocity.x]
    for _ in range(15):
        env.step(np.array([0.5, 0.0], dtype=np.float32))
        _xs.append(env.agent.position.x)
        _vs.append(env.agent.velocity.x)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.5))
    _axes[0].plot(np.diff(_xs), "o-")
    _axes[0].axhline(50, color="r", ls=":", label="commanded offset (0.5 × 100)")
    _axes[0].set_title("distance moved per env step (px)")
    _axes[0].set_xlabel("step")
    _axes[0].legend()
    _axes[1].plot(_vs, "o-")
    _axes[1].set_title("agent vx at end of step (px/s)")
    _axes[1].set_xlabel("step")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    第一步从静止出发只走了约 15 像素，之后稳定在每步约 20 像素——只有指令位移 50 的
    40% 左右。原因：target 跟着 agent 一起往前移，拉力 `k_p·50` 和阻尼 `k_v·v` 很快平衡，
    速度就固定了。**同一个动作的效果取决于当前速度**——这就是为什么世界模型需要看**好几帧
    历史**（LeWM 用 3 帧）才能预测得准。

    ## 3. 推方块

    把 agent 放在方块下方，一直往上推（`a = (0, -0.3)`），看方块怎么动。`info["n_contacts"]`
    是这一步里 agent 和方块的接触点数。
    """)
    return


@app.cell
def _(mo):
    push_x = mo.ui.slider(200, 320, value=256, label="agent 起始 x（改变推的位置 → 方块会转）", show_value=True)
    push_x
    return (push_x,)


@app.cell
def _(env, np, plt, push_x):
    env.reset(seed=0, options={"state": np.array([push_x.value, 450.0, 256.0, 300.0, 0.0, 0.0, 0.0])})
    _frames, _poses, _contacts = [env.render()], [np.array([*env.block.position, env.block.angle])], [0]
    for _ in range(12):
        _, _, _, _, _info = env.step(np.array([0.0, -0.3], dtype=np.float32))
        _frames.append(env.render())
        _poses.append(_info["block_pose"])
        _contacts.append(_info["n_contacts"])
    _poses = np.array(_poses)

    _fig = plt.figure(figsize=(14, 5.5))
    for _k, _t in enumerate([0, 3, 6, 9, 12]):
        _a = _fig.add_subplot(2, 5, _k + 1)
        _a.imshow(_frames[_t])
        _a.set_title(f"step {_t}")
        _a.axis("off")
    _a = _fig.add_subplot(2, 3, 4)
    _a.plot(_poses[:, 1])
    _a.set_title("block y (px)")
    _a = _fig.add_subplot(2, 3, 5)
    _a.plot(np.degrees(_poses[:, 2]))
    _a.set_title("block angle (deg)")
    _a = _fig.add_subplot(2, 3, 6)
    _a.bar(range(len(_contacts)), _contacts)
    _a.set_title("n_contacts")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    正对 T 的竖杆底部推，方块基本平移；推偏了（x 远离 256），方块就会转。

    ## 4. 两个"目标"：`goal_pose` 和 `goal_state`（容易混！）

    PushT 里有两样东西都叫目标：

    | | 是什么 | 用在哪 |
    |---|---|---|
    | `env.goal_pose` | `(x, y, angle)`，默认 `(256, 256, π/4)`。画面上那个**绿色的 T** 就画在这里 | 只是画出来；`eval.py` 的 full-solve 模式把它当目标 |
    | `env.goal_state` | 一个完整的 7 维状态（agent + 方块）。`reset` 时**随机抽**一个（除非 `options["goal_state"]` 指定） | `info["goal"]` 这张**目标图像**就是把环境摆成 `goal_state` 渲染出来的；`step` 的 `reward` 和 `terminated` 也按它算 |

    所以默认 `reset` 下，policy 被要求到达的是 `info["goal"]` 图里那个随机状态，**不是**
    绿色 T。两者一起画出来看：
    """)
    return


@app.cell
def _(env, np, plt):
    _fig, _axes = plt.subplots(2, 4, figsize=(13, 6.5))
    for _i in range(4):
        _obs, _info = env.reset(seed=10 + _i)
        _axes[0, _i].imshow(env.render())
        _axes[0, _i].set_title(f"seed {10 + _i}: start")
        _axes[1, _i].imshow(_info["goal"])
        _axes[1, _i].set_title(f"goal image (goal_state)\nblock={np.round(_info['goal_state'][2:5], 1).tolist()}", fontsize=8)
    for _ax in _axes.flat:
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    每张图里都有绿色 T（`goal_pose`，固定不动），而下排目标图里的灰色 T 在各不相同的
    位置——那才是 `terminated` 要求的目标。

    ## 5. 成功条件

    **环境自带**（`PushT.eval_state`，决定 `step` 返回的 `terminated`）：

    ```python
    pos_diff   = ‖goal_state[:4] − state[:4]‖        # agent 位置和方块位置**一起**算
    angle_diff = 方块角度差（取 2π 周期内最小值）
    success    = pos_diff < 20 and angle_diff < π/9   # 20 像素、20°
    reward     = −‖goal_state − state‖               # 7 维全部参与
    ```

    注意 `pos_diff` 把 **agent 的位置也算进去了**：方块推到位但 agent 停在别处，也不算成功。
    这就是 `eval.py` 的 dataset 模式（从专家数据里取第 t 帧为起点、t+25 帧为目标）用的规则。

    **full-solve 模式**（`eval.py:BlockOnGoalSuccess`）把它换成：只看**方块**和
    **绿色 T**（`goal_pose`），阈值一样（20 像素、π/9），不管 agent 在哪。

    用上面第 1 节的滑块摆状态，这里实时算两个规则（环境自带规则的目标是
    "方块在绿 T 上、agent 停在 (153, 356)"，也就是 full-solve 渲染目标图时用的状态）：
    """)
    return


@app.cell
def _(env, manual_obs_state, mo, np):
    _goal_state = np.concatenate([[153.0, 356.0], env.goal_pose, [0.0, 0.0]])
    _success, _dist = env.eval_state(_goal_state, manual_obs_state)
    _pos_diff = np.linalg.norm(_goal_state[:4] - manual_obs_state[:4])
    _block_pos = np.linalg.norm(manual_obs_state[2:4] - env.goal_pose[:2])
    _block_ang = abs((manual_obs_state[4] - env.goal_pose[2] + np.pi) % (2 * np.pi) - np.pi)
    _block_ok = _block_pos < 20 and _block_ang < np.pi / 9
    mo.md(
        f"""
    | 规则 | 位置误差 | 角度误差 | 成功？ |
    |---|---|---|---|
    | `env.eval_state`（agent + 方块） | {_pos_diff:.1f} px | {np.degrees(min(abs(_goal_state[4] - manual_obs_state[4]), 2 * np.pi - abs(_goal_state[4] - manual_obs_state[4]))):.1f}° | {"✅" if _success else "❌"} |
    | `BlockOnGoalSuccess`（只看方块） | {_block_pos:.1f} px | {np.degrees(_block_ang):.1f}° | {"✅" if _block_ok else "❌"} |

    reward = {-_dist:.1f}。试试把方块拖到 x=256, y=256, 角度≈0.79：第二条会变 ✅；
    再把 agent 拖到 (153, 356)，第一条也变 ✅。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结

    - 世界 512 px、y 向下；`state` 7 维（agent 位置、方块位置+角度、agent 速度），没有方块速度；
    - 动作是"往哪儿拉"的相对目标点（×100 px），PD 控制，同一动作效果取决于当前速度；
    - `goal_pose` = 画出来的绿 T；`goal_state` = 目标图像和 `terminated` 用的完整状态；
    - 两种成功规则：环境自带（含 agent）和 full-solve（只看方块对绿 T）。

    下一篇：专家数据长什么样。
    """)
    return


if __name__ == "__main__":
    app.run()
