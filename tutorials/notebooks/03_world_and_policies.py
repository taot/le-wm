import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import tempfile
    from functools import partial
    from pathlib import Path
    from typing import Any

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import stable_worldmodel as swm
    from stable_worldmodel.envs.pusht import WeakPolicy

    return Any, Path, WeakPolicy, mo, np, partial, plt, swm, tempfile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 03 · World、policy 和 rollout

    上一篇是**一个**环境。做研究时我们通常要**同时跑很多个**（评估 50 个 episode、
    收集几万条数据），而且希望所有环境的输出格式统一。`swm.World` 就是干这个的：

    ```python
    world = swm.World("swm/PushT-v1", num_envs=4, image_shape=(96, 96), max_episode_steps=100)
    world.set_policy(policy)
    world.evaluate(episodes=8, seed=0)      # 评估
    world.collect(writer=..., episodes=8)   # 收集数据（04 篇）
    ```

    它内部有三样东西：

    1. `world.envs`：一个 `EnvPool`，装着 `num_envs` 个 gym 环境，`step` 时一起走；
    2. 每个环境外面套了 `MegaWrapper`：渲染图像并缩放到 `image_shape`、把 obs 里的东西
       也搬进 `info`，保证所有环境的 `info` 键都一样；
    3. 一个 rollout 循环：`policy.get_action(world.infos)` → `envs.step` → 处理结束/重置。
    """)
    return


@app.cell
def _(swm):
    N_ENVS = 4
    world = swm.World("swm/PushT-v1", num_envs=N_ENVS, image_shape=(96, 96), max_episode_steps=100)
    world.reset(seed=0)
    return N_ENVS, world


@app.cell
def _(mo, world):
    def _fmt(v: object) -> str:
        shape = getattr(v, "shape", None)
        if shape is None:
            return f"`{type(v).__name__}` (len {len(v)})" if isinstance(v, list) else f"`{v!r}`"
        return f"`{tuple(shape)}` {getattr(v, 'dtype', '')}"

    _rows = ["| 键 | 形状 / 类型 |", "|---|---|"] + [f"| `{k}` | {_fmt(v)} |" for k, v in world.infos.items()]
    mo.md(
        "## 1. `world.infos`：policy 看到的东西\n\n"
        "`world.reset(seed=0)` 之后，`world.infos` 是所有环境 `info` **堆叠**起来的字典：\n\n"
        + "\n".join(_rows)
        + "\n\n形状统一是 **`(num_envs, 1, ...)`**：第一维是第几个环境，第二维是**时间/历史**维度"
        "（这里只保留当前这 1 步）。`pixels` 已经是缩放后的 `image_shape`；环境原本 obs 里的 "
        "`proprio`、`state` 也被搬进来了；`reward/terminated/truncated/action/step_idx` 是 "
        "MegaWrapper 额外记录的。"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `world.envs.envs[i]` 是第 i 个（被包过的）环境，`.unwrapped` 拿到最里面真正的
    `PushT` 对象，可以直接读它的物理状态（06 篇会大量用到）：
    """)
    return


@app.cell
def _(world):
    _raw = world.envs.envs[0].unwrapped
    type(_raw).__name__, tuple(_raw.agent.position), tuple(_raw.block.position), _raw.block.angle
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. policy 是什么

    policy 就是有 `get_action(infos) -> actions` 方法的对象（继承 `swm.policy.BasePolicy`）。
    `world.set_policy(p)` 会调用 `p.set_env(world.envs)`，让 policy 知道动作空间和环境个数。
    返回的动作形状必须是 `(num_envs, action_dim)`。

    swm 自带的：

    | policy | 做什么 |
    |---|---|
    | `swm.policy.RandomPolicy` | 在动作空间里均匀随机 |
    | `swm.envs.pusht.WeakPolicy` | PushT 的"弱专家"：随机动作，但把目标点限制在方块附近，这样经常碰到方块。用来**收集数据** |
    | `swm.policy.FeedForwardPolicy` | 一个网络直接从 info 输出动作 |
    | `swm.policy.WorldModelPolicy` | 世界模型 + solver 规划（05、09 篇） |

    自己写一个也很简单。下面这个 `ChaseBlockPolicy` 每一步都朝方块中心走：
    """)
    return


@app.cell
def _(Any, np, swm):
    class ChaseBlockPolicy(swm.policy.BasePolicy):
        """Move the agent straight toward the block's center every step."""

        def get_action(self, infos: dict[str, Any], **kwargs: Any) -> np.ndarray:
            agent = infos["pos_agent"][:, -1]  # (num_envs, 2), world pixels
            block = infos["block_pose"][:, -1, :2]
            # PushT actions are relative offsets in units of 100 px (06 explains why).
            return np.clip((block - agent) / 100.0, -1.0, 1.0).astype(np.float32)

    return (ChaseBlockPolicy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 手写一个 rollout 循环

    `world.evaluate` / `world.collect` 内部就是下面这个循环（简化版）。手写一遍，能看清
    每一步发生了什么，也方便记录自己关心的量——这里记录每个环境里方块离目标的距离。
    """)
    return


@app.cell
def _(mo):
    policy_choice = mo.ui.dropdown(
        options=["RandomPolicy", "WeakPolicy", "ChaseBlockPolicy"], value="ChaseBlockPolicy", label="policy"
    )
    policy_choice
    return (policy_choice,)


@app.cell
def _(Any, ChaseBlockPolicy, WeakPolicy, policy_choice, swm):
    def make_policy(name: str) -> Any:
        if name == "RandomPolicy":
            return swm.policy.RandomPolicy(seed=0)
        if name == "WeakPolicy":
            return WeakPolicy(seed=0)
        return ChaseBlockPolicy()

    policy = make_policy(policy_choice.value)
    return make_policy, policy


@app.cell
def _(N_ENVS, np, policy, world):
    world.set_policy(policy)
    world.reset(seed=0)

    STEPS = 100
    block_goal_dist = np.zeros((STEPS, N_ENVS))
    for _t in range(STEPS):
        _actions = policy.get_action(world.infos)  # (num_envs, 2)
        _obs, _rewards, _terminated, _truncated, world.infos = world.envs.step(_actions)
        _block = world.infos["block_pose"][:, -1, :2]
        _goal = world.infos["goal_pose"][:, -1, :2]
        block_goal_dist[_t] = np.linalg.norm(_block - _goal, axis=1)
    final_frames = world.infos["pixels"][:, -1].copy()
    return block_goal_dist, final_frames


@app.cell
def _(N_ENVS, block_goal_dist, final_frames, plt, policy_choice):
    _fig = plt.figure(figsize=(13, 4))
    _ax = _fig.add_subplot(1, N_ENVS + 2, (1, 2))
    _ax.plot(block_goal_dist)
    _ax.set_xlabel("step")
    _ax.set_ylabel("block ↔ goal distance (px)")
    _ax.set_title(policy_choice.value)
    for _i in range(N_ENVS):
        _a = _fig.add_subplot(1, N_ENVS + 2, _i + 3)
        _a.imshow(final_frames[_i])
        _a.set_title(f"env {_i} @ step 100")
        _a.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    三种 policy 都不"知道"目标在哪，所以方块只会被乱推。要真正推到目标，policy 必须
    看 `goal`——这就是世界模型规划要做的事。

    ## 4. `world.evaluate`：跑完整的评估

    `evaluate(episodes, seed)` 把上面的循环包好了：某个环境结束（`terminated` 成功或
    `truncated` 超时）后自动用新 seed 重置，直到凑满 `episodes` 个 episode，返回成功率。
    传 `video=目录` 会给每个 episode 存一个 mp4。
    """)
    return


@app.cell
def _(Path, make_policy, mo, policy_choice, swm, tempfile):
    _video_dir = Path(tempfile.mkdtemp(prefix="swm_tutorial_"))
    _w = swm.World("swm/PushT-v1", num_envs=4, image_shape=(96, 96), max_episode_steps=60)
    _w.set_policy(make_policy(policy_choice.value))
    eval_results = _w.evaluate(episodes=4, seed=0, video=_video_dir)
    _w.close()
    _videos = sorted(_video_dir.glob("episode_*.mp4"))
    mo.vstack(
        [
            mo.md(f"`evaluate` 返回：`{eval_results}`"),
            mo.hstack([mo.video(src=p.read_bytes(), width=200, autoplay=True, loop=True) for p in _videos]),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 在 World 上加 wrapper

    `World(..., extra_wrappers=[...])` 在 `MegaWrapper` **之后**再包一层，常用来加视觉扰动，
    测试模型对"看起来不一样"的鲁棒性；`pre_wrappers` 则包在 `MegaWrapper` **之前**
    （改环境本身，比如 `eval.py` 用 `extra_wrappers=[BlockOnGoalSuccess]` 换掉成功条件）。
    `swm.wrapper` 里现成的：
    """)
    return


@app.cell
def _(mo, swm):
    mo.md(", ".join(f"`{n}`" for n in swm.wrapper.__all__ if n.endswith("Wrapper")))
    return


@app.cell
def _(partial, plt, swm):
    _variants = {
        "none": [],
        "NoiseWrapper(std=40)": [partial(swm.wrapper.NoiseWrapper, std=40.0, seed=0)],
        "ColorJitterWrapper": [partial(swm.wrapper.ColorJitterWrapper, brightness=0.5, hue=0.2, seed=0)],
        "BlurWrapper(kernel=9)": [partial(swm.wrapper.BlurWrapper, kernel=9)],
        "CutoutWrapper": [partial(swm.wrapper.CutoutWrapper, num=3, size=(0.15, 0.25), seed=0)],
    }
    _fig, _axes = plt.subplots(1, len(_variants), figsize=(3 * len(_variants), 3.2))
    for _ax, (_name, _wrappers) in zip(_axes, _variants.items()):
        _w = swm.World("swm/PushT-v1", num_envs=1, image_shape=(96, 96), extra_wrappers=_wrappers)
        _w.reset(seed=0)
        _ax.imshow(_w.infos["pixels"][0, -1])
        _ax.set_title(_name, fontsize=9)
        _ax.axis("off")
        _w.close()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结

    - `World` = N 个并行环境 + `MegaWrapper` + rollout 循环；
    - `world.infos` 的所有值形状都是 `(num_envs, 1, ...)`；
    - policy 只需要实现 `get_action(infos) -> (num_envs, action_dim)`；
    - `evaluate` 算成功率，`extra_wrappers` / `pre_wrappers` 改观测或环境。

    下一篇：用 `World.collect` 收集数据，以及读取已有的数据集。
    """)
    return


if __name__ == "__main__":
    app.run()
