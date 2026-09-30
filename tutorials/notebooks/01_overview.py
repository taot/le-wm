import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import stable_worldmodel as swm

    return mo, np, plt, swm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 · stable-worldmodel 全貌

    `stable-worldmodel`（下面简称 **swm**，`import stable_worldmodel as swm`）是一个做
    **世界模型（world model）研究** 的工具包。它不负责模型本身的结构和训练目标——那是
    我们这个仓库（LeWM，`jepa.py` / `train.py`）的事——它负责模型**周围**的所有东西：

    - 一组统一接口的**环境**（PushT、TwoRoom、DMControl、OGBench……）；
    - 批量跑环境、收集数据、做评估的 **`World`**；
    - 数据集的**读写**（lance / hdf5 / folder / video / lerobot）；
    - 基于世界模型做**规划（planning）**的 **policy + solver**（CEM、MPPI、梯度下降……）。

    这套教程分两部分：

    | 编号 | 内容 |
    |---|---|
    | 01 | 全貌（本篇） |
    | 02 | 环境与 variation space |
    | 03 | `World`、policy、rollout |
    | 04 | 数据集 |
    | 05 | solver 与 MPC 规划 |
    | 06 | PushT：物理、状态、动作、成功条件 |
    | 07 | PushT：专家数据集 |
    | 08 | PushT：LeWM 世界模型（编码、预测、代价） |
    | 09 | PushT：用 CEM 规划并评估 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 包里有什么

    | 子模块 | 作用 | 最常用的东西 |
    |---|---|---|
    | `swm.envs` | 注册到 gymnasium 的环境，id 都以 `swm/` 开头 | `gym.make("swm/PushT-v1")`，`swm.envs.WORLDS` |
    | `swm.spaces` | 扩展版 gym space，带"当前值"和"初始值"，用来描述**环境的可变因素**（颜色、形状、起点……） | `env.unwrapped.variation_space` |
    | `swm.World` | N 个并行环境 + 统一的预处理 + rollout 循环 | `World(...)`, `.set_policy`, `.evaluate`, `.collect` |
    | `swm.wrapper` | `MegaWrapper`（把观测统一放进 `info` 字典、缩放图像）和一堆视觉扰动 wrapper | `NoiseWrapper`, `ColorJitterWrapper` … |
    | `swm.policy` | 决策者：随机、专家、前馈网络、**世界模型 + 规划** | `RandomPolicy`, `WorldModelPolicy`, `PlanConfig` |
    | `swm.solver` | 规划优化器：给定"动作序列 → 代价"，找代价最低的动作序列 | `CEMSolver`, `ICEMSolver`, `MPPISolver`, `GradientSolver` |
    | `swm.wm` | 几个现成的世界模型（LeWM/PLDM/DINO-WM…）和加载/保存工具 | `swm.wm.utils.load_pretrained` |
    | `swm.data` | 数据集格式、读取、内存 replay buffer | `load_dataset`, `ReplayBuffer`, `list_formats` |

    下面几个 cell 直接从包里把这些东西"列"出来。
    """)
    return


@app.cell
def _(swm):
    # Registered envs, grouped by family (the part of the id before the version).
    import re
    from collections import defaultdict

    _groups: dict[str, list[str]] = defaultdict(list)
    for _id in sorted(swm.envs.WORLDS):
        _name = _id.removeprefix("swm/")
        _family = re.sub(r"(DMControl|Control|Dense|Dict|Pixels|Symbolic|Classic).*|-v\d+$", "", _name)
        _groups[_family].append(_name)
    env_groups = dict(_groups)
    return (env_groups,)


@app.cell
def _(env_groups, mo, swm):
    _rows = ["| 家族 | 环境 id（去掉 `swm/` 前缀） |", "|---|---|"]
    for _fam, _ids in env_groups.items():
        _rows.append(f"| {_fam} | {', '.join(f'`{i}`' for i in _ids)} |")
    mo.md(f"### 已注册的环境（共 {len(swm.envs.WORLDS)} 个）\n\n" + "\n".join(_rows))
    return


@app.cell
def _(mo, swm):
    mo.md(
        "### 可用的 solver\n\n"
        + "\n".join(f"- `swm.solver.{name}`" for name in swm.solver.__all__)
        + "\n\n### 数据格式\n\n"
        + ", ".join(f"`{f}`" for f in swm.data.list_formats())
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 数据是怎么流动的

    swm 里所有东西都围绕一个 **`info` 字典** 转：环境每一步把观测（图像 `pixels`、
    本体状态 `proprio`、目标图像 `goal` ……）都放进这个字典；policy 读它、输出动作；
    如果是世界模型 policy，它再把字典交给 solver，solver 反复调用模型的
    `get_cost(info, 候选动作)` 来挑动作。
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        """
        flowchart LR
          subgraph World
            E[EnvPool<br/>N 个 gym 环境] --> W[MegaWrapper<br/>缩放图像 / 放进 info]
          end
          W -- "infos 字典<br/>(N, 1, ...)" --> P[policy.get_action]
          P -- "动作 (N, action_dim)" --> E
          P -. 世界模型 policy .-> S[solver.solve<br/>CEM / MPPI / GD]
          S -- "候选动作序列" --> M[model.get_cost<br/>例如 jepa.JEPA]
          M -- "每个候选的代价" --> S
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    这个仓库和 swm 的分工：

    | 在哪 | 用到 swm 的什么 |
    |---|---|
    | `train.py` | `swm.data.load_dataset` 读 PushT 专家数据，`swm.data.utils.get_cache_dir` 决定 checkpoint 存哪 |
    | `jepa.py` | `JEPA` 实现了 swm 要求的 `get_cost(info, action_candidates)` 接口（`swm.solver.Costable` 协议） |
    | `eval.py` | `swm.World` 建环境，`swm.wm.utils.load_pretrained` 读模型，`CEMSolver` + `WorldModelPolicy` 做规划，`world.evaluate` 算成功率 |

    所有数据和 checkpoint 都在 `$STABLEWM_HOME` 下（没设置时是 `~/.stable_worldmodel`）：
    """)
    return


@app.cell
def _(mo, swm):
    _home = swm.data.utils.get_cache_dir()
    _sub = sorted(p.name for p in _home.iterdir()) if _home.exists() else []
    mo.md(f"`get_cache_dir()` = `{_home}`，里面有：{', '.join(f'`{s}/`' for s in _sub) or '（空）'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 十行代码跑一遍

    下面是 swm 最小的完整用法：建一个有 4 个并行 PushT 环境的 `World`，挂一个随机
    policy，跑 4 个 episode 并算成功率。随机乱推当然不会成功，成功率是 0——后面的
    教程会一步步换成世界模型 + 规划。
    """)
    return


@app.cell
def _(swm):
    world = swm.World("swm/PushT-v1", num_envs=4, image_shape=(96, 96), max_episode_steps=50)
    world.set_policy(swm.policy.RandomPolicy(seed=0))
    results = world.evaluate(episodes=4, seed=0)
    results
    return (world,)


@app.cell
def _(plt, world):
    # After evaluate, world.infos holds the last step of every env.
    _fig, _axes = plt.subplots(2, 4, figsize=(12, 6))
    for _i in range(4):
        _axes[0, _i].imshow(world.infos["pixels"][_i, 0])
        _axes[0, _i].set_title(f"env {_i}: current")
        _axes[1, _i].imshow(world.infos["goal"][_i, 0])
        _axes[1, _i].set_title(f"env {_i}: goal")
    for _ax in _axes.flat:
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    上面一行是每个环境最后一帧，下面一行是它的**目标图像**（`goal`）：PushT 的目标是把
    灰色的 T 推到和目标图像里一样的位置和角度。注意每个环境的起点和目标都不同——这是
    variation space 按 seed 随机抽出来的，下一篇（02）细讲。
    """)
    return


if __name__ == "__main__":
    app.run()
