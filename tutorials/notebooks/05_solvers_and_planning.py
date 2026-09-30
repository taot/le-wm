import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    from typing import Any

    import gymnasium as gym
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import stable_worldmodel as swm
    import torch
    from matplotlib.patches import Circle
    from stable_worldmodel.solver.callbacks import Callback

    return Any, Callback, Circle, gym, mo, np, plt, swm, time, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 05 · solver 与 MPC 规划

    有了世界模型以后，怎么用它来**选动作**？思路是"在脑子里试"：

    1. 随便想很多条**动作序列**（比如 300 条，每条 5 步）；
    2. 用世界模型**预测**每条序列执行后会到哪里；
    3. 用一个**代价**（cost）打分：预测的终点离目标越远，代价越高；
    4. 挑代价最低的——或者更聪明地，根据打分**改进**候选，再试一轮。

    第 4 步就是 **solver** 做的事。swm 把这件事拆成两个接口（`swm/solver/solver.py`）：

    | 接口 | 谁实现 | 要做什么 |
    |---|---|---|
    | `Costable` | 世界模型（例如 `jepa.JEPA`） | `get_cost(info_dict, action_candidates) -> costs`：输入候选动作 `(B, S, H, D)`，返回代价 `(B, S)` |
    | `Solver` | `CEMSolver`、`ICEMSolver`、`MPPISolver`、`GradientSolver` …… | `configure(...)`，然后 `solve(info_dict) -> {"actions": (B, H, D), ...}` |

    `B` = 环境数，`S` = 每个环境的候选数，`H` = 规划步数（horizon），`D` = 每步动作维度。

    solver **完全不知道**模型是什么，只会调 `get_cost`。所以这一篇先不用神经网络，
    用一个**解析的玩具模型**把 solver 看清楚——跑得飞快，CPU 就够。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 玩具模型：绕开障碍走到目标

    一个在平面上的点，每一步的动作 `a ∈ [-1, 1]²` 就是位移。模型预测的轨迹是
    `p_t = start + a_1 + … + a_t`。代价 = 轨迹上**每一步**到目标距离平方的平均（鼓励尽快
    到达并停在那里）+ 穿过圆形障碍的惩罚。
    直线过去会撞障碍，所以 solver 得学会绕路。
    """)
    return


@app.cell
def _(torch):
    OBSTACLE_C = torch.tensor([5.0, 4.6])
    OBSTACLE_R = 2.0

    class ToyPointModel:
        """Analytic 'world model': the point moves by the (clipped) action each step."""

        def rollout(self, start: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            # start (..., 2), actions (..., H, 2) -> positions (..., H, 2)
            return start.unsqueeze(-2) + torch.cumsum(actions.clamp(-1, 1), dim=-2)

        def get_cost(self, info_dict: dict[str, torch.Tensor], action_candidates: torch.Tensor) -> torch.Tensor:
            # info tensors arrive expanded by the solver to (B, S, ...), candidates are (B, S, H, D).
            p = self.rollout(info_dict["start"], action_candidates)
            # Mean over the whole path (not just the end), so plans get there early and stay.
            goal_cost = ((p - info_dict["goal_pos"].unsqueeze(-2)) ** 2).sum(-1).mean(-1)
            dist = torch.linalg.norm(p - OBSTACLE_C, dim=-1)
            obstacle_cost = 20.0 * torch.relu(OBSTACLE_R - dist).sum(-1)
            return goal_cost + obstacle_cost  # (B, S)

    toy_model = ToyPointModel()
    START = torch.tensor([[1.0, 1.0]])  # (n_envs=1, 2)
    GOAL = torch.tensor([[9.0, 9.0]])
    return GOAL, OBSTACLE_C, OBSTACLE_R, START, toy_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. CEM：交叉熵方法

    本仓库评估用的就是 `CEMSolver`（`config/eval/solver/cem.yaml`）。它维护每一步动作的
    一个高斯分布（均值 `mean`、标准差 `var`），反复做：

    ```text
    从 N(mean, var) 采 num_samples 条动作序列（第 0 条固定为 mean）
    costs = model.get_cost(info, candidates)
    取代价最低的 topk 条（"精英"）
    mean, var = 精英的均值, 精英的标准差
    ```
    重复 `n_steps` 轮，最后返回 `mean` 作为规划结果。

    solver 用 `configure(action_space, n_envs, config)` 知道动作维度和 horizon。
    `config` 是 `swm.PlanConfig`（第 4 节细讲）。为了看每一轮发生了什么，这里写一个
    **callback**（`swm.solver.callbacks.Callback` 的子类），每轮把候选、精英和均值存下来。
    """)
    return


@app.cell
def _(Any, Callback, torch):
    class SnapshotRecorder(Callback):
        """Keep the candidates, elites and new mean of env 0 at every CEM iteration."""

        def compute(self, **state: Any) -> dict[str, torch.Tensor]:
            return {
                "candidates": state["candidates"][0].cpu().clone(),
                "elites": state["topk_candidates"][0].cpu().clone(),
                "mean": state["mean"][0].cpu().clone(),
                "costs": state["costs"][0].cpu().clone(),
            }

    return (SnapshotRecorder,)


@app.cell
def _(mo):
    horizon_slider = mo.ui.slider(5, 25, value=15, label="horizon H", show_value=True)
    samples_slider = mo.ui.slider(20, 500, step=20, value=200, label="num_samples", show_value=True)
    topk_slider = mo.ui.slider(5, 100, step=5, value=20, label="topk", show_value=True)
    iters_slider = mo.ui.slider(1, 40, value=20, label="n_steps (CEM 轮数)", show_value=True)
    mo.vstack([horizon_slider, samples_slider, topk_slider, iters_slider])
    return horizon_slider, iters_slider, samples_slider, topk_slider


@app.cell
def _(
    GOAL,
    START,
    SnapshotRecorder,
    gym,
    horizon_slider,
    iters_slider,
    np,
    samples_slider,
    swm,
    topk_slider,
    toy_model,
):
    ACTION_SPACE = gym.spaces.Box(-1, 1, shape=(1, 2), dtype=np.float32)  # batched: (n_envs, action_dim)

    _recorder = SnapshotRecorder()
    cem = swm.solver.CEMSolver(
        toy_model,
        num_samples=samples_slider.value,
        topk=min(topk_slider.value, samples_slider.value),
        n_steps=iters_slider.value,
        var_scale=1.0,
        seed=0,
        callbacks=[_recorder],
    )
    cem.configure(
        action_space=ACTION_SPACE, n_envs=1, config=swm.PlanConfig(horizon=horizon_slider.value, receding_horizon=1)
    )
    cem_out = cem.solve({"start": START.clone(), "goal_pos": GOAL.clone()})
    snapshots = cem_out["callbacks"]["SnapshotRecorder"][0]  # [batch 0] -> one entry per iteration
    return ACTION_SPACE, cem_out, snapshots


@app.cell
def _(mo, snapshots):
    iter_view = mo.ui.slider(0, len(snapshots) - 1, value=0, label="看第几轮", show_value=True)
    iter_view
    return (iter_view,)


@app.cell
def _(
    Circle,
    GOAL,
    OBSTACLE_C,
    OBSTACLE_R,
    START,
    iter_view,
    np,
    plt,
    snapshots,
    toy_model,
):
    _snap = snapshots[iter_view.value]
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _ax = _axes[0]
    _ax.add_patch(Circle(OBSTACLE_C.tolist(), OBSTACLE_R, color="0.8"))
    for _traj in toy_model.rollout(START[0], _snap["candidates"][:60]):
        _p = np.vstack([START[0].numpy(), _traj.numpy()])
        _ax.plot(_p[:, 0], _p[:, 1], color="tab:blue", alpha=0.15, lw=1)
    for _traj in toy_model.rollout(START[0], _snap["elites"]):
        _p = np.vstack([START[0].numpy(), _traj.numpy()])
        _ax.plot(_p[:, 0], _p[:, 1], color="tab:orange", alpha=0.6, lw=1.2)
    _p = np.vstack([START[0].numpy(), toy_model.rollout(START[0], _snap["mean"]).numpy()])
    _ax.plot(_p[:, 0], _p[:, 1], "k.-", lw=2.5, label="new mean")
    _ax.plot(*START[0].tolist(), "gs", ms=10, label="start")
    _ax.plot(*GOAL[0].tolist(), "r*", ms=16, label="goal")
    _ax.set_xlim(-6, 16)
    _ax.set_ylim(-6, 16)
    _ax.set_aspect("equal")
    _ax.legend(loc="lower right")
    _ax.set_title(f"iteration {iter_view.value}: 60 of the samples (blue), elites (orange)")

    _best = [s["costs"].min().item() for s in snapshots]
    _med = [s["costs"].median().item() for s in snapshots]
    _axes[1].semilogy(_best, label="best sample cost")
    _axes[1].semilogy(_med, label="median sample cost")
    _axes[1].axvline(iter_view.value, color="k", ls=":")
    _axes[1].set_xlabel("CEM iteration")
    _axes[1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    拖"看第几轮"：第 0 轮的样本是一大团乱线（均值 0、方差 1）；几轮之后精英集中到
    障碍的一侧，分布越缩越窄，最后收敛成一条绕开障碍的路径。

    可以试试：
    - `num_samples` 很小（20）时，容易收敛到很差的解——样本太少，没"见过"好的方向；
    - `topk` 很大（接近 `num_samples`）时，分布几乎不缩，收敛很慢；
    - `horizon` 太短（5），根本走不到目标（每步最多挪 √2）。

    ## 3. 其它 solver

    同样的 `configure` + `solve` 接口，换一个类就行。下面在同一个问题上比较四个：

    | solver | 思路 |
    |---|---|
    | `CEMSolver` | 上面讲的：采样 → 精英 → 重新拟合高斯 |
    | `ICEMSolver` | 改进版 CEM：有色噪声（时间上更平滑的样本）、保留上一轮部分精英、动量更新 |
    | `MPPISolver` | 不只取精英，而是按 `exp(-cost / temperature)` 给所有样本加权平均 |
    | `GradientSolver` | 不采样，直接对动作做梯度下降——要求代价对动作**可导**（我们的玩具模型可导，JEPA 也可导） |
    """)
    return


@app.cell
def _(ACTION_SPACE, GOAL, START, horizon_slider, swm, time, toy_model, torch):
    _solvers = {
        "CEM": swm.solver.CEMSolver(toy_model, num_samples=200, topk=20, n_steps=20, seed=0),
        "iCEM": swm.solver.ICEMSolver(toy_model, num_samples=200, topk=20, n_steps=20, seed=0),
        "MPPI": swm.solver.MPPISolver(toy_model, num_samples=200, topk=20, n_steps=20, seed=0),
        "GD (Adam)": swm.solver.GradientSolver(
            toy_model, n_steps=100, num_samples=8, optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 0.1}, seed=0
        ),
    }
    solver_results = {}
    for _name, _solver in _solvers.items():
        _solver.configure(
            action_space=ACTION_SPACE, n_envs=1, config=swm.PlanConfig(horizon=horizon_slider.value, receding_horizon=1)
        )
        _t0 = time.perf_counter()
        _actions = _solver.solve({"start": START.clone(), "goal_pos": GOAL.clone()})["actions"]
        _secs = time.perf_counter() - _t0
        _cost = toy_model.get_cost({"start": START, "goal_pos": GOAL}, _actions.detach()).item()
        solver_results[_name] = (_actions.detach(), _cost, _secs)
    return (solver_results,)


@app.cell
def _(
    Circle,
    GOAL,
    OBSTACLE_C,
    OBSTACLE_R,
    START,
    np,
    plt,
    solver_results,
    toy_model,
):
    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.add_patch(Circle(OBSTACLE_C.tolist(), OBSTACLE_R, color="0.85"))
    for _name, (_a, _cost, _secs) in solver_results.items():
        _p = np.vstack([START[0].numpy(), toy_model.rollout(START[0], _a[0]).numpy()])
        _ax.plot(_p[:, 0], _p[:, 1], ".-", label=f"{_name}: cost {_cost:.3f}, {_secs * 1000:.0f} ms")
    _ax.plot(*START[0].tolist(), "gs", ms=10)
    _ax.plot(*GOAL[0].tolist(), "r*", ms=16)
    _ax.set_aspect("equal")
    _ax.legend(fontsize=8, loc="lower right")
    _ax.set_title("same problem, four solvers")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. MPC：规划一段、只走一小段、再规划

    真实的世界模型是**不准**的。如果一次规划 H 步然后闭着眼睛全走完（开环），误差会
    越积越大。**模型预测控制（MPC）**的做法是：规划 H 步，但只执行前 `receding_horizon`
    步，然后看新的观测，**重新规划**。

    `swm.PlanConfig` 的字段：

    | 字段 | 含义 | PushT eval 里的值 |
    |---|---|---|
    | `horizon` | solver 规划多少个**模型步** | 5 |
    | `receding_horizon` | 每次规划后执行多少个模型步再重新规划 | 5（dataset 模式） |
    | `action_block` | 一个模型步 = 几个环境步（04 篇的 frameskip） | 5 |
    | `history_len` | 给模型看几帧历史 | 1 |
    | `warm_start` | 用上一次规划没执行完的部分作为下一次 CEM 的初始均值 | True |

    所以 PushT 的一次规划覆盖 `5 × 5 = 25` 个环境步。

    下面模拟"模型不准"：真实世界里有一股**风**，每步把点往左上方吹（x 减小、y 增大），但模型不知道。
    比较开环（规划一次走到底）和 MPC（每走 `receding_horizon` 步重新规划）。
    """)
    return


@app.cell
def _(mo):
    wind_slider = mo.ui.slider(0.0, 0.5, step=0.05, value=0.25, label="风力（模型不知道）", show_value=True)
    receding_slider = mo.ui.slider(1, 15, value=3, label="receding_horizon", show_value=True)
    mo.vstack([wind_slider, receding_slider])
    return receding_slider, wind_slider


@app.cell
def _(ACTION_SPACE, GOAL, START, np, swm, toy_model, torch):
    def run_mpc(receding: int, wind: float, horizon: int = 15, total_steps: int = 20) -> np.ndarray:
        """Plan with the toy model, act in a 'real' world that also has wind."""
        solver = swm.solver.CEMSolver(toy_model, num_samples=200, topk=20, n_steps=15, seed=0)
        solver.configure(
            action_space=ACTION_SPACE, n_envs=1, config=swm.PlanConfig(horizon=horizon, receding_horizon=receding)
        )
        pos = START.clone()
        path = [pos[0].numpy().copy()]
        warm: torch.Tensor | None = None
        wind_vec = torch.tensor([-wind, wind])
        steps = 0
        while steps < total_steps:
            plan = solver.solve({"start": pos.clone(), "goal_pos": GOAL.clone()}, init_action=warm)["actions"]
            for a in plan[0, :receding]:  # execute only the first `receding` actions
                pos = pos + a.clamp(-1, 1) + wind_vec  # the real world has wind
                path.append(pos[0].numpy().copy())
                steps += 1
            warm = plan[:, receding:]  # warm start: the rest of this plan seeds the next solve
        return np.array(path)

    return (run_mpc,)


@app.cell
def _(
    Circle,
    GOAL,
    OBSTACLE_C,
    OBSTACLE_R,
    plt,
    receding_slider,
    run_mpc,
    wind_slider,
):
    _open_loop = run_mpc(receding=15, wind=wind_slider.value, total_steps=15)
    _mpc = run_mpc(receding=receding_slider.value, wind=wind_slider.value)
    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.add_patch(Circle(OBSTACLE_C.tolist(), OBSTACLE_R, color="0.85"))
    _ax.plot(_open_loop[:, 0], _open_loop[:, 1], "o-", ms=3, label="open loop (plan once, 15 steps)")
    _ax.plot(_mpc[:, 0], _mpc[:, 1], "o-", ms=3, label=f"MPC, replan every {receding_slider.value} steps")
    _ax.plot(*GOAL[0].tolist(), "r*", ms=16)
    _ax.set_aspect("equal")
    _ax.legend(fontsize=8)
    _ax.set_title(f"wind = {wind_slider.value}")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    风越大，开环偏得越远，终点离目标很远；MPC 因为不断用**真实位置**重新规划，终点
    离目标近得多。它也不是完美的：模型始终不知道有风，每次都以为接下来会准确走到，
    所以风大时终点仍会差一点（一个持续存在的偏差）。`receding_horizon` 越小纠正越及时，但规划次数越多、越慢。

    ## 5. `WorldModelPolicy` 就是上面那个循环

    `swm.policy.WorldModelPolicy(solver, config, process, transform)` 把 MPC 循环封装成一个
    policy（`swm/policy.py`），每次 `get_action(infos)`：

    1. 用 `process`（标准化）和 `transform`（图像预处理）把 `infos` 转成模型要的张量；
    2. 如果某个环境的**动作缓存**空了：调用 `solver(infos, init_action=warm)` 规划；
       取前 `receding_horizon` 个模型步，按 `action_block` 展开成
       `receding_horizon × action_block` 个环境动作放进缓存；剩下的部分留作 warm start；
    3. 从每个环境的缓存里弹出一个动作，反标准化后返回。

    所以对 `World` 来说它和 `RandomPolicy` 没区别，都是"给 infos，拿动作"。09 篇会用
    真正的 LeWM 模型把它跑起来。

    ## 小结

    - 模型只需实现 `get_cost(info, candidates) -> (B, S)`；solver 只管优化；
    - CEM：采样 → 取精英 → 重拟合高斯；`num_samples`、`topk`、`n_steps` 决定质量和速度；
    - MPC：只执行规划的前一段就重新规划，用来对付模型误差；
    - `PlanConfig` + `WorldModelPolicy` 把这些接到 `World` 上。
    """)
    return


if __name__ == "__main__":
    app.run()
