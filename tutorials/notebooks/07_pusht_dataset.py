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
    # 07 · PushT 专家数据集

    LeWM 是从 `librakevin/lewm-pusht` 学出来的：约 1.9 万条**专家**（人或脚本）把 T 推到
    绿色目标上的演示。这一篇看看这些数据长什么样，以及 `eval.py` 怎样从里面挑评估题目。

    数据的读法见 04 篇；这里 `keys_to_cache` 把动作和状态整列读进内存，图像按需读。
    """)
    return


@app.cell
def _(swm):
    ds = swm.data.load_dataset("librakevin/lewm-pusht", keys_to_cache=["action", "proprio", "state"])
    lengths = ds.lengths.astype(int)
    offsets = ds.offsets.astype(int) - int(ds.offsets[0])  # row of each episode's first step
    all_state = ds.get_col_data("state")  # (total_steps, 7)
    all_action = ds.get_col_data("action")  # (total_steps, 2)
    return all_action, all_state, ds, lengths, offsets


@app.cell
def _(lengths, mo, np, plt):
    _fig, _ax = plt.subplots(figsize=(8, 3.2))
    _ax.hist(lengths, bins=60)
    _ax.set_xlabel("episode length (env steps)")
    _ax.set_ylabel("# episodes")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                f"**{len(lengths):,}** 条 episode，共 **{lengths.sum():,}** 步。长度中位数 "
                f"{int(np.median(lengths))} 步（10 Hz，约 {np.median(lengths) / 10:.0f} 秒），"
                f"90% 分位 {int(np.percentile(lengths, 90))} 步。"
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 看一条 episode

    选一条 episode，拖时间滑块。右边画出整条 episode 里 agent（蓝）和方块原点（灰）
    的路径，以及方块角度随时间的变化；黑点是当前时刻。
    """)
    return


@app.cell
def _(lengths, mo):
    ep_input = mo.ui.number(start=0, stop=len(lengths) - 1, value=3, label="episode")
    ep_input
    return (ep_input,)


@app.cell
def _(ds, ep_input, lengths, mo):
    episode = ds.load_episode(int(ep_input.value))
    t_slider = mo.ui.slider(0, int(lengths[int(ep_input.value)]) - 1, value=0, label="step", show_value=True)
    t_slider
    return episode, t_slider


@app.cell
def _(episode, np, plt, t_slider):
    _state = episode["state"].numpy()
    _t = t_slider.value
    _fig = plt.figure(figsize=(14, 4.5))
    _a = _fig.add_subplot(1, 3, 1)
    _a.imshow(episode["pixels"][_t].permute(1, 2, 0).numpy())
    _a.set_title(f"t = {_t}, action = {np.round(episode['action'][_t].numpy(), 2).tolist()}")
    _a.axis("off")

    _a = _fig.add_subplot(1, 3, 2)
    _a.plot(_state[:, 0], _state[:, 1], color="royalblue", lw=1, label="agent")
    _a.plot(_state[:, 2], _state[:, 3], color="gray", lw=2, label="block origin")
    _a.plot(_state[_t, 0], _state[_t, 1], "ko")
    _a.plot(_state[_t, 2], _state[_t, 3], "ko")
    _a.plot(256, 256, "g+", ms=15, mew=2, label="goal_pose")
    _a.set_xlim(0, 512)
    _a.set_ylim(512, 0)  # y down, like the image
    _a.set_aspect("equal")
    _a.legend(fontsize=8)
    _a.set_title("paths (world px)")

    _a = _fig.add_subplot(1, 3, 3)
    _a.plot(np.degrees(np.unwrap(_state[:, 4])), color="gray")
    _a.axhline(45, color="g", ls=":", label="goal angle (45°)")
    _a.axvline(_t, color="k", lw=0.8)
    _a.set_xlabel("step")
    _a.set_title("block angle (deg, unwrapped)")
    _a.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    典型的专家策略：先绕到方块合适的一侧，推一段、换个位置再推，最后微调角度。
    动作大部分时候很小（04 篇的直方图），只有"换位置"时才有大动作。

    ## 2. 专家最后都推到绿 T 上了吗？

    把每条 episode **最后一步**方块的位置和角度，和绿 T（`goal_pose = (256, 256, 45°)`）
    比较，用 full-solve 的阈值（20 px、20°）判断：
    """)
    return


@app.cell
def _(all_state, lengths, mo, np, offsets, plt):
    GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])
    _last = all_state[offsets + lengths - 1]
    _first = all_state[offsets]

    def block_error(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(position px, angle rad) error of the block vs the green T, for states (N, 7)."""
        pos = np.linalg.norm(s[:, 2:4] - GOAL_POSE[:2], axis=1)
        ang = np.abs((s[:, 4] - GOAL_POSE[2] + np.pi) % (2 * np.pi) - np.pi)
        return pos, ang

    _p_last, _a_last = block_error(_last)
    _p_first, _a_first = block_error(_first)
    _ok = (_p_last < 20) & (_a_last < np.pi / 9)

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _bins = np.linspace(0, 300, 61)
    _axes[0].hist(_p_first, bins=_bins, alpha=0.6, label="first step")
    _axes[0].hist(_p_last, bins=_bins, alpha=0.6, label="last step")
    _axes[0].axvline(20, color="k", ls=":")
    _axes[0].set_title("block position error vs green T (px)")
    _axes[0].legend()
    _axes[1].scatter(_last[_ok, 0], _last[_ok, 1], s=1, alpha=0.2)
    _axes[1].set_xlim(0, 512)
    _axes[1].set_ylim(512, 0)
    _axes[1].set_aspect("equal")
    _axes[1].set_title("final agent position (episodes that end on the T)")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                f"**{_ok.mean():.0%}** 的 episode 最后方块在绿 T 上（20 px、20° 以内）。"
                f"起点时方块离绿 T 的中位距离是 {np.median(_p_first):.0f} px。"
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    大约三分之二的演示是"完整解决"的，其余在推到之前就结束了。右图是成功的 episode
    结束时 agent 停在哪：比较集中在 T 的左下方。`config/eval/pusht_full.yaml` 里的
    `goal_agent_pos: [153, 356]` 就是这个最常见的位置——full-solve 评估需要一张目标图，
    图里 agent 总得在某个地方，就放在专家最常停的地方。

    ## 3. dataset 模式的评估题目是怎么出的

    `config/eval/pusht.yaml`（默认模式）不要求把方块推到绿 T，而是出"**短程**"题：

    1. 在数据集里随机挑一个位置 `(episode e, 第 t 步)`，要求 `t + 25` 还在这条 episode 内；
    2. 把环境摆成第 t 步的状态（`_set_state`），把第 `t + 25` 步设为目标（`_set_goal_state`，
       目标图像用数据集里那一帧的图像）；
    3. 给 policy 50 步（`eval_budget`），用**环境自带**的成功规则（agent + 方块，06 篇）判断。

    专家用 25 步就做到了，所以这些题目一定可解。下面按 `eval.py` 同样的方法抽几道：
    """)
    return


@app.cell
def _(mo):
    goal_offset = mo.ui.slider(5, 50, step=5, value=25, label="goal_offset_steps", show_value=True)
    sample_seed = mo.ui.number(start=0, value=42, label="seed")
    mo.hstack([goal_offset, sample_seed])
    return goal_offset, sample_seed


@app.cell
def _(goal_offset, lengths, np, offsets, sample_seed):
    # Same logic as eval.py: every (episode, step) with step <= length - offset - 1 is a valid start.
    _max_start = lengths - goal_offset.value - 1
    _row_episode = np.repeat(np.arange(len(lengths)), lengths)
    _row_step = np.arange(lengths.sum()) - np.repeat(offsets, lengths)
    valid_rows = np.nonzero(_row_step <= _max_start[_row_episode])[0]
    _g = np.random.default_rng(int(sample_seed.value))
    _pick = np.sort(valid_rows[_g.choice(len(valid_rows), size=4, replace=False)])
    eval_problems = list(zip(_row_episode[_pick].tolist(), _row_step[_pick].tolist()))
    return eval_problems, valid_rows


@app.cell
def _(ds, eval_problems, goal_offset, np, plt, valid_rows):
    _fig, _axes = plt.subplots(2, 4, figsize=(13, 7))
    for _i, (_e, _t) in enumerate(eval_problems):
        _chunk = ds.load_chunk(np.array([_e]), np.array([_t]), np.array([_t + goal_offset.value + 1]))[0]
        _axes[0, _i].imshow(_chunk["pixels"][0].permute(1, 2, 0).numpy())
        _axes[0, _i].set_title(f"ep {_e}, t={_t} (start)", fontsize=9)
        _axes[1, _i].imshow(_chunk["pixels"][-1].permute(1, 2, 0).numpy())
        _s0, _s1 = _chunk["state"][0].numpy(), _chunk["state"][-1].numpy()
        _axes[1, _i].set_title(
            f"t={_t + goal_offset.value} (goal)\nblock moved {np.linalg.norm(_s1[2:4] - _s0[2:4]):.0f} px", fontsize=9
        )
    for _ax in _axes.flat:
        _ax.axis("off")
    _fig.suptitle(f"{len(valid_rows):,} valid start rows for goal_offset={goal_offset.value}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    注意 `load_chunk(episodes, starts, ends)` 这个接口：一次读好几段，每段是 `[start, end)`
    的连续步——`World.evaluate(dataset=...)` 内部就是用它取出起点和目标的。

    25 步里方块通常只移动几十像素、转几十度，比 full-solve（从随机起点一路推到绿 T，
    通常要 100 多步）容易得多。这也是为什么本仓库有两个评估模式。

    ## 小结

    - 约 1.9 万条专家 episode，中位长度约 120 步；
    - 大约 2/3 的 episode 以方块在绿 T 上结束，agent 常停在 (153, 356) 附近；
    - dataset 模式：以专家第 t 帧为起点、第 t+25 帧为目标，50 步预算；
    - full-solve 模式：随机起点，目标是绿 T，300 步预算。

    下一篇：加载 LeWM，看它怎么"想象"未来。
    """)
    return


if __name__ == "__main__":
    app.run()
