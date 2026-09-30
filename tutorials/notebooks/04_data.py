import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.envs.pusht import WeakPolicy

    return Path, StandardScaler, WeakPolicy, mo, np, plt, swm, tempfile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 04 · 数据集

    世界模型是从**轨迹数据**里学的：一条条 episode，每一步有图像、状态、动作。swm 的
    数据部分解决三个问题：

    1. **收集**：`world.collect(...)` 把 rollout 直接写成数据集；
    2. **存储格式**：lance（本仓库用的）、hdf5、folder、video、lerobot，统一接口；
    3. **读取**：`swm.data.load_dataset(name)` 返回一个数据集对象，可以按 episode 读、
       按"片段（clip）"读，也可以一次拿一整列来算统计量。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 收集到内存：`ReplayBuffer`

    `world.collect(writer=..., episodes=N)` 用当前 policy 跑 N 个 episode，把 `world.infos`
    的每个键都当作一列写进去。`swm.data.ReplayBuffer` 是一个**内存里**的 writer，适合
    小实验和在线训练，不用写盘。这里用 PushT 的 `WeakPolicy`（它就是用来收集数据的）。
    """)
    return


@app.cell
def _(WeakPolicy, swm):
    world = swm.World("swm/PushT-v1", num_envs=4, image_shape=(64, 64), max_episode_steps=30)
    world.set_policy(WeakPolicy(seed=0))
    buffer = swm.data.ReplayBuffer(max_steps=10_000)
    world.collect(writer=buffer, episodes=6, seed=0, progress=False)
    buffer
    return buffer, world


@app.cell
def _(buffer, mo):
    mo.md(
        f"""
    - `buffer.num_episodes` = {buffer.num_episodes}，`buffer.lengths` = `{buffer.lengths.tolist()}`
    - `buffer.offsets` = `{buffer.offsets.tolist()}`（每个 episode 在"扁平的步序列"里从哪一行开始）
    - 列：{", ".join(f"`{c}`" for c in buffer.column_names)}

    一个细节：`MegaWrapper` 在第 t 步记下的 `action` 是"**导致**第 t 步的那个动作"，
    而训练时我们要的是"在第 t 步**执行**的动作"。所以 `collect` 会把每个 episode 的
    `action` 列向前挪一格（第一个挪到最后）。之后读到的第 t 行就是：
    **看到 `pixels[t]` → 执行 `action[t]` → 得到 `pixels[t+1]`**。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 片段（clip）：`history_len` 和 `frameskip`

    训练世界模型时，一个样本不是一帧，而是**连续几帧**。`buffer.sample(batch_size, history_len)`
    随机抽片段，每一列的形状是 `(batch, history_len, ...)`：
    """)
    return


@app.cell
def _(buffer, mo):
    _batch = buffer.sample(batch_size=8, history_len=4)
    mo.md("\n".join(f"- `{k}`: `{v.shape}`" for k, v in _batch.items() if k in ("pixels", "action", "state", "proprio")))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 写到硬盘再读回来

    同一个 `collect` 传 `path=...` 就写成文件（默认 lance 格式，`format=` 可选其它），
    然后用 `swm.data.load_dataset` 读回来。这里写到一个临时目录。
    """)
    return


@app.cell
def _(Path, swm, tempfile, world):
    _tmp = Path(tempfile.mkdtemp(prefix="swm_tutorial_"))
    disk_path = _tmp / "demo.lance"
    world.collect(path=disk_path, episodes=4, seed=100, progress=False)
    small_ds = swm.data.load_dataset(str(disk_path))
    type(small_ds).__name__, small_ds.column_names, small_ds.lengths.tolist()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 真正的 PushT 专家数据集

    本仓库训练 / 评估都用 `librakevin/lewm-pusht`。`load_dataset` 的名字可以是：

    1. 本地路径；
    2. HuggingFace 仓库名 `<user>/<repo>`——第一次会下载到
       `$STABLEWM_HOME/datasets/<user>--<repo>/`，之后直接用缓存；
    3. 带协议前缀的名字，比如 `lerobot://lerobot/pusht`。

    `keys_to_cache` 里的列会一次性读进内存（小列：动作、状态），图像按需读取。
    读取要几秒钟。
    """)
    return


@app.cell
def _(swm):
    DATASET = "librakevin/lewm-pusht"
    ds = swm.data.load_dataset(DATASET, keys_to_cache=["action", "proprio", "state"])
    return DATASET, ds


@app.cell
def _(ds, mo, np):
    _lengths = np.asarray(ds.lengths)
    mo.md(
        f"""
    - 类型：`{type(ds).__name__}`，列：`{ds.column_names}`
    - episode 数：**{len(_lengths):,}**，总步数：**{_lengths.sum():,}**
    - episode 长度：最短 {_lengths.min()}，中位数 {int(np.median(_lengths))}，最长 {_lengths.max()}
    - `len(ds)` = {len(ds):,}（默认一个样本 = 1 步）
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 按 episode 读：`load_episode`

    返回一个字典，每列是 `torch.Tensor`，图像是 **`(T, 3, 224, 224)` uint8**
    （通道在前，和 `World` 里的 `(H, W, 3)` 不一样）。
    """)
    return


@app.cell
def _(mo):
    episode_idx = mo.ui.number(start=0, stop=18000, value=3, label="episode")
    episode_idx
    return (episode_idx,)


@app.cell
def _(ds, episode_idx, mo, np, plt):
    ep = ds.load_episode(int(episode_idx.value))
    _T = len(ep["pixels"])
    _idx = np.linspace(0, _T - 1, 6).astype(int)
    _fig, _axes = plt.subplots(1, 6, figsize=(15, 3))
    for _ax, _t in zip(_axes, _idx):
        _ax.imshow(ep["pixels"][_t].permute(1, 2, 0).numpy())
        _ax.set_title(f"t={_t}")
        _ax.axis("off")
    _fig.tight_layout()
    mo.vstack([mo.md("  \n".join(f"`{k}`: {tuple(v.shape)} {v.dtype}" for k, v in ep.items())), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 按片段读：训练时用的格式

    `load_dataset(..., num_steps=K, frameskip=F)` 让 `ds[i]` 返回一个 **K 帧、每帧间隔 F 步**
    的片段。本仓库训练 PushT 用 `num_steps=4`（3 帧历史 + 预测 1 帧）、`frameskip=5`
    （见 `config/train/data/pusht.yaml`）。

    注意 `action` 的形状：两帧之间隔了 5 个环境步，中间 5 个动作（每个 2 维）被拼成
    **一个 10 维向量**。所以世界模型的"一步"= 环境的 5 步，模型的动作维度 = 10
    （checkpoint 的 `action_encoder.input_dim: 10` 就是这么来的；规划时的
    `action_block: 5` 也是这个意思）。
    """)
    return


@app.cell
def _(DATASET, mo, swm):
    clip_ds = swm.data.load_dataset(DATASET, num_steps=4, frameskip=5, keys_to_cache=["action", "proprio", "state"])
    _clip = clip_ds[0]
    mo.md(
        f"`len(clip_ds)` = {len(clip_ds):,} 个合法片段起点\n\n"
        + "\n".join(f"- `{k}`: `{tuple(v.shape)}`" for k, v in _clip.items())
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 整列读取和归一化

    `ds.get_col_data(col)` 返回一整列（所有 episode 的所有步）的 numpy 数组。它主要
    用来算**归一化统计量**：动作、状态的数值范围差别很大（位置是 0–512 像素，动作是
    -1–1），模型吃的是**标准化**后的数值（减均值、除标准差）。

    - 训练时（`train.py`）：`swm.data.column_normalizer` 在数据上 fit 一个 z-score；
    - 评估时（`eval.py:fit_process`）：用 sklearn `StandardScaler` 在同一份数据上 fit，
      交给 `WorldModelPolicy(process=...)`，policy 会把环境给的原始数值先标准化再给模型，
      模型输出的动作再反标准化回环境单位。

    两边必须用**同一份数据**算统计量，否则模型看到的数值分布就对不上了。
    """)
    return


@app.cell
def _(StandardScaler, ds, mo, np):
    _rows = ["| 列 | 形状 | 均值 | 标准差 |", "|---|---|---|---|"]
    for _col in ["action", "proprio", "state"]:
        _data = ds.get_col_data(_col)
        _sc = StandardScaler().fit(_data[~np.isnan(_data).any(axis=1)])
        _rows.append(
            f"| `{_col}` | `{_data.shape}` | `{np.round(_sc.mean_, 2).tolist()}` | `{np.round(_sc.scale_, 2).tolist()}` |"
        )
    mo.md("\n".join(_rows))
    return


@app.cell
def _(ds, np, plt):
    _a = ds.get_col_data("action")
    _rng = np.random.default_rng(0)
    _sub = _a[_rng.choice(len(_a), 50_000, replace=False)]
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].hist2d(_sub[:, 0], _sub[:, 1], bins=80, cmap="viridis")
    _axes[0].set_title("expert actions (50k sample)")
    _axes[0].set_xlabel("action x")
    _axes[0].set_ylabel("action y")
    _axes[0].set_aspect("equal")
    _axes[1].hist(np.linalg.norm(_sub, axis=1), bins=100)
    _axes[1].set_title("|action|")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    专家动作大多很小：|action| 的中位数约 0.2（一步把目标点放在 20 像素外），约 70% 小于
    0.3，远小于动作空间允许的 [-1, 1]。这对规划很重要：CEM 在标准化空间里采样，自然就落在专家常用的
    范围附近。

    ## 小结

    - `world.collect(writer=ReplayBuffer)` 或 `collect(path=...)` 收集数据；
    - `load_dataset(name, num_steps, frameskip, keys_to_cache)` 读数据；
    - 片段里的 `action` 把 `frameskip` 个动作拼成一个向量；
    - 归一化统计量来自 `get_col_data`，训练和评估要一致。

    下一篇：solver 怎么用"代价函数"挑出动作。
    """)
    return


if __name__ == "__main__":
    app.run()
