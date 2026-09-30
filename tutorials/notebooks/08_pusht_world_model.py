import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # eval.py lives in the repo root and must be imported before torch:
    # it sets MUJOCO_GL and caps torch's CPU threads on import.
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))

    import eval as ev  # noqa: E402,F401  (sets env vars before torch loads)
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from sklearn.preprocessing import StandardScaler
    from torchvision.transforms import v2 as T

    return StandardScaler, T, mo, np, plt, spt, swm, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 08 · LeWM 世界模型：编码、预测、代价

    05 篇的玩具模型是手写的公式；这里换成真的：本仓库训练出的 **LeWM**（`jepa.py:JEPA`）。
    它不预测像素，而是在一个 192 维的**嵌入空间（latent）**里预测：

    ```text
    图像 ──encoder(ViT-tiny)──► 嵌入 z_t        (192 维)
    (z_{t-2..t}, 动作 a_{t-2..t}) ──predictor──► ẑ_{t+1}
    代价 = ‖ ẑ_最后一步 − encoder(目标图像) ‖²
    ```

    这一篇逐个看这三件事。需要一个 checkpoint；有 CUDA 就用 GPU，没有就用 CPU（慢一些，
    但这篇的计算量很小）。
    """)
    return


@app.cell
def _(torch):
    # Checkpoint path is relative to $STABLEWM_HOME/checkpoints/ (default ~/.stable_worldmodel/checkpoints/).
    # IMG_SIZE must match the image size the checkpoint was trained with (112 for an img112 run).
    CHECKPOINT = "pusht/2026-09-28_114752_img112_s3072/weights_epoch_003.pt"
    IMG_SIZE = 112
    FRAMESKIP = 5  # env steps per model step (action_block / dataset frameskip)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return CHECKPOINT, DEVICE, FRAMESKIP, IMG_SIZE


@app.cell
def _(CHECKPOINT, DEVICE, mo, swm):
    model = swm.wm.utils.load_pretrained(CHECKPOINT).to(DEVICE).eval().requires_grad_(False)
    _rows = ["| 部件 | 类 | 参数量 |", "|---|---|---|"]
    for _name, _mod in model.named_children():
        _rows.append(f"| `{_name}` | `{type(_mod).__name__}` | {sum(p.numel() for p in _mod.parameters()) / 1e6:.2f} M |")
    mo.md(
        f"`swm.wm.utils.load_pretrained` 读 `config.json`（用 hydra 实例化模型）+ `.pt` 权重。"
        f"设备：**{DEVICE}**，总参数 {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M。\n\n" + "\n".join(_rows)
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 数据和预处理

    模型训练时见到的图像经过了：转 float、按 ImageNet 均值方差标准化、缩放到 112×112
    （和 `eval.py:img_transform` 一样）。动作按数据集统计量做 z-score（`eval.py:fit_process`），
    然后每 5 个环境动作拼成一个 10 维的模型动作（04 篇）。
    """)
    return


@app.cell
def _(IMG_SIZE, StandardScaler, T, spt, swm, torch):
    ds = swm.data.load_dataset("librakevin/lewm-pusht", keys_to_cache=["action"])
    action_scaler = StandardScaler().fit(ds.get_col_data("action"))

    img_transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**spt.data.dataset_stats.ImageNet),
            T.Resize(size=IMG_SIZE),
        ]
    )

    def prep_pixels(pixels: torch.Tensor) -> torch.Tensor:
        """uint8 (N, 3, 224, 224) -> normalized float (N, 3, IMG_SIZE, IMG_SIZE)."""
        return torch.stack([img_transform(x) for x in pixels])

    return action_scaler, ds, prep_pixels


@app.cell
def _(ds, mo):
    ep_input = mo.ui.number(start=0, stop=len(ds.lengths) - 1, value=3, label="episode")
    ep_input
    return (ep_input,)


@app.cell
def _(DEVICE, ds, ep_input, model, prep_pixels, torch):
    episode = ds.load_episode(int(ep_input.value))
    ep_pixels = prep_pixels(episode["pixels"])  # (L, 3, 112, 112)
    with torch.inference_mode():
        # encode() takes {"pixels": (B, T, C, H, W)} and adds "emb": (B, T, 192)
        ep_emb = model.encode({"pixels": ep_pixels[None].to(DEVICE)})["emb"][0].cpu()
    return ep_emb, ep_pixels, episode


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 编码：一帧 → 一个 192 维向量

    `model.encode` 把整条 episode 的每一帧编码成一个向量。LeWM 的训练目标之一
    （SIGReg）让这些向量大致服从**标准正态分布**，所以各维的数值大约在 ±3 之间。

    左图：每一帧的嵌入到**最后一帧**嵌入的距离。中图：把整条 episode 的嵌入用 PCA 投到
    2 维，颜色表示时间。右图：所有嵌入数值的分布。

    看左图要注意一件事：距离**不是**随着进度平稳下降的。离终点很远时，距离基本在一个平台上
    波动；只有最后十来步（离终点几十像素以内）才迅速降到 0。也就是说，嵌入距离是一个
    **局部**的"像不像"量度，而不是"还要推多少步"。这对规划很关键：目标离得近（dataset
    模式的 25 步）时，代价能很好地区分好坏动作；目标很远（full-solve）时，大部分候选的
    代价都差不多，CEM 缺少方向感。
    """)
    return


@app.cell
def _(ep_emb, np, plt):
    _d = np.linalg.norm(ep_emb.numpy() - ep_emb[-1].numpy(), axis=1)
    _x = ep_emb.numpy() - ep_emb.numpy().mean(0)
    _u, _s, _vt = np.linalg.svd(_x, full_matrices=False)
    _pc = _x @ _vt[:2].T
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
    _axes[0].plot(_d)
    _axes[0].set_xlabel("step")
    _axes[0].set_title("‖z_t − z_last‖")
    _sc = _axes[1].scatter(_pc[:, 0], _pc[:, 1], c=np.arange(len(_pc)), cmap="viridis", s=12)
    _axes[1].plot(_pc[:, 0], _pc[:, 1], color="0.7", lw=0.5, zorder=0)
    _fig.colorbar(_sc, ax=_axes[1], label="step")
    _axes[1].set_title("episode embeddings, PCA")
    _axes[2].hist(ep_emb.numpy().ravel(), bins=80, density=True)
    _g = np.linspace(-4, 4, 100)
    _axes[2].plot(_g, np.exp(-_g**2 / 2) / np.sqrt(2 * np.pi), "r", label="N(0, 1)")
    _axes[2].legend()
    _axes[2].set_title("all embedding values")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 预测：在嵌入空间里"想象"未来

    `model.rollout(info, action_sequence)` 做规划时用的预测：

    - `info["pixels"]`：起始图像，形状 `(B, S, 1, C, H, W)`（B 个环境、S 个候选，1 帧历史）；
    - `action_sequence`：`(B, S, 5, 10)`——5 个模型步，每步 10 维（= 5 个环境动作）；
    - 返回 `info["predicted_emb"]`：`(B, S, 6, 192)`，第 0 个是起始帧的编码，后面 5 个是
      预测器一步步**自回归**预测出来的（每一步把自己的预测当作下一步的输入，最多看 3 步历史）。

    下面用专家在这条 episode 里**真实执行**的动作去 rollout，然后和真实未来帧的编码比较。
    """)
    return


@app.cell
def _(episode, mo):
    start_slider = mo.ui.slider(0, len(episode["action"]) - 26, value=20, label="起始步 t", show_value=True)
    start_slider
    return (start_slider,)


@app.cell
def _(FRAMESKIP, action_scaler, np, torch):
    HORIZON = 5  # model steps; 5 x 5 = 25 env steps, as in config/eval/pusht.yaml

    def to_model_actions(env_actions: np.ndarray) -> torch.Tensor:
        """Raw env actions (..., HORIZON * FRAMESKIP, 2) -> normalized model actions (..., HORIZON, 10)."""
        a = np.asarray(env_actions, dtype=np.float64)
        flat = action_scaler.transform(a.reshape(-1, 2)).reshape(a.shape)
        return torch.as_tensor(flat.reshape(*a.shape[:-2], HORIZON, FRAMESKIP * 2), dtype=torch.float32)

    return HORIZON, to_model_actions


@app.cell
def _(
    DEVICE,
    FRAMESKIP,
    HORIZON,
    ep_emb,
    ep_pixels,
    episode,
    model,
    start_slider,
    to_model_actions,
    torch,
):
    t0 = start_slider.value
    expert_env_actions = episode["action"][t0 : t0 + HORIZON * FRAMESKIP].numpy()  # (25, 2)
    _cand = to_model_actions(expert_env_actions)[None, None].to(DEVICE)  # (B=1, S=1, 5, 10)
    with torch.inference_mode():
        _info = {"pixels": ep_pixels[t0][None, None, None].to(DEVICE)}  # (1, 1, 1, C, H, W)
        pred_emb = model.rollout(_info, _cand)["predicted_emb"][0, 0].cpu()  # (6, 192)
    true_emb = ep_emb[t0 : t0 + HORIZON * FRAMESKIP + 1 : FRAMESKIP]  # frames t0, t0+5, ..., t0+25
    return expert_env_actions, pred_emb, t0, true_emb


@app.cell
def _(FRAMESKIP, episode, np, plt, pred_emb, t0, true_emb):
    _err_pred = ((pred_emb - true_emb) ** 2).sum(-1).numpy()
    _err_copy = ((true_emb[0] - true_emb) ** 2).sum(-1).numpy()  # baseline: "nothing changes"
    _fig = plt.figure(figsize=(15, 4.2))
    _a = _fig.add_subplot(1, 3, 1)
    _k = np.arange(len(_err_pred)) * FRAMESKIP
    _a.plot(_k, _err_pred, "o-", label="LeWM prediction")
    _a.plot(_k, _err_copy, "s--", label="baseline: copy start embedding")
    _a.set_xlabel("env steps after t")
    _a.set_ylabel("squared error to true embedding")
    _a.legend()
    for _j, _dt in enumerate([0, 25]):
        _a = _fig.add_subplot(1, 3, 2 + _j)
        _a.imshow(episode["pixels"][t0 + _dt].permute(1, 2, 0).numpy())
        _a.set_title(f"true frame t+{_dt}")
        _a.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    基线"假设什么都不变"的误差随时间越来越大（因为画面在变）；LeWM 的预测误差应该
    明显更小——说明它确实"想象"出了动作带来的变化。误差也会随步数累积，这就是 05 篇讲的
    MPC 要不断重新规划的原因。

    ## 4. 代价：给候选动作打分

    `model.get_cost(info, candidates)` 就是 solver 调用的那个接口（05 篇的 `Costable`）：

    1. 把 `info["goal"]`（目标图像）编码成 `z_goal`；
    2. 对每个候选做上面的 rollout；
    3. 代价 = `‖ẑ_最后一步 − z_goal‖²`（`jepa.py:criterion`）。

    取第 `t + 25` 帧作为目标，比较三类动作序列的代价：专家真实动作、专家动作加噪声、
    完全随机动作。好的世界模型应该给专家动作最低的代价——CEM 正是靠这个信号
    "找回"专家那样的动作。
    """)
    return


@app.cell
def _(mo):
    noise_slider = mo.ui.slider(0.0, 0.5, step=0.05, value=0.15, label="加在专家动作上的噪声（环境单位）", show_value=True)
    noise_slider
    return (noise_slider,)


@app.cell
def _(
    DEVICE,
    FRAMESKIP,
    HORIZON,
    ep_pixels,
    expert_env_actions,
    model,
    noise_slider,
    np,
    t0,
    to_model_actions,
    torch,
):
    N_EACH = 63
    _rng = np.random.default_rng(0)
    _noisy = np.clip(expert_env_actions + _rng.normal(0, noise_slider.value, (N_EACH, *expert_env_actions.shape)), -1, 1)
    _random = _rng.uniform(-1, 1, (N_EACH, *expert_env_actions.shape))
    _all = np.concatenate([expert_env_actions[None], _noisy, _random])  # (S, 25, 2)
    _cand = to_model_actions(_all)[None].to(DEVICE)  # (1, S, 5, 10)
    _S = _cand.shape[1]

    def _expand(x: torch.Tensor) -> torch.Tensor:
        return x[None, None, None].expand(1, _S, 1, *x.shape).to(DEVICE)

    with torch.inference_mode():
        _info = {
            "pixels": _expand(ep_pixels[t0]),
            "goal": _expand(ep_pixels[t0 + HORIZON * FRAMESKIP]),
            "action": torch.zeros(1, _S, 1, FRAMESKIP * 2, device=DEVICE),  # get_cost expects an "action" key
        }
        costs = model.get_cost(_info, _cand)[0].cpu().numpy()
    cost_groups = {"expert": costs[:1], "expert + noise": costs[1 : 1 + N_EACH], "random": costs[1 + N_EACH :]}
    return (cost_groups,)


@app.cell
def _(cost_groups, mo, np, plt):
    _fig, _ax = plt.subplots(figsize=(9, 3.8))
    _bins = np.logspace(np.log10(max(1e-2, min(v.min() for v in cost_groups.values()))), np.log10(max(v.max() for v in cost_groups.values())), 40)
    _ax.hist(cost_groups["expert + noise"], bins=_bins, alpha=0.6, label="expert + noise")
    _ax.hist(cost_groups["random"], bins=_bins, alpha=0.6, label="random")
    _ax.axvline(cost_groups["expert"][0], color="k", lw=2, label="expert")
    _ax.set_xscale("log")
    _ax.set_xlabel("cost = ‖predicted final emb − goal emb‖²")
    _ax.legend()
    _fig.tight_layout()
    _rank = int((np.concatenate(list(cost_groups.values())) < cost_groups["expert"][0]).sum())
    mo.vstack(
        [
            mo.md(
                f"专家动作的代价 **{cost_groups['expert'][0]:.2f}**，加噪声的中位数 "
                f"{np.median(cost_groups['expert + noise']):.2f}，随机的中位数 {np.median(cost_groups['random']):.2f}。"
                f"在全部 {sum(len(v) for v in cost_groups.values())} 个候选里，有 {_rank} 个比专家代价更低。"
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    噪声越大，代价分布越往右移，接近随机动作。偶尔会有加噪声的候选比专家代价还低——
    专家动作不一定是唯一的、也不一定是最短的路径，模型也有误差。

    ## 小结

    - `encode`：图像 → 192 维嵌入；嵌入距离只在离目标很近时才明显变小（局部量度）；
    - `rollout`：从一帧 + 一串动作，自回归预测未来嵌入（每个模型步 = 5 个环境步）；
    - `get_cost`：预测终点嵌入和目标图像嵌入的距离；专家动作代价低、随机动作代价高。

    下一篇：把它交给 CEM，在真实环境里闭环规划。
    """)
    return


if __name__ == "__main__":
    app.run()
