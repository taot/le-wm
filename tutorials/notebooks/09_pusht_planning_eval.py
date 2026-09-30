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

    import eval as ev  # noqa: E402
    import tempfile
    import time
    from typing import Any

    import hydra
    import marimo as mo
    import numpy as np
    import stable_worldmodel as swm
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf

    return (
        Any,
        DictConfig,
        OmegaConf,
        Path,
        ROOT,
        compose,
        ev,
        hydra,
        initialize_config_dir,
        mo,
        np,
        swm,
        tempfile,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 09 · 用 CEM 规划并评估 PushT

    把前面的东西拼起来，就是 `eval.py` 做的事：

    ```text
    World(PushT)  ──infos──►  WorldModelPolicy  ──►  CEMSolver  ──►  JEPA.get_cost
         ▲                         │ (process: 标准化, transform: 图像预处理)
         └──────── 动作 ◄──────────┘ (每次规划 5 个模型步 = 25 个环境步，全部执行后再规划)
    ```

    这篇用和 `eval.py` **相同的配置文件**（`config/eval/pusht.yaml`，通过 hydra 读入），
    只是把几个会影响速度的参数放到下面的表单里，方便在笔记本（CPU / 小 GPU）和 RunPod
    上都能跑。默认值：有 CUDA 时和论文评估一样（300 个样本 × 30 轮），只有 CPU 时缩小很多
    （结果会差一些，但几分钟内能跑完）。改完点"提交"才会重新跑。
    """)
    return


@app.cell
def _(torch):
    # Checkpoint path is relative to $STABLEWM_HOME/checkpoints/ (default ~/.stable_worldmodel/checkpoints/).
    CHECKPOINT = "pusht/2026-09-28_114752_img112_s3072/weights_epoch_003.pt"
    IMG_SIZE = 112
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEFAULTS = (
        {"num_samples": 300, "n_steps": 30, "num_eval": 4, "seed": 42}
        if DEVICE == "cuda"
        else {"num_samples": 64, "n_steps": 10, "num_eval": 2, "seed": 42}
    )
    return CHECKPOINT, DEFAULTS, DEVICE, IMG_SIZE


@app.cell
def _(DEFAULTS, DEVICE, mo):
    settings_form = (
        mo.md(
            f"""
    **规划设置**（设备：`{DEVICE}`）

    {{num_samples}} {{n_steps}}

    {{num_eval}} {{seed}}
    """
        )
        .batch(
            num_samples=mo.ui.slider(16, 600, step=16, value=DEFAULTS["num_samples"], label="CEM num_samples", show_value=True),
            n_steps=mo.ui.slider(2, 40, value=DEFAULTS["n_steps"], label="CEM n_steps（轮数）", show_value=True),
            num_eval=mo.ui.slider(1, 16, value=DEFAULTS["num_eval"], label="评估题目数", show_value=True),
            seed=mo.ui.number(start=0, value=DEFAULTS["seed"], label="seed"),
        )
        .form(submit_button_label="提交并运行")
    )
    settings_form
    return (settings_form,)


@app.cell
def _(
    CHECKPOINT,
    DEFAULTS,
    DEVICE,
    IMG_SIZE,
    ROOT,
    compose,
    initialize_config_dir,
    settings_form,
):
    settings = settings_form.value or DEFAULTS
    # Load config/eval/pusht.yaml exactly as eval.py would see it with these overrides.
    _overrides = [
        f"policy={CHECKPOINT}",
        f"eval.img_size={IMG_SIZE}",
        f"eval.num_eval={settings['num_eval']}",
        f"seed={settings['seed']}",
        f"solver.num_samples={settings['num_samples']}",
        f"solver.n_steps={settings['n_steps']}",
        f"solver.device={DEVICE}",
    ]
    with initialize_config_dir(config_dir=str(ROOT / "config/eval"), version_base=None):
        cfg = compose("pusht", _overrides)
    return cfg, settings


@app.cell
def _(OmegaConf, cfg, mo):
    mo.accordion({"合成后的配置（和 eval.py 看到的一样）": mo.md(f"```yaml\n{OmegaConf.to_yaml(cfg)}\n```")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 组装 policy

    和 `eval.py:build_policy` 一步一步对应（只把 `"cuda"` 换成了自动选择的设备）：

    1. **process**：`eval.fit_process` 在专家数据上给 `action`、`proprio`、`state`（以及
       `goal_proprio`、`goal_state`）各拟合一个 `StandardScaler`。policy 用它把环境的原始数值
       标准化后交给模型，并把 solver 输出的动作**反标准化**回环境单位；
    2. **transform**：图像预处理（ImageNet 标准化 + 缩放到 112），同时作用于 `pixels` 和 `goal`；
    3. **solver**：`hydra.utils.instantiate(cfg.solver, model=model)`，即
       `CEMSolver(model, num_samples, n_steps, topk=30, var_scale=1, ...)`；
    4. **PlanConfig**：`horizon=5, receding_horizon=5, action_block=5`——每次规划 25 个环境步，
       全部执行完再重新规划（05 篇）。
    """)
    return


@app.cell
def _(cfg, ev):
    # Expert dataset: fits the scalers here, and is also the source of the eval problems below.
    dataset = ev.get_dataset(cfg, cfg.eval.dataset_name)
    process = ev.fit_process(cfg, dataset)
    return dataset, process


@app.cell
def _(Any, DEVICE, DictConfig, cfg, ev, hydra, swm):
    model = swm.wm.utils.load_pretrained(cfg.policy).to(DEVICE).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True

    def build_policy(cfg: DictConfig, process: dict[str, Any]) -> swm.policy.WorldModelPolicy:
        """eval.build_policy, but reusing the loaded model on DEVICE."""
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        return swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(**cfg.plan_config),
            process=process,
            transform={"pixels": ev.img_transform(cfg), "goal": ev.img_transform(cfg)},
        )

    return (build_policy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. dataset 模式评估

    按 `eval.py` 的方法从专家数据里抽 `num_eval` 道题（07 篇第 3 节）：起点是专家第 t 步，
    目标是第 t+25 步，预算 50 步，成功规则是环境自带的（agent + 方块都要到位）。

    `world.evaluate(dataset=..., callables=...)` 会：
    - 用 `callables`（配置里的 `_set_state` / `_set_goal_state`）把每个环境摆到起点、设好目标状态；
    - 把 `info["goal"]` 换成数据集里第 t+25 帧的图像；
    - 跑 policy 直到成功或用完 50 步，并为每道题存一个视频（agent / 专家 / 目标 三栏）。

    同样的题目也给 `RandomPolicy` 做一遍当对照。
    """)
    return


@app.cell
def _(Any, cfg, dataset, np):
    def sample_problems(cfg: Any, dataset: Any) -> tuple[list[int], list[int]]:
        """eval.py's sampling of (episode, start step) pairs, written as a function."""
        lengths = np.asarray(dataset.lengths)
        offsets = np.asarray(dataset.offsets)
        max_start = lengths - cfg.eval.goal_offset_steps - 1
        row_episode = np.repeat(np.arange(len(lengths)), lengths)
        row_step = np.arange(lengths.sum()) - np.repeat(offsets - offsets[0], lengths)
        valid = np.nonzero(row_step <= max_start[row_episode])[0]
        g = np.random.default_rng(cfg.seed)
        picked = np.sort(valid[g.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)])
        return row_episode[picked].tolist(), row_step[picked].tolist()

    eval_episodes, eval_starts = sample_problems(cfg, dataset)
    return eval_episodes, eval_starts


@app.cell
def _(
    OmegaConf,
    Path,
    build_policy,
    cfg,
    dataset,
    eval_episodes,
    eval_starts,
    process,
    swm,
    tempfile,
    time,
):
    def run_dataset_eval(policy: object, video_dir: Path | None) -> tuple[dict, float]:
        world = swm.World(
            cfg.world.env_name,
            num_envs=cfg.eval.num_eval,
            max_episode_steps=2 * cfg.eval.eval_budget,
            image_shape=(224, 224),
        )
        world.set_policy(policy)
        t0 = time.perf_counter()
        results = world.evaluate(
            dataset=dataset,
            episodes_idx=eval_episodes,
            start_steps=eval_starts,
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
            video=video_dir,
        )
        world.close()
        return results, time.perf_counter() - t0

    video_dir = Path(tempfile.mkdtemp(prefix="swm_tutorial_eval_"))
    wm_results, wm_secs = run_dataset_eval(build_policy(cfg, process), video_dir)
    random_results, _ = run_dataset_eval(swm.policy.RandomPolicy(seed=cfg.seed), None)
    return random_results, video_dir, wm_results, wm_secs


@app.cell
def _(
    eval_episodes,
    eval_starts,
    mo,
    random_results,
    settings,
    video_dir,
    wm_results,
    wm_secs,
):
    _rows = ["| 题目 | episode | 起点 t | LeWM + CEM | 随机 |", "|---|---|---|---|---|"]
    for _i, (_e, _t) in enumerate(zip(eval_episodes, eval_starts)):
        _rows.append(
            f"| {_i} | {_e} | {_t} | {'✅' if wm_results['episode_successes'][_i] else '❌'} "
            f"| {'✅' if random_results['episode_successes'][_i] else '❌'} |"
        )
    _videos = sorted(video_dir.glob("*.mp4"))
    mo.vstack(
        [
            mo.md(
                f"**LeWM + CEM 成功率 {wm_results['success_rate']:.0f}%**，随机 policy "
                f"{random_results['success_rate']:.0f}%（{len(eval_episodes)} 道题，"
                f"CEM {settings['num_samples']} 样本 × {settings['n_steps']} 轮，用时 {wm_secs:.0f} 秒）\n\n"
                + "\n".join(_rows)
            ),
            mo.md("每道题的视频：左 = agent 实际执行，中 = 专家原本的 25 步，右 = 目标图像"),
            *[mo.vstack([mo.md(f"`{p.name}`"), mo.video(src=p.read_bytes(), width=520, loop=True)]) for p in _videos],
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    题目数少时成功率波动很大；要得到可以对比的数字，按 `eval.py` 的设置跑 50 道题
    （`python eval.py --config-name pusht policy=...`，最好在 GPU 上）。可以试着调小
    `num_samples` / `n_steps`，看成功率怎么掉——这就是 CEM 的"算力 ↔ 质量"权衡。

    ## 3. full-solve 模式（可选，较慢）

    full-solve（`config/eval/pusht_full.yaml`）从随机起点出发，目标是绿 T，预算 300 步，
    成功规则只看方块（`eval.BlockOnGoalSuccess`，06 篇）。它难得多：目标常常离得很远，
    而 08 篇看到嵌入距离只在接近目标时才有明显区分。

    一条 episode 在 CPU 上要跑很久，所以默认不运行，打开下面的开关才跑
    （一个环境，`eval.evaluate_full_solve`）。想一步一步看它的内部过程，见仓库里的
    `notebooks/one_run_full_solve.py`。
    """)
    return


@app.cell
def _(mo):
    full_switch = mo.ui.switch(label="运行一条 full-solve episode")
    full_budget = mo.ui.slider(25, 300, step=25, value=150, label="eval_budget（环境步）", show_value=True)
    mo.vstack([full_switch, full_budget])
    return full_budget, full_switch


@app.cell
def _(
    CHECKPOINT,
    DEVICE,
    IMG_SIZE,
    Path,
    ROOT,
    build_policy,
    compose,
    ev,
    full_budget,
    full_switch,
    initialize_config_dir,
    mo,
    np,
    process,
    settings,
    swm,
    tempfile,
    time,
):
    mo.stop(not full_switch.value, mo.md("（开关关闭，未运行）"))

    with initialize_config_dir(config_dir=str(ROOT / "config/eval"), version_base=None):
        _full_cfg = compose(
            "pusht_full",
            [
                f"policy={CHECKPOINT}",
                f"eval.img_size={IMG_SIZE}",
                "eval.num_eval=1",
                f"eval.eval_budget={full_budget.value}",
                f"seed={settings['seed']}",
                f"solver.num_samples={settings['num_samples']}",
                f"solver.n_steps={settings['n_steps']}",
                f"solver.device={DEVICE}",
            ],
        )
    _world = swm.World(
        _full_cfg.world.env_name,
        num_envs=1,
        max_episode_steps=_full_cfg.eval.eval_budget,
        image_shape=(224, 224),
        extra_wrappers=[ev.BlockOnGoalSuccess],
    )
    _world.set_policy(build_policy(_full_cfg, process))
    _dir = Path(tempfile.mkdtemp(prefix="swm_tutorial_full_"))
    _t0 = time.perf_counter()
    _metrics = ev.evaluate_full_solve(_full_cfg, _world, _dir)
    _secs = time.perf_counter() - _t0
    _err = ev.block_goal_error(_world.envs.envs[0].unwrapped)
    _world.close()
    _video = sorted(_dir.glob("episode_*.mp4"))
    mo.vstack(
        [
            mo.md(
                f"成功：**{bool(_metrics['episode_successes'][0])}**，用时 {_secs:.0f} 秒。"
                f"结束时方块离绿 T：{_err[0]:.1f} px、{np.degrees(_err[1]):.1f}°。"
            ),
            mo.hstack(
                [
                    mo.image(src=(_dir / "goal.png").read_bytes(), width=224, caption="goal image"),
                    *[mo.video(src=p.read_bytes(), width=224, loop=True) for p in _video],
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 整套教程回顾

    | 篇 | 学到了什么 |
    |---|---|
    | 01 | swm 的模块和数据流：env → `info` → policy → solver → `model.get_cost` |
    | 02 | 环境 = gym + `info` + `variation_space`；`reset(options=...)` |
    | 03 | `World`：并行环境、`infos` 形状 `(N, 1, ...)`、自定义 policy、`evaluate`、wrapper |
    | 04 | `collect` / `ReplayBuffer` / `load_dataset`；片段、frameskip、归一化 |
    | 05 | `Costable` / `Solver` 接口；CEM；MPC 和 `PlanConfig` |
    | 06 | PushT 的状态、相对动作、两个目标、两种成功规则 |
    | 07 | 专家数据；dataset 模式的题目怎么来 |
    | 08 | LeWM：编码、嵌入空间里的预测、代价 |
    | 09 | 组装 `WorldModelPolicy`，跑和 `eval.py` 一样的评估 |

    接下来可以读：`jepa.py`（模型）、`train.py`（训练）、`eval.py`（评估），以及
    `notebooks/` 里更深入的单条 episode 分析。
    """)
    return


if __name__ == "__main__":
    app.run()
