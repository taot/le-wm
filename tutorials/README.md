# stable-worldmodel 教程

一套 marimo notebook，用来学习 [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)（本仓库用它做环境、数据、规划和评估）。
前半部分讲包的通用概念，后半部分聚焦 PushT 任务和本仓库的 LeWM 模型。

| notebook | 内容 | 需要 |
|---|---|---|
| [01_overview.py](notebooks/01_overview.py) | 包的全貌：模块、数据流、十行代码跑一遍 | — |
| [02_envs_and_variation_spaces.py](notebooks/02_envs_and_variation_spaces.py) | 环境接口、`info` 字典、variation space、reset 选项 | — |
| [03_world_and_policies.py](notebooks/03_world_and_policies.py) | `World`、`infos` 的形状、自定义 policy、`evaluate`、视觉 wrapper | — |
| [04_data.py](notebooks/04_data.py) | `collect` / `ReplayBuffer` / `load_dataset`、片段与 frameskip、归一化 | 数据集 |
| [05_solvers_and_planning.py](notebooks/05_solvers_and_planning.py) | `Costable` / `Solver` 接口、CEM 可视化、各 solver 比较、MPC 与 `PlanConfig` | — |
| [06_pusht_env.py](notebooks/06_pusht_env.py) | PushT 的状态、相对动作与 PD 控制、`goal_pose` vs `goal_state`、两种成功规则 | — |
| [07_pusht_dataset.py](notebooks/07_pusht_dataset.py) | 专家数据统计、episode 浏览、dataset 模式评估题目怎么出 | 数据集 |
| [08_pusht_world_model.py](notebooks/08_pusht_world_model.py) | LeWM：编码、嵌入空间里的预测、代价 | 数据集 + checkpoint |
| [09_pusht_planning_eval.py](notebooks/09_pusht_planning_eval.py) | 组装 `WorldModelPolicy`，跑和 `eval.py` 一样的评估（可选 full-solve） | 数据集 + checkpoint |

- **数据集**：`librakevin/lewm-pusht`，第一次用时自动下载到 `$STABLEWM_HOME/datasets/`（默认 `~/.stable_worldmodel`）。
- **checkpoint**：08、09 开头的 `CHECKPOINT` 常量，路径相对 `$STABLEWM_HOME/checkpoints/`；换成你自己的 run 时，`IMG_SIZE` 也要和训练时一致。
- 08、09 有 CUDA 就用 GPU，否则用 CPU（09 在 CPU 上会自动用小一些的 CEM 设置）。

## Slides

[slides/](slides/) 里有一套讲 world model 的 slides（Slidev）：定义、核心难题、技术路线、评测、可解释性，最后落到本仓库的 LeWM。运行方法见 [slides/README.md](slides/README.md)。

## 运行

在本机：

```bash
marimo edit tutorials/notebooks/01_overview.py
```

在 RunPod 上（浏览器在本机打开）：

```bash
ssh -L 2718:localhost:2718 runpod
cd /workspace/le-wm && marimo edit --headless --port 2718 tutorials/notebooks/
```

然后打开终端里打印的 `http://localhost:2718?access_token=...` 链接。
