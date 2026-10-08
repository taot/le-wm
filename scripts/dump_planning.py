"""Record one real CEM planning episode for the planning visualizer.

Runs a single Push-T eval episode (same pipeline as eval.py) with a trained
LeWM checkpoint, hooks the solver and the model, and writes a compact JS data
file that docs/viz/lewm_planning.html loads with a <script> tag.

Run on the GPU pod, from the repo root:
    python scripts/dump_planning.py \
        policy=pusht/<run>/weights_epoch_003.pt eval.img_size=112

Extra keys (Hydra overrides, all optional):
    ++dump.out=docs/viz/lewm_planning_data.js
    ++dump.tries=12      episodes to try until one succeeds
    ++dump.start=0       first random-draw offset (change to get another episode)
"""

import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval as lewm_eval  # noqa: E402  (sets MUJOCO_GL / thread env vars first)

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf, open_dict  # noqa: E402
from PIL import Image  # noqa: E402
import stable_worldmodel as swm  # noqa: E402


# ---------------------------------------------------------------- recording


class Recorder:
    """Collects raw data from the policy, the solver and the model."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []  # one entry per env step
        self.plans: list[dict[str, Any]] = []  # one entry per solver call
        self.goal_pixels: np.ndarray | None = None
        self.goal_state: np.ndarray | None = None

    def hook(self, policy: swm.policy.WorldModelPolicy) -> None:
        model = policy.solver.model
        solver = policy.solver
        rec = self

        orig_get_cost = model.get_cost
        orig_solve = solver.solve
        orig_get_action = policy.get_action

        def get_cost(info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
            cost = orig_get_cost(info_dict, action_candidates)
            plan = rec.plans[-1]
            if "z0" not in plan:
                dim = info_dict["emb"].shape[-1]
                plan["z0"] = info_dict["emb"].reshape(-1, dim)[0].float().cpu().numpy()
                plan["goal_emb"] = info_dict["goal_emb"].reshape(-1, dim)[-1].float().cpu().numpy()
            plan["iters"].append(
                {
                    "cand": action_candidates[0].float().cpu().numpy(),  # (S, H, A*block)
                    "cost": cost[0].float().cpu().numpy(),  # (S,)
                    "pred": info_dict["predicted_emb"][0].float().cpu().numpy(),  # (S, T, D)
                }
            )
            return cost

        def solve(info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
            init = None if init_action is None else init_action[0].float().cpu().numpy()
            rec.plans.append({"env_step": len(rec.steps) - 1, "iters": [], "init": init})
            out = orig_solve(info_dict, init_action=init_action)
            rec.plans[-1]["actions"] = out["actions"][0].float().cpu().numpy()
            return out

        def get_action(info_dict: dict, **kwargs: Any) -> np.ndarray:
            if rec.goal_pixels is None:
                rec.goal_pixels = np.array(info_dict["goal"][0, -1], copy=True)
                rec.goal_state = np.array(info_dict["goal_state"][0, -1], copy=True)
            term = info_dict.get("terminated")
            rec.steps.append(
                {
                    "pixels": np.array(info_dict["pixels"][0, -1], copy=True),
                    "state": np.array(info_dict["state"][0, -1], copy=True),
                    "terminated": bool(np.asarray(term).reshape(-1)[0]) if term is not None else False,
                }
            )
            action = orig_get_action(info_dict, **kwargs)
            rec.steps[-1]["action"] = np.asarray(action).reshape(-1)
            return action

        model.get_cost = get_cost
        solver.solve = solve
        policy.get_action = get_action


# ---------------------------------------------------------------- encoding


def jpeg_b64(img: np.ndarray, size: int = 160, quality: int = 80) -> str:
    im = Image.fromarray(np.asarray(img, dtype=np.uint8)).resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def q16(arr: np.ndarray) -> dict[str, Any]:
    """Quantize a float array to int16 and base64 it. Decode: v = q * scale."""
    arr = np.asarray(arr, dtype=np.float64)
    scale = float(np.abs(arr).max()) / 32000.0 or 1.0
    q = np.round(arr / scale).astype("<i2")
    return {"shape": list(arr.shape), "scale": scale, "b64": base64.b64encode(q.tobytes()).decode()}


def q8(arr: np.ndarray) -> dict[str, Any]:
    """Like q16 but int8 (for the bulky candidate actions)."""
    arr = np.asarray(arr, dtype=np.float64)
    scale = float(np.abs(arr).max()) / 127.0 or 1.0
    q = np.round(arr / scale).astype("i1")
    return {"shape": list(arr.shape), "scale": scale, "b64": base64.b64encode(q.tobytes()).decode(), "int8": True}


def r(x: Any, nd: int = 4) -> Any:
    return np.round(np.asarray(x, dtype=np.float64), nd).tolist()


def build_payload(
    rec: Recorder,
    cfg: DictConfig,
    action_scaler: Any,
    topk: int,
    success: bool,
    meta: dict[str, Any],
) -> dict[str, Any]:
    block = int(cfg.plan_config.action_block)
    plans_out = []
    for plan in rec.plans:
        z0, g = plan["z0"], plan["goal_emb"]
        iters = plan["iters"]
        pred = np.stack([it["pred"] for it in iters])  # (I, S, T, D)
        cost = np.stack([it["cost"] for it in iters])  # (I, S)
        cand = np.stack([it["cand"] for it in iters])  # (I, S, H, A*block)
        n_iter, n_samp, n_t, dim = pred.shape

        # 2D view: x = progress along z0 -> goal, y = main spread orthogonal to it.
        e1 = (g - z0) / (np.linalg.norm(g - z0) + 1e-9)
        flat = pred.reshape(-1, dim) - z0
        resid = flat - np.outer(flat @ e1, e1)
        sub = resid[np.random.default_rng(0).choice(len(resid), min(20000, len(resid)), replace=False)]
        _, _, vt = np.linalg.svd(sub - sub.mean(0), full_matrices=False)
        e2 = vt[0] - (vt[0] @ e1) * e1
        e2 /= np.linalg.norm(e2) + 1e-9
        basis = np.stack([e1, e2], 1)  # (D, 2)
        proj = (pred - z0) @ basis  # (I, S, T, 2)
        goal_xy = (g - z0) @ basis
        # how much of each sample's final error to the goal the 2D view keeps
        err = pred[:, :, -1] - g
        kept = ((err @ basis) ** 2).sum(-1) / ((err**2).sum(-1) + 1e-9)

        # elite indices and the CEM distribution after each iteration
        order = np.argsort(cost, axis=1)[:, :topk]  # (I, K)
        elites = np.take_along_axis(cand, order[:, :, None, None], axis=1)
        mean = elites.mean(1)  # (I, H, A*block)
        std = elites.std(1, ddof=1)

        # actions in env units: (I, S, H*block, 2), relative agent-target offsets in pixels
        a_dim = cand.shape[-1] // block
        raw = action_scaler.inverse_transform(cand.reshape(-1, a_dim)).reshape(
            n_iter, n_samp, -1, a_dim
        )
        mean_raw = action_scaler.inverse_transform(mean.reshape(-1, a_dim)).reshape(n_iter, -1, a_dim)

        plans_out.append(
            {
                "env_step": plan["env_step"],
                "goal_xy": r(goal_xy),
                "goal_dist": float(np.linalg.norm(g - z0)),
                "lat": q16(proj),
                "cost": q16(cost),
                "elite": order.tolist(),
                "act": q8(raw),
                "mean_act": r(mean_raw, 3),
                "mean_norm": r(mean, 3),
                "std_norm": r(std, 3),
                "view_kept": r(np.median(kept, axis=1), 3),
                "init_mean": None if plan["init"] is None else r(plan["init"], 3),
            }
        )

    steps_out = [
        {
            "img": jpeg_b64(s["pixels"]),
            "state": r(s["state"], 2),
            "action": r(s.get("action", [np.nan, np.nan]), 4),
            "terminated": s["terminated"],
        }
        for s in rec.steps
    ]
    return {
        "meta": meta
        | {
            "success": success,
            "horizon": int(cfg.plan_config.horizon),
            "receding_horizon": int(cfg.plan_config.receding_horizon),
            "action_block": block,
            "num_samples": int(cfg.solver.num_samples),
            "n_steps": int(cfg.solver.n_steps),
            "topk": topk,
            "var_scale": float(cfg.solver.var_scale),
            "eval_budget": int(cfg.eval.eval_budget),
            "goal_offset_steps": int(cfg.eval.goal_offset_steps),
            "img_size": int(cfg.eval.img_size),
            "embed_dim": int(rec.plans[0]["z0"].shape[0]),
            "action_scale": 100.0,
            "window": 512,
        },
        "goal": {"img": jpeg_b64(rec.goal_pixels), "state": r(rec.goal_state, 2)},
        "steps": steps_out,
        "plans": plans_out,
    }


# ---------------------------------------------------------------- main


@hydra.main(version_base=None, config_path="../config/eval", config_name="pusht")
def main(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.eval.num_eval = 1
        dump = cfg.get("dump") or {}
    out = Path(hydra.utils.to_absolute_path(dump.get("out", "docs/viz/lewm_planning_data.js")))
    tries = int(dump.get("tries", 12))
    start = int(dump.get("start", 0))
    assert cfg.policy != "random", "set policy=<checkpoint>"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    dataset = lewm_eval.get_dataset(cfg, cfg.eval.dataset_name)
    process = lewm_eval.fit_process(cfg, dataset)

    episode_len = np.asarray(dataset.lengths)
    offsets = np.asarray(dataset.offsets)
    max_start = episode_len - cfg.eval.goal_offset_steps - 1
    row_ep = np.repeat(np.arange(len(episode_len)), episode_len)
    row_step = np.arange(episode_len.sum()) - np.repeat(offsets - offsets[0], episode_len)
    valid = np.nonzero(row_step <= max_start[row_ep])[0]
    picks = np.random.default_rng(cfg.seed).permutation(valid)[start : start + tries]

    for i, row in enumerate(picks):
        ep, st = int(row_ep[row]), int(row_step[row])
        print(f"[try {i}] episode {ep} start {st}")
        world = swm.World(**cfg.world, image_shape=(224, 224))
        policy = lewm_eval.build_policy(cfg, process)
        rec = Recorder()
        world.set_policy(policy)
        rec.hook(policy)
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=[st],
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=[ep],
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        )
        success = bool(np.asarray(metrics["episode_successes"]).reshape(-1)[0])
        # final frame after the last env step
        rec.steps.append(
            {
                "pixels": np.array(world.infos["pixels"][0, -1], copy=True),
                "state": np.array(world.infos["state"][0, -1], copy=True),
                "terminated": success,
            }
        )
        print(f"  success={success} env_steps={len(rec.steps) - 1} replans={len(rec.plans)}")
        if success and len(rec.plans) >= 2 or i == len(picks) - 1:
            break

    payload = build_payload(
        rec,
        cfg,
        process["action"],
        topk=int(cfg.solver.topk),
        success=success,
        meta={"policy": str(cfg.policy), "episode": ep, "start_step": st, "env": cfg.world.env_name},
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.LEWM_PLAN = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
