"""Web playground for PushT: compare the two eval setups by hand.

Drive the agent with the mouse, or hand control to the world model (the same
CEM policy as eval.py) or, in 25-step mode, to the expert demo. Both setups:

- 25-step (eval.py, config pusht.yaml): start at a dataset state, goal = the
  expert's state 25 steps later; success = agent + block near that goal state.
- Full solve (config pusht_full.yaml): random start, goal = block on the green T;
  success = block within 20 units / 20 deg of the green T.

Run on the GPU machine (Hydra overrides as for eval.py):
    python playground.py policy=pusht/<run>/weights_epoch_003.pt eval.img_size=112
Then tunnel and open http://localhost:8000:
    ssh -N -L 8000:localhost:8000 runpod
"""

import argparse
import asyncio
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
import uvicorn
from hydra import compose, initialize_config_dir
from PIL import Image
from starlette.applications import Starlette
from starlette.responses import FileResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocketDisconnect

import eval as ev  # sets the torch thread cap before torch is used
import stable_worldmodel as swm

ROOT = Path(__file__).parent
HZ = 10  # PushT control rate (env.metadata["render_fps"])
SUCCESS_POS, SUCCESS_ANGLE = 20.0, np.pi / 9


def load_configs(overrides):
    with initialize_config_dir(config_dir=str(ROOT / "config/eval"), version_base=None):
        return compose("pusht", overrides), compose("pusht_full", overrides)


class Playground:
    """One PushT env; every open browser tab views and drives the same episode."""

    def __init__(self, cfg_ds, cfg_full):
        self.cfg_ds, self.cfg_full = cfg_ds, cfg_full
        self.offset = cfg_ds.eval.goal_offset_steps

        dataset = ev.get_dataset(cfg_ds, cfg_ds.eval.dataset_name)
        self.process = ev.fit_process(cfg_ds, dataset)
        self.policy = ev.build_policy(cfg_full, self.process)
        self.model = self.policy.solver.model
        self.plan_len = cfg_full.plan_config.horizon * cfg_full.plan_config.action_block

        # wrap the solver to score the plan it picks, on exactly the inputs it saw
        solve = self.policy.solver.solve

        def solve_and_score(info_dict, init_action=None):
            out = solve(info_dict, init_action=init_action)
            self.imagined = self.imagined_cost(info_dict, out["actions"])
            return out

        self.policy.solver.solve = solve_and_score

        # expert windows long enough for a start + offset-step goal
        self.lengths = np.asarray(dataset.lengths)
        self.offsets = np.asarray(dataset.offsets) - np.asarray(dataset.offsets)[0]
        self.states = dataset.get_col_data("state")
        self.actions = dataset.get_col_data("action")
        self.rng = np.random.default_rng(cfg_ds.seed)

        # no time limit: the budget is shown, not enforced, so you can keep going
        self.world = swm.World(
            cfg_ds.world.env_name, num_envs=1, image_shape=(224, 224), max_episode_steps=100_000
        )
        self.world.set_policy(self.policy)
        self.env = self.world.envs.envs[0].unwrapped
        # scratch env for geometry and for previewing a plan in the real simulator
        self.sim = swm.World(
            cfg_ds.world.env_name, num_envs=1, image_shape=(224, 224), max_episode_steps=100_000
        ).envs.envs[0].unwrapped
        self.sim.reset(seed=0)

        self.controller = "human"
        self.mouse = None  # world-space target while the button is held
        self.new_episode("dataset")

    # -- episodes ---------------------------------------------------------

    def new_episode(self, mode, same=False):
        if not same:
            if mode == "dataset":
                ep = self.rng.choice(np.nonzero(self.lengths > self.offset)[0])
                start = self.rng.integers(0, self.lengths[ep] - self.offset)
                row = self.offsets[ep] + start
                self.episode = {
                    "mode": mode,
                    "label": f"dataset episode {ep}, step {start}",
                    "options": {
                        "state": self.states[row],
                        "goal_state": self.states[row + self.offset],
                    },
                    "expert": self.actions[row : row + self.offset],
                    "budget": self.cfg_ds.eval.eval_budget,
                    "seed": int(self.rng.integers(1 << 31)),
                }
            else:
                goal_state = np.concatenate(
                    [self.cfg_full.eval.goal_agent_pos, self.env.goal_pose, [0.0, 0.0]]
                )
                seed = int(self.rng.integers(1 << 31))
                self.episode = {
                    "mode": mode,
                    "label": f"random start, seed {seed}",
                    "options": {"goal_state": goal_state},
                    "expert": None,
                    "budget": self.cfg_full.eval.eval_budget,
                    "seed": seed,
                }

        self.world.reset(seed=self.episode["seed"], options=self.episode["options"])
        self.goal_state = np.asarray(self.env.goal_state, dtype=np.float64)
        self.step_idx = 0
        self.history = []  # per step: latent cost, success flags
        # per model plan: step it started, real cost then, cost the model imagined
        # at the plan's end, and the real cost once the plan has run (None if cut short)
        self.plans = []
        self.solved_at = None
        self.plan = None
        self.flush_plan()
        self.goal_emb = self.embed("goal")
        self.record()

    def flush_plan(self):
        # drop the policy's queued actions so it replans from the current state
        # (EnvPool reuses its info dict, so an "_needs_flush" key would stick)
        self.policy._action_buffer[0].clear()
        self.policy._next_init = None
        self.plan = None
        if getattr(self, "plans", None) and self.plans[-1]["real"] is None:
            self.plans[-1]["cut"] = True  # not run to the end, so no real score

    # -- model ------------------------------------------------------------

    @torch.no_grad()
    def embed(self, key):
        info = self.policy._prepare_info({key: self.world.infos[key]})
        pixels = info[key].to(next(self.model.parameters()).device)
        return self.model.encode({"pixels": pixels})["emb"][0, -1]

    def latent_cost(self):
        # what CEM minimizes: squared distance between embedding and goal embedding
        return float(((self.embed("pixels") - self.goal_emb) ** 2).sum())

    @torch.inference_mode()
    def imagined_cost(self, info_dict, actions):
        """The planner's own cost for one action sequence: how far from the goal
        the model predicts the scene will be when the sequence ends."""
        solver = self.policy.solver
        infos = {}
        for k, v in info_dict.items():  # same layout as CEMSolver.solve, 1 sample
            if torch.is_tensor(v):
                dtype = solver.dtype if v.is_floating_point() else None
                infos[k] = v.to(device=solver.device, dtype=dtype).unsqueeze(1)
            elif isinstance(v, np.ndarray):
                infos[k] = v[:, None]
            else:
                infos[k] = v
        candidate = actions.to(device=solver.device, dtype=solver.dtype).unsqueeze(1)
        return float(self.model.get_cost(infos, candidate)[0, 0])

    def model_action(self):
        replanning = len(self.policy._action_buffer[0]) == 0
        self.world.infos["terminated"][:] = False  # keep planning after a success
        action = self.policy.get_action(self.world.infos)
        if replanning:
            rest = [a.numpy() for a in self.policy._action_buffer[0]]
            if rest:
                rest = self.process["action"].inverse_transform(np.stack(rest))
            self.plan = self.preview(np.concatenate([action.reshape(1, -1), np.reshape(rest, (-1, 2))]))
            self.plans.append(
                {
                    "start": self.step_idx,
                    "end": self.step_idx + self.plan_len,
                    "before": self.history[-1]["cost"],
                    "imagined": self.imagined,
                    "real": None,
                }
            )
        return action.reshape(-1)

    def preview(self, actions):
        """Run a plan in the scratch simulator: agent path and final block pose."""
        self.sim._set_state(self.env._get_obs())
        path = [list(self.sim.agent.position)]
        for a in actions:
            self.sim.step(np.asarray(a, dtype=np.float32))
            path.append(list(self.sim.agent.position))
        return {"path": path, "block": self.block_polys(self.sim.block)}

    # -- stepping ---------------------------------------------------------

    def action_for_step(self):
        """Action for this tick, or None to hold still."""
        if self.controller == "human":
            if self.mouse is None:
                return None
            delta = (np.asarray(self.mouse) - np.asarray(self.env.agent.position)) / self.env.action_scale
            return np.clip(delta, -1, 1)
        if self.controller == "expert":
            expert = self.episode["expert"]
            if expert is None or self.step_idx >= len(expert):
                return None
            return expert[self.step_idx]
        return self.model_action()

    def step(self, action):
        _, _, _, _, self.world.infos = self.world.envs.step(
            np.asarray(action, dtype=np.float32).reshape(1, 2)
        )
        self.step_idx += 1
        self.record()

    def record(self):
        ds_ok, _ = self.env.eval_state(self.goal_state, self.env._get_obs())
        pos, ang = ev.block_goal_error(self.env)
        full_ok = bool(pos < SUCCESS_POS and ang < SUCCESS_ANGLE)
        ok = ds_ok if self.episode["mode"] == "dataset" else full_ok
        if ok and self.solved_at is None:
            self.solved_at = self.step_idx
        cost = self.latent_cost()
        self.history.append(
            {"step": self.step_idx, "cost": cost, "ds_ok": bool(ds_ok), "full_ok": full_ok}
        )
        last = self.plans[-1] if self.plans else None
        if last and last["real"] is None and not last.get("cut") and self.step_idx == last["end"]:
            last["real"] = cost

    # -- geometry / state for the browser --------------------------------

    def block_polys(self, body):
        """World-space T polygons for a body posed like the block.

        Uses the block's shapes, since the env's goal body has none of its own.
        """
        return [[list(body.local_to_world(v)) for v in s.get_vertices()] for s in self.env.block.shapes]

    def goal_block_polys(self):
        self.sim._set_state(self.goal_state)
        return self.block_polys(self.sim.block)

    def snapshot(self, full=False):
        env, obs = self.env, self.env._get_obs()
        g = self.goal_state
        agent_err = float(np.linalg.norm(obs[:2] - g[:2]))
        block_err = float(np.linalg.norm(obs[2:4] - g[2:4]))
        ang_err = float(abs((obs[4] - g[4] + np.pi) % (2 * np.pi) - np.pi))
        pos, ang = ev.block_goal_error(env)
        msg = {
            "type": "state",
            "step": self.step_idx,
            "budget": self.episode["budget"],
            "controller": self.controller,
            "agent": list(env.agent.position),
            "block": self.block_polys(env.block),
            "plan": self.plan,
            "plans": self.plans,
            "solved_at": self.solved_at,
            "history": self.history[-1:],
            "view": encode_png(self.world.infos["pixels"][0, -1]),
            "metrics": {
                # dataset rule: norm over (agent xy, block xy) < 20 and angle < 20 deg
                "ds_pos": float(np.linalg.norm(obs[:4] - g[:4])),
                "ds_agent": agent_err,
                "ds_block": block_err,
                "ds_angle": np.degrees(ang_err),
                "full_pos": float(pos),
                "full_angle": float(np.degrees(ang)),
            },
        }
        if full:
            msg.update(
                type="episode",
                mode=self.episode["mode"],
                label=self.episode["label"],
                has_expert=self.episode["expert"] is not None,
                history=self.history,
                green=self.block_polys(env._get_goal_pose_body(env.goal_pose)),
                goal_block=self.goal_block_polys(),
                goal_agent=list(g[:2]),
                goal_image=encode_png(self.world.infos["goal"][0, -1]),
                agent_radius=15,
                world=env.window_size,
            )
        return msg


def encode_png(arr):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_app(pg):
    """One env, one tick loop; every open tab sees (and can drive) the same episode."""
    lock = asyncio.Lock()
    clients = set()
    state = {"paused": False, "ticker": None}

    async def broadcast(msg):
        for ws in list(clients):
            try:
                await ws.send_json(msg)
            except Exception:
                clients.discard(ws)

    async def tick():
        loop = asyncio.get_running_loop()
        while True:
            t0 = loop.time()
            async with lock:
                if not state["paused"] and clients:
                    if pg.controller == "model" and len(pg.policy._action_buffer[0]) == 0:
                        await broadcast({"type": "planning"})
                        action = await loop.run_in_executor(None, pg.action_for_step)
                    else:
                        action = pg.action_for_step()
                    if action is not None:
                        pg.step(action)
                        await broadcast(pg.snapshot())
            await asyncio.sleep(max(0.0, 1 / HZ - (loop.time() - t0)))

    async def index(request):
        return FileResponse(ROOT / "playground.html")

    async def ws_endpoint(ws):
        await ws.accept()
        if state["ticker"] is None:
            state["ticker"] = asyncio.create_task(tick())
        async with lock:
            clients.add(ws)
            await ws.send_json(pg.snapshot(full=True) | {"paused": state["paused"]})
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                async with lock:
                    kind = msg["type"]
                    if kind == "reset":
                        pg.new_episode(msg["mode"], same=msg.get("same", False))
                        pg.controller = "human"
                        await broadcast(pg.snapshot(full=True) | {"paused": state["paused"]})
                    elif kind == "mouse":
                        pg.mouse = msg["pos"]  # None when released
                    elif kind == "controller":
                        pg.controller = msg["who"]
                        pg.flush_plan()
                        await broadcast(pg.snapshot())
                    elif kind == "pause":
                        state["paused"] = msg["paused"]
                        await broadcast({"type": "paused", "paused": state["paused"]})
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)
            if not clients:
                pg.mouse = None

    return Starlette(routes=[Route("/", index), WebSocketRoute("/ws", ws_endpoint)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args, overrides = parser.parse_known_args()

    cfg_ds, cfg_full = load_configs(overrides)
    pg = Playground(cfg_ds, cfg_full)
    print(f"Playground ready on http://{args.host}:{args.port}")
    uvicorn.run(make_app(pg), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
