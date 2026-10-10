# Let the planner see older frames (history_len > 1)

Status: idea, not started.

## Background

At eval time, the CEM planner uses `history_len = 1`. It encodes only the
current frame (z₀) and starts all imagined rollouts from it. See
`docs/viz/lewm_planning_pipeline.html` ("the planner sees only the current
frame, not older frames").

`history_len` is not set in `config/eval/*.yaml`. It is a field of the
stable_worldmodel `PlanConfig`, with default `history_len: int = 1`
(`.venv/lib/python3.10/site-packages/stable_worldmodel/policy.py:31`).

But the library **does not use** `history_len`. No code reads it. The time
axis of size 1 is hard-coded in the env pool: `_stack_fresh` in
`stable_worldmodel/world/env_pool.py` adds a time dim of 1 to each info value,
so `pixels` is always `(E, 1, ...)`. So to set `history_len = 3` in the config
does nothing. We must add the frame history ourselves.

The predictor was trained with `history_size: 3` (`config/train/lewm.yaml`).
So it can take up to 3 real context latents without retraining.

## Why it can help

- One frame does not show velocity. Two or three frames do.
- Tasks with fast or dynamic objects, or with occlusion, can need this
  information.
- With `history_len = 3`, the context at plan time is the same as in training
  (3 real frames).

## Why 1 is the current default

- In PushT, TwoRoom, Reacher and Cube, one frame shows almost all of the state.
  The tasks are slow, and each action block is 5 env steps.
- Causal attention means that "predict from 1 real frame" is already a training
  case, so the current setup is in-distribution.
- More frames cost more: more encoding for each candidate and each CEM
  iteration.

## What already works in the model

- `rollout()` in `jepa.py` reads the number of known frames from the input:
  `H = info["pixels"].size(2)`. It encodes all H frames and keeps the last
  `history_size = 3` latents as context.
- The goal and the cost (`criterion`) use only the last frame and the last
  predicted latent. They need no change.

## What is missing

1. **A frame history.** The env pool always gives `pixels` as `(E, 1, ...)`.
   Something must keep the last N frames for each env, one frame every
   `action_block` = 5 env steps, and give `(E, N, ...)`. A policy subclass is
   probably the smallest change. The policy's `_prepare_info` already handles
   an `(E, T, ...)` time axis.
2. **The real past actions (the hard part).** `rollout()` takes the first H
   action blocks from the candidates:
   ```python
   act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
   ```
   With H = 3, CEM would sample and "optimize" the action blocks of the 2 past
   frames, but those actions already happened. The fix: keep the last N - 1
   executed action blocks, pass them in `info` (for example
   `info["past_action"]`), and put them before the candidate blocks in
   `rollout()`. Then CEM still plans only the 5 future blocks.
3. **Episode start.** For the first plans there are fewer than N frames.
   Options: repeat the first frame (and use zero past actions), or start with
   H = 1 and grow it.

Estimate: about 100 lines of new code (a policy subclass and a small
`rollout()` change), plus eval runs on RunPod.

## Work items

1. **Offline check first (go / no-go).** Do this before any planner change.
   - Take clips from the PushT dataset.
   - Roll out the predictor open-loop for 5 blocks with the **real** dataset
     actions. Start once from 1 real frame and once from 3 real frames.
   - Compare the latent error of the last predicted frame against the real
     encoded frame (also at each step).
   - If 3 frames do not make the prediction clearly better, the planning eval
     will not improve. Stop here.
   - This needs only the model and the data, no env and no planner.
2. Add the frame history (missing part 1).
3. Add the real past actions to `rollout()` (missing part 2).
4. Handle episode start (missing part 3).
5. Encode the history frames one time for each plan, not in each `get_cost`
   call. Then broadcast the latents to all candidates.
6. Run eval on RunPod with N = 1, 2, 3 on PushT (then the other tasks).
   Compare success rate and time per plan.

## Done when

- The offline check gives a clear answer. If it is "no gain", record the
  numbers here and close the idea.
- If it is "gain": the table of success rate and plan time for each N and task
  is available, and we decide to keep 1 or change the default for some tasks.
