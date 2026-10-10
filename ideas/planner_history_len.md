# Let the planner see older frames (history_len > 1)

Status: idea, not started.

## Background

At eval time, the CEM planner uses `history_len = 1`. It encodes only the
current frame (z₀) and starts all imagined rollouts from it. See
`docs/viz/lewm_planning_pipeline.html` ("the planner sees only the current
frame, not older frames").

`history_len` is not set in `config/eval/*.yaml`. It is probably the default of
the planning library (stable_worldmodel). Check this before you start.

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

## Work items

1. Find where `history_len` is set and how the policy fills the context.
2. Keep a buffer of the last N observed frames and the last N executed action
   blocks. Make sure the action blocks align with `action_block` (frameskip).
3. Decide what to do at episode start when there are fewer than N frames
   (pad with the first frame, or use a shorter context).
4. Encode the history frames one time for each plan, not in each `get_cost`
   call. Then broadcast the latents to all candidates.
5. Run eval on RunPod with `history_len` = 1, 2, 3 on PushT (then the other
   tasks). Compare success rate and time per plan.

## Done when

- The table of success rate and plan time for each `history_len` and task is
  available.
- We have a decision: keep 1, or change the default for some tasks.
