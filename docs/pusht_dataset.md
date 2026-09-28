# Understanding the PushT dataset

This guide explains the contents of `pusht_expert_train.lance` (HF: `librakevin/lewm-pusht`). To browse it in a web UI, see [viewing_lance_datasets.md](viewing_lance_datasets.md).

The dataset is a flat table: **each row is one timestep of one PushT demonstration**.

## The task

PushT is a 2D pushing task:

![PushT frame: episode 0, step 0](assets/pusht_frame0.png)

*`pixels` of episode 0, step 0.*

- **Blue circle**: the agent (the "pusher" you control).
- **Grey T**: the block to be pushed.
- **Green T**: the goal pose, fixed and drawn on the floor.

The goal is to push the grey T until it covers the green one. `expert` means these are expert (scripted or skilled) demonstrations, used as training data.

## Size

- **2,336,736 rows** (timesteps).
- **18,685 episodes**, 49 to 246 steps each (about 125 on average).

## Columns

| Column | Shape | Meaning |
|---|---|---|
| `episode_idx` | int32 | Which demonstration the row belongs to (0 to 18684) |
| `step_idx` | int32 | Timestep within that episode (0 to 245) |
| `pixels` | JPEG bytes | 224×224 RGB render of the scene at this step; the model's visual input |
| `action` | 2 × float32 | Target offset for the agent, in units of 100 px: `target = agent_pos + action * 100`. Raw env action (not normalized), nominally in [−1, 1]. See [The `action` field](#the-action-field) |
| `proprio` | 4 × float32 | The agent's own body state: `[agent_x, agent_y, agent_vx, agent_vy]`, in pixel units of a 512×512 world |
| `state` | 7 × float32 | Full simulator state: `[agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]`. The angle is in radians (0 to 2π) |

Things to notice:

- `proprio` is a subset of `state` (agent position and velocity). `state` adds the T's position and angle, which is the privileged information the model has to infer from pixels.
- In the first rows of episode 0, the block values (`192.67, 335.16, 2.95`) stay constant while the agent moves: the agent hasn't reached the T yet.
- At `step_idx=0` the velocity is `0, 0`; every episode starts at rest.

Example (first three rows of episode 0):

| step | action | proprio | state |
|---|---|---|---|
| 0 | `[0.14, -0.25]` | `[423.0, 182.0, 0.0, 0.0]` | `[423.0, 182.0, 192.67, 335.16, 2.95, 0.0, 0.0]` |
| 1 | `[0.06, -0.09]` | `[427.14, 174.61, 52.38, -93.54]` | `[427.14, 174.61, 192.67, 335.16, 2.95, 52.38, -93.54]` |
| 2 | `[-0.02, 0.04]` | `[430.44, 169.26, 21.58, -31.59]` | `[430.44, 169.26, 192.67, 335.16, 2.95, 21.58, -31.59]` |

## The `action` field

An action is a **relative target** for the agent, not a velocity or an absolute position. In one env step (one row, 0.1 s), `PushTEnv.step()` does:

```python
target = agent_pos + action * 100           # action_scale = 100
for _ in range(10):                          # 10 physics substeps of 0.01 s
    accel = 100 * (target - pos) + 20 * (0 - vel)   # PD controller: spring + damper
    vel += accel * 0.01
    pos += vel * 0.01
```

The substeps happen inside the env and aren't stored; the next row holds only the result. The agent is pulled toward `target` but usually doesn't reach it within one step. For row 0 of episode 0:

| | Value |
|---|---|
| agent position | (423.0, 182.0), at rest |
| `action` | (0.14, −0.25) |
| target = pos + 100 · action | (437, 157) |
| next row's agent position | (427.14, 174.61), about 30% of the way |
| next row's agent velocity | (52.38, −93.54) px/s |

Replaying the loop above reproduces the next row's position and velocity exactly. Because the agent keeps its velocity between steps, the same action from a different velocity leads somewhere else; this is why `proprio` includes velocity.

Across the dataset:

- The action space is [−1, 1] per axis (at most 100 px of offset); 99.88% of rows fall inside it, with a few outliers up to about ±2.
- Actions are small: mean ≈ 0, std ≈ 0.21 per axis. The action vector's length is 0.20 at the median, 0.47 at the 90th percentile and 0.76 at the 99th, so a typical offset is about 20 px.
- The agent moves almost exactly in the commanded direction (median cosine 0.999) and covers a median of about 40% of the offset in one step.

[notebooks/inspect_pusht.py](../notebooks/inspect_pusht.py) draws a row's action over its frame.

## Reading it in the viewer

- Filter or sort by `episode_idx` and scroll through `step_idx` in order to replay one demonstration.
- Compare `pixels` with `state`: the blue dot's position should match `state[0:2]`, and the grey T should match `state[2:5]`.

## How training uses it

[config/train/data/pusht.yaml](../config/train/data/pusht.yaml) configures the loader. Training doesn't read single rows. It slices windows of `history_size + num_preds` consecutive steps from one episode, subsampled with `frameskip: 5`. It loads `pixels`, `action`, `proprio` and `state`, and caches everything except `pixels` in memory.

Two transforms change how the model sees `action`:

- **Z-scoring.** [train.py](../train.py) normalizes every non-pixel column, including `action`, with the dataset's mean and std. The stored values are raw; normalization happens only at load time.
- **Frameskip concatenation.** One model step spans 5 rows (0.5 s), so the 5 actions in between are concatenated, and the action encoder's input size is `5 × 2 = 10`.

The JEPA world model learns: given past frames and actions, predict the embedding of future frames.
