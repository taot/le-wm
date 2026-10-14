# Project instructions

## Debugging a crashed run

When a training or eval run crashes:

1. Rerun the exact failing command (same overrides, same machine) and save all
   output: `> run.log 2>&1`. Read the real error there.
2. Try the smallest likely fix with that real command, for example a default
   value that is known to work.
3. Only then investigate the root cause.

Do not treat a simplified test script as proof of a fix, and do not change
config or say that a fix works until the real command passes. If the machine
has several resource limits, read all of them first, and compare the cost of
each worker with each limit.

## Crashes on RunPod (`ssh runpod`)

- **The real error is in the console output.** Hydra's
  `outputs/<date>/<time>/train.log` can be almost empty.
- **DataLoader workers are spawned.** `stable_worldmodel` forces the `spawn`
  start method, because LanceDB is not fork-safe
  (`.venv/lib/python3.10/site-packages/stable_worldmodel/data/formats/lance.py`).
  Each worker imports `train.py` again (~10 s), gets the pickled dataset
  (~136 MB with `keys_to_cache`), and the workers start one after the other.
  So threads, memory and start time all grow with `num_workers`. The train and
  val loaders both keep their workers (`persistent_workers`).
- **Spawned workers do not match `pgrep -f train.py`.** Their command line is
  `python -c from multiprocessing.spawn ...`. Use `pgrep -f spawn_main`.
- **`nproc` and `free` show the host, not the pod.** The real limits are in
  `/sys/fs/cgroup/`: `cpu.max`, `memory.max`, `memory.events` (look at
  `oom_kill`), `pids.max`, `pids.events`, `pids.peak`.
- **`DataLoader worker ... exited unexpectedly` with exit code 1** usually
  means a Python error in the worker, for example `can't start new thread`.
  A `Killed` main process together with a higher `oom_kill` count means the
  memory limit.

The current settings for RunPod (`num_workers: 6`, `OMP_NUM_THREADS=1`) and
the reason for them are in the comments of `config/train/launcher/runpod.yaml`
and `.env-runpod`.
