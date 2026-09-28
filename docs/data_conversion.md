# Converting datasets from HDF5 to Lance

The LeWM datasets on [HuggingFace](https://huggingface.co/collections/quentinll/lewm) are published as HDF5 (`.h5`) files. Training can also read [Lance](https://lancedb.github.io/lance/) datasets, a columnar format with fast random row access that works well with many parallel DataLoader workers. This guide covers converting an `.h5` dataset to `.lance` with the `swm` CLI that ships with `stable-worldmodel`.

## Prerequisites

Install the project environment:

```bash
uv sync
```

This includes the `format` extra of `stable-worldmodel` (`h5py` and `hdf5plugin`). Without it, `stable-worldmodel` silently disables its HDF5 reader, and every `.h5` file fails with:

```
ValueError: No format detected for '.../foo.h5'; pass format= explicitly.
```

To check that both formats are available:

```bash
python -c "from stable_worldmodel.data.format import FORMATS; print(list(FORMATS))"
```

The output should include both `hdf5` and `lance`.

## Directory layout

`swm` and `train.py` look for datasets in a `datasets/` subfolder of a root directory:

| Tool | Root directory |
|---|---|
| `swm convert` | `$STABLEWM_HOME` (default `~/.stable_worldmodel`) |
| `train.py` | `$LOCAL_DATASET_DIR`, falling back to `$STABLEWM_HOME` |

So for a root of `/data/swm`, the files should look like this:

```
/data/swm/
└── datasets/
    ├── pusht_expert_train.h5       # input
    └── pusht_expert_train.lance/   # output (a directory)
```

## 1. Download and decompress

| HF dataset repo | File | Decompress with |
|---|---|---|
| `quentinll/lewm-pusht` | `pusht_expert_train.h5.zst` | `zstd -d --rm pusht_expert_train.h5.zst` |
| `quentinll/lewm-tworooms` | `tworoom.tar.zst` | `tar --zstd -xvf tworoom.tar.zst` |
| `quentinll/lewm-reacher` | `reacher.tar.zst` | `tar --zstd -xvf reacher.tar.zst` |
| `quentinll/lewm-cube` | `cube_single_expert.tar.zst` | `tar --zstd -xvf cube_single_expert.tar.zst` |

Note that pusht is a single compressed `.h5` file, not a tar archive.

Example for pusht:

```bash
export STABLEWM_HOME=/data/swm
hf download quentinll/lewm-pusht pusht_expert_train.h5.zst --repo-type dataset --local-dir $STABLEWM_HOME/datasets
zstd -d --rm $STABLEWM_HOME/datasets/pusht_expert_train.h5.zst
```

After extracting a tar archive, check that the resulting `.h5` path matches the `name:` in the corresponding `config/train/data/*.yaml`.

## 2. Convert

### Using the CLI

```bash
STABLEWM_HOME=/data/swm swm convert pusht_expert_train pusht_expert_train.lance -f lance
```

- The first argument is the dataset name **without** `.h5`. It is looked up in `$STABLEWM_HOME/datasets/`.
- The second argument is the output name, written to the same `datasets/` folder.
- **`-f lance` is required.** The CLI's default output format is `video`.

### Using Python, for arbitrary paths

`swm convert` only reads from `$STABLEWM_HOME/datasets/`. For files elsewhere, call the underlying function with full paths:

```bash
python -c "from stable_worldmodel.data import convert; convert('/path/to/foo.h5', '/path/to/out/foo.lance', dest_format='lance')"
```

### Expected warning

```
WARNING:root:LanceWriter: dropping incoming columns ['episode_idx', 'step_idx'] — names reserved for the writer-managed index columns.
```

This is harmless. The Lance writer rebuilds `episode_idx` and `step_idx` itself, using the same episode order as the source. A similar warning about dropping *non-numeric* columns is also harmless, as long as it doesn't list a column used for training (`pixels`, `action`, `proprio`, `state`).

Conversion takes a while for the larger datasets. The uncompressed pusht file is about 46 GB.

### Why the Lance dataset is much smaller

The `.lance` output is expected to be about a third of the size of the `.h5` input. For pusht, 44 GB became 14 GB. That's because the formats store images differently:

- **HDF5:** `pixels` are stored with lossless compression (about 36 GB for pusht).
- **Lance:** each frame is stored as a JPEG image at quality 95 (about 5.8 KB per frame for pusht). All other columns are copied exactly.

JPEG is lossy, but at quality 95 the difference can't be seen. On 200 random pusht frames, the average pixel difference was 0.11 out of 255 (PSNR 53 dB), with larger differences only at a few sharp edges. `episode_idx`, `step_idx` and `action` matched exactly. Training uses the full 224×224 frames (`img_size: 224`), so the artifacts do reach the model, but they are far smaller than normal frame-to-frame variation and are not expected to affect results. One side effect: evaluation environments render raw pixels, so a model trained on the Lance data sees slightly different images at eval time. To confirm this doesn't matter for your setup, train two short runs with the same seed, one on `.h5` and one on `.lance`, and compare their loss curves.

If you need pixels closer to the originals, raise the JPEG quality using the Python function. The result is still JPEG and will be larger:

```bash
python -c "from stable_worldmodel.data import convert; convert('/path/to/foo.h5', '/path/to/out/foo.lance', dest_format='lance', jpeg_quality=100)"
```

## 3. Verify

```bash
STABLEWM_HOME=/data/swm python -c "
import stable_worldmodel as swm
for name in ['pusht_expert_train.h5', 'pusht_expert_train.lance']:
    d = swm.data.load_dataset(name)
    print(name, 'episodes:', len(d.lengths), 'steps:', int(d.lengths.sum()))
"
```

The episode and step counts should match between the two formats.

## 4. Train on the converted dataset

Set `name:` in the dataset config to the `.lance` name:

```yaml
dataset:
  name: pusht_expert_train.lance
```

`name:` can also be a HuggingFace repo id. `config/train/data/pusht.yaml` uses the converted pusht dataset published at [`librakevin/lewm-pusht`](https://huggingface.co/datasets/librakevin/lewm-pusht):

```yaml
dataset:
  name: librakevin/lewm-pusht
```

A repo id resolves to `datasets/<user>--<repo>/` (here `datasets/librakevin--lewm-pusht/`), which holds one copy of the repo: its `README.md` and the top-level `*.lance` directory. If that folder doesn't exist, the repo is downloaded into it on first use.

The other data configs (`tworoom.yaml`, `dmc.yaml`, `ogb.yaml`) still point to `.h5` files. Change their `name:` if you convert those datasets too.

Then point `train.py` at the root directory. It appends `datasets/` itself:

```bash
export LOCAL_DATASET_DIR=/data/swm
python train.py data=pusht
```
