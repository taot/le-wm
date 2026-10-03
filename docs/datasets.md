# Datasets

This guide covers where datasets live, how to download them, how to convert HDF5 to Lance, and how to browse a Lance dataset. For what is inside the PushT dataset, see [pusht_dataset.md](pusht_dataset.md).

## Formats

- The upstream LeWM datasets on [Hugging Face](https://huggingface.co/collections/quentinll/lewm) are **HDF5** (`.h5`) files.
- This repo trains PushT on a **Lance** (`.lance`) copy, published as [`librakevin/lewm-pusht`](https://huggingface.co/datasets/librakevin/lewm-pusht). Lance is a columnar format with fast random row access, which works well with many parallel data-loader workers.

## Where datasets live

Every tool looks for datasets in `$STABLEWM_HOME/datasets/`. Checkpoints also go under `$STABLEWM_HOME`, in `checkpoints/`. `$STABLEWM_HOME` defaults to `~/.stable_worldmodel`.

Set it once per machine in the repo's `.env` file (copy `.env.example`). `train.py`, `eval.py` and `playground.py` read it automatically. Run `source .env` to use `$STABLEWM_HOME` in the shell commands on this page.

For a root of `/data/swm`:

```
/data/swm/
└── datasets/
    ├── librakevin--lewm-pusht/         # a Hugging Face repo (see below)
    │   └── pusht_expert_train.lance/
    ├── pusht_expert_train.h5           # a local .h5 file
    └── pusht_expert_train.lance/       # a local .lance dataset (a folder)
```

The `name:` in a data config (`config/train/data/*.yaml`) can be:

- a **Hugging Face repo id**, like `librakevin/lewm-pusht` (used by `pusht.yaml`). It resolves to `datasets/<user>--<repo>/` and is downloaded there on first use if that folder doesn't exist.
- a **local file name**, like `pusht_expert_train.lance` or `pusht_expert_train` (for `pusht_expert_train.h5`).

The other data configs (`tworoom.yaml`, `dmc.yaml`, `ogb.yaml`) still point to `.h5` files.

## Download

### PushT (Lance)

Training downloads `librakevin/lewm-pusht` automatically on first use, but downloading it yourself is more reliable. `hf download` fetches files in parallel and can resume. The built-in loader fetches one file at a time, and if it is interrupted, it later treats the partial folder as a complete dataset (it only checks that the folder isn't empty).

From the repo folder, load `STABLEWM_HOME` from `.env` (see [Where datasets live](#where-datasets-live)), then download:

```bash
source .env
```

```bash
hf download librakevin/lewm-pusht --repo-type dataset \
  --local-dir $STABLEWM_HOME/datasets/librakevin--lewm-pusht
```

If it stops partway, run the same command again. Afterward, the folder should contain `pusht_expert_train.lance/`.

### Upstream HDF5 datasets

| HF dataset repo | File | Decompress with |
|---|---|---|
| `quentinll/lewm-pusht` | `pusht_expert_train.h5.zst` | `zstd -d --rm pusht_expert_train.h5.zst` |
| `quentinll/lewm-tworooms` | `tworoom.tar.zst` | `tar --zstd -xvf tworoom.tar.zst` |
| `quentinll/lewm-reacher` | `reacher.tar.zst` | `tar --zstd -xvf reacher.tar.zst` |
| `quentinll/lewm-cube` | `cube_single_expert.tar.zst` | `tar --zstd -xvf cube_single_expert.tar.zst` |

PushT is a single compressed `.h5` file, not a tar archive. Example, from the repo folder:

```bash
source .env
hf download quentinll/lewm-pusht pusht_expert_train.h5.zst --repo-type dataset --local-dir $STABLEWM_HOME/datasets
zstd -d --rm $STABLEWM_HOME/datasets/pusht_expert_train.h5.zst
```

After extracting a tar archive, check that the `.h5` file name matches the `name:` in the matching `config/train/data/*.yaml`.

## Convert HDF5 to Lance

`uv sync` installs the `format` extra of `stable-worldmodel` (`h5py`, `hdf5plugin`), which the HDF5 reader needs. Without it, every `.h5` file fails with `ValueError: No format detected for '.../foo.h5'`. To check:

```bash
python -c "from stable_worldmodel.data.format import FORMATS; print(list(FORMATS))"
```

The output should include both `hdf5` and `lance`.

### Run the conversion

From the repo folder, so `swm convert` gets `STABLEWM_HOME` from `.env`:

```bash
source .env
swm convert pusht_expert_train pusht_expert_train.lance -f lance
```

- The first argument is the dataset name **without** `.h5`, looked up in `$STABLEWM_HOME/datasets/`.
- The second argument is the output name, written to the same folder.
- **`-f lance` is required.** The default output format is `video`.

For files outside `$STABLEWM_HOME/datasets/`, call the Python function with full paths:

```bash
python -c "from stable_worldmodel.data import convert; convert('/path/to/foo.h5', '/path/to/out/foo.lance', dest_format='lance')"
```

This warning is harmless (the Lance writer rebuilds these two columns itself, in the same order):

```
WARNING:root:LanceWriter: dropping incoming columns ['episode_idx', 'step_idx'] — names reserved for the writer-managed index columns.
```

A similar warning about dropping *non-numeric* columns is also harmless, as long as it doesn't list a column used for training (`pixels`, `action`, `proprio`, `state`). Converting the larger datasets takes a while (uncompressed pusht is about 46 GB).

### Why the Lance copy is smaller

The `.lance` output is about a third of the size (pusht: 44 GB → 14 GB), because frames are stored differently:

- **HDF5:** `pixels` use lossless compression.
- **Lance:** each frame is a JPEG at quality 95 (about 5.8 KB per frame for pusht). All other columns are copied exactly.

At quality 95 the difference can't be seen: on 200 random pusht frames, the average pixel difference was 0.11 out of 255 (PSNR 53 dB). This is much smaller than normal frame-to-frame change and is not expected to affect training. Note that evaluation environments render raw pixels, so a Lance-trained model sees slightly different images at eval time. To check this for your setup, train two short runs with the same seed, one on `.h5` and one on `.lance`, and compare the loss curves.

For pixels closer to the originals, raise the quality (the result is larger):

```bash
python -c "from stable_worldmodel.data import convert; convert('/path/to/foo.h5', '/path/to/out/foo.lance', dest_format='lance', jpeg_quality=100)"
```

### Check the result

```bash
source .env
python -c "
import stable_worldmodel as swm
for name in ['pusht_expert_train.h5', 'pusht_expert_train.lance']:
    d = swm.data.load_dataset(name)
    print(name, 'episodes:', len(d.lengths), 'steps:', int(d.lengths.sum()))
"
```

The episode and step counts should be the same for both. To train on it, set `name: pusht_expert_train.lance` in the data config.

## Browse a Lance dataset

[Lance Data Viewer](https://github.com/lance-format/lance-data-viewer) is a read-only web UI for a dataset's schema and rows. Mount the folder that **contains** the `.lance` dataset(s) at `/data`. From the repo folder:

```bash
source .env
docker run --rm -p 8080:8080 \
    -v $STABLEWM_HOME/datasets/librakevin--lewm-pusht:/data:ro \
    ghcr.io/lance-format/lance-data-viewer:lancedb-0.36.0
```

Then open <http://localhost:8080>. Every `*.lance` folder under the mount shows up as a dataset.

If it doesn't work:

- **`Data path /data does not exist or is not mounted`:** use the `host:container` form for `-v`. With only a host path, Docker creates an empty volume instead.
- **`Data path /data is not readable`:** the container runs as UID 1001. Run `chmod -R o+rX` on the folder, or add `--user $(id -u):$(id -g)`.
- **A dataset won't open:** the image tag pins the `lancedb` version. Switch to the tag that matches the Lance version the dataset was written with.
