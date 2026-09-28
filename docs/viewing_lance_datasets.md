# Viewing Lance datasets

[Lance Data Viewer](https://github.com/lance-format/lance-data-viewer) is a read-only web UI for browsing `.lance` datasets: their schema, rows, and vector columns. This guide covers running it with Docker to inspect a converted dataset (see [data_conversion.md](data_conversion.md)).

## Run the viewer

Pull the image:

```bash
docker pull ghcr.io/lance-format/lance-data-viewer:lancedb-0.36.0
```

Mount the directory that **contains** the `.lance` dataset(s) at `/data` in the container:

```bash
docker run --rm -p 8080:8080 \
    -v /home/taot/data/ml_data/manual_download/stable_worldmodel/datasets/librakevin--lewm-pusht:/data:ro \
    ghcr.io/lance-format/lance-data-viewer:lancedb-0.36.0
```

Then open <http://localhost:8080>. Every `*.lance` directory under the mount (here, `pusht_expert_train.lance`) appears as a dataset.

## Notes

- **Use `host:container` for `-v`.** If you pass `-v /some/host/path` with no container path, Docker creates an empty anonymous volume at that path *inside* the container and mounts nothing at `/data`. The container then exits with:

  ```
  ERROR: Data path /data does not exist or is not mounted
  ```

- **`:ro`** mounts the data read-only. The viewer never writes, so this is just a safeguard.
- **Permissions.** The container runs as UID 1001. If the dataset isn't world-readable, the container fails with `Data path /data is not readable`. Either run `chmod -R o+rX` on the directory, or add `--user $(id -u):$(id -g)` to `docker run`.
- **Lance version.** The image tag pins the `lancedb` version (`lancedb-0.36.0`). If a dataset won't open, switch to the tag that matches the Lance version the dataset was written with.

