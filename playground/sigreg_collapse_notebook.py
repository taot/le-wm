import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    import os
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from dotenv import load_dotenv

    # The notebook is in playground/; module.py and .env are in the repo root.
    REPO = mo.notebook_dir().parent
    sys.path.insert(0, str(REPO))
    load_dotenv(REPO / ".env")
    from module import SIGReg

    EMB_DIR = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel")) / "embeddings" / "pusht"

    # Run label -> run folder. Color slots 1-3 (blue, orange, aqua) of the reference palette.
    RUNS = {
        "SIGReg": "2026-09-28_114752_img112_s3072",
        "no SIGReg": "2026-10-14_115337_img112_s3072_nosigreg",
        "SIGReg 10 epochs (img224)": "2026-10-10_120411_img224_s3072",
    }
    COLORS = {"SIGReg": "#2a78d6", "no SIGReg": "#eb6834", "SIGReg 10 epochs (img224)": "#1baf7a"}
    # Runs that logged to W&B (the 10-epoch run had wandb.enabled=false).
    WANDB_RUNS = ["SIGReg", "no SIGReg"]

    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
    return COLORS, EMB_DIR, RUNS, SIGReg, WANDB_RUNS, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Latent collapse with and without SIGReg

    This notebook compares the latent states of PushT runs. The main
    comparison is between run A and run B. They are the same, except for
    one setting: the SIGReg weight.

    | Run | `loss.sigreg.weight` | Image size | Epochs | Folder |
    |---|---|---|---|---|
    | **SIGReg** (run A, baseline) | 0.09 | 112 | 3 | `pusht/2026-09-28_114752_img112_s3072` |
    | **no SIGReg** (run B) | 0 | 112 | 3 | `pusht/2026-10-14_115337_img112_s3072_nosigreg` |
    | **SIGReg 10 epochs (img224)** | 0.09 | 224 | 10 | `pusht/2026-10-10_120411_img224_s3072` |

    The third run is an older run with SIGReg and longer training. We use
    it to see what happens to the latents after epoch 3 (section 10). It
    also uses a larger image size, so it is not a clean comparison with
    run A.

    **Why SIGReg is necessary.** The training loss asks the predictor to
    predict the next embedding. There is an easy way to make this loss zero:
    the encoder gives the same vector for every image. Then the prediction is
    always correct, but the embedding has no information about the image.
    We call this **representation collapse**. SIGReg prevents it: it pushes
    the embeddings to the distribution N(0, I). For the details of SIGReg,
    see `playground/sigreg_notebook.py`.

    There are two types of collapse. We look for both:

    - **Complete collapse.** All images go to almost the same point.
      The spread of the cloud of points becomes very small.
    - **Dimensional collapse.** The points spread out, but only along a few
      directions. The other directions are not used.

    **Before you run this notebook**, encode the frames one time. The
    script needs one image size for all checkpoints in one call, so the
    224-pixel run is a separate call:

    ```
    uv run python scripts/extract_embeddings.py
    uv run python scripts/extract_embeddings.py --runs pusht/2026-10-10_120411_img224_s3072 --batch-size 128
    ```

    On a laptop, the first call takes about 4 minutes (epochs 1–3 of runs
    A and B), and the second takes about 22 minutes (epochs 1–10).
    Both calls encode the same 21,094 frames (same seed). The script saves
    one `.npz` file for each checkpoint in
    `$STABLEWM_HOME/embeddings/pusht/<run>/`. If a run has no files, the
    notebook leaves it out.
    """)
    return


@app.cell
def _(EMB_DIR, RUNS, mo, np):
    data = {}  # (run label, epoch) -> arrays from the .npz file
    for _label, _run in RUNS.items():
        for _f in sorted((EMB_DIR / _run).glob("weights_epoch_*.npz")):
            with np.load(_f) as _d:
                data[(_label, int(_d["epoch"]))] = {k: _d[k] for k in _d.files}
    if not data:
        raise FileNotFoundError(f"no embeddings in {EMB_DIR}; run scripts/extract_embeddings.py first")

    labels = [_l for _l in RUNS if any(_k[0] == _l for _k in data)]
    epochs = sorted({e for _, e in data})

    def runs_at(epoch: int) -> list[str]:
        """Runs that have a checkpoint at this epoch."""
        return [_l for _l in labels if (_l, epoch) in data]

    def run_epochs(label: str) -> list[int]:
        """Epochs of one run."""
        return sorted(e for _l, e in data if _l == label)

    mo.ui.table(
        [
            {
                "run": _l,
                "epoch": _epoch,
                "sigreg weight": float(data[(_l, _epoch)]["sigreg_weight"]),
                "random frames": int((data[(_l, _epoch)]["group"] == 0).sum()),
                "episode frames": int((data[(_l, _epoch)]["group"] == 1).sum()),
                "checkpoint": str(data[(_l, _epoch)]["checkpoint"]),
            }
            for (_l, _epoch) in data
        ],
        selection=None,
    )
    return data, epochs, labels, run_epochs, runs_at


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. What is in the files, and which latent to look at

    Each file has the latents of the same frames, so the runs are compared
    on the same images. There are two groups of frames:

    - **Group 0:** 20,000 random frames from the full dataset. We use them
      for all the statistics.
    - **Group 1:** all the frames of 10 random episodes. We use them to
      draw trajectories in section 6.

    The files also keep the true `state` of each frame: agent x, agent y,
    block x, block y, block angle, agent vx, agent vy.

    Each file has two latents. Use the selector below to choose one:

    - **`emb` (projector output).** This is the embedding that the predictor
      uses and that SIGReg acts on. **This is the main latent.**
    - **`cls` (encoder CLS token).** This is the encoder output, before the
      projector. Use it to see if the encoder itself collapsed, or only the
      projector.

    Note: the projector is `Linear → BatchNorm → GELU → Linear`. The
    BatchNorm is in the hidden layer, not at the output. So the BatchNorm
    does not prevent a collapse of `emb`.

    The **epoch** selector controls the plots that show one epoch. Each
    plot shows only the runs that have a checkpoint at that epoch. So for
    epochs 1–3 you see all three runs, and for epochs 4–10 you see only
    the 10-epoch run. The "over epochs" plots in section 7 always show all
    epochs.
    """)
    return


@app.cell
def _(epochs, mo):
    space = mo.ui.dropdown(
        {"emb (projector output)": "emb", "cls (encoder CLS token)": "cls"},
        value="emb (projector output)",
        label="Latent",
    )
    # Default: the last epoch that all runs have (3), so the first plots compare all runs.
    epoch_select = mo.ui.dropdown({str(e): e for e in epochs}, value=str(min(3, epochs[-1])), label="Epoch")
    mo.hstack([space, epoch_select], justify="start", gap=2)
    return epoch_select, space


@app.cell
def _(SIGReg, data, np, space, torch):
    def latents(label: str, epoch: int, group: int = 0) -> np.ndarray:
        """Latents of one checkpoint, as float64 (frames, 192)."""
        d = data[(label, epoch)]
        return d[space.value][d["group"] == group].astype(np.float64)

    def eigenvalues(x: np.ndarray) -> np.ndarray:
        """Eigenvalues of the covariance matrix, largest first."""
        return np.linalg.eigvalsh(np.cov(x, rowvar=False))[::-1].clip(min=0)

    def sigreg_statistic(x: np.ndarray, batch: int = 128, seed: int = 0) -> float:
        """SIGReg loss as in training: random batches of 128 frames, averaged over batches.

        The frames are shuffled first, because training uses random batches. The
        frames in the files are sorted by dataset row, and nearby episodes look
        alike, so batches of neighboring rows would give a much too high value.
        The shuffle and the random directions use a fixed seed, so every model
        gets the same batches and directions.
        """
        x = x[np.random.default_rng(seed).permutation(len(x))]
        n = len(x) // batch * batch
        proj = torch.from_numpy(x[:n]).float().reshape(-1, batch, x.shape[1])  # (batches, B, D)
        sigreg = SIGReg()
        values = []
        for chunk in proj.split(8):  # 8 batches at a time, to keep the memory small
            torch.manual_seed(seed)
            values.append(sigreg(chunk).item() * len(chunk))
        return sum(values) / len(proj)

    return eigenvalues, latents, sigreg_statistic


@app.cell
def _(data, eigenvalues, labels, latents, np, sigreg_statistic):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    PROBE_TARGETS = ["agent x", "agent y", "block x", "block y", "cos(block angle)", "sin(block angle)"]

    def probe_targets(state: np.ndarray) -> np.ndarray:
        return np.column_stack([state[:, 0], state[:, 1], state[:, 2], state[:, 3], np.cos(state[:, 4]), np.sin(state[:, 4])])

    def linear_probe(label: str, epoch: int) -> np.ndarray:
        """Test R² of a ridge regression from the standardized latent to each target.

        Train and test frames come from different episodes, so the probe cannot
        remember frames that it saw.
        """
        d = data[(label, epoch)]
        g0 = d["group"] == 0
        x, y, groups = latents(label, epoch), probe_targets(d["state"][g0]), d["episode"][g0]
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(x, y, groups))
        probe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        probe.fit(x[train], y[train])
        pred = probe.predict(x[test])
        ss_res = ((y[test] - pred) ** 2).sum(0)
        ss_tot = ((y[test] - y[test].mean(0)) ** 2).sum(0)
        return 1 - ss_res / ss_tot

    # All metrics for every run and epoch, for the selected latent.
    metrics = {}
    for (_label, _epoch) in data:
        _x = latents(_label, _epoch)
        _ev = eigenvalues(_x)
        _p = _ev / _ev.sum()
        _p = _p[_p > 0]
        _std = _x.std(0)
        _cum = np.cumsum(_ev) / _ev.sum()
        metrics[(_label, _epoch)] = {
            "mean_norm": np.linalg.norm(_x.mean(0)),
            "total_var": _ev.sum(),
            "spread": np.sqrt(_ev.sum()),
            "eigenvalues": _ev,
            "dim_std": _std,
            "erank": np.exp(-(_p * np.log(_p)).sum()),
            "pr": _ev.sum() ** 2 / (_ev**2).sum(),
            "n90": int(np.searchsorted(_cum, 0.9) + 1),
            "n99": int(np.searchsorted(_cum, 0.99) + 1),
            "sigreg":sigreg_statistic(_x),
            "sigreg_z": sigreg_statistic((_x - _x.mean(0)) / _std),
            "r2": linear_probe(_label, _epoch),
        }
    gauss_ref = sigreg_statistic(np.random.default_rng(0).standard_normal((20000, 192)))
    assert set(labels) == {_l for _l, _ in metrics}
    return PROBE_TARGETS, gauss_ref, metrics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Scale: how big is the cloud of points?

    Look at the scale first. A complete collapse shows here most clearly.

    The table below gives these values for each checkpoint:

    - **|mean|**: the length of the mean latent vector. This is the
      position of the center of the cloud.
    - **spread**: the square root of the total variance
      (the sum of the variances of all 192 dimensions). This is the typical
      distance from a point to the center of the cloud.
    - **spread / |mean|**: the size of the cloud, relative to its distance
      from zero. If this value is very small, all points are almost at the
      same place, far from zero.

    With N(0, I) in 192 dimensions, the spread is √192 ≈ 13.9 and the mean
    is 0. SIGReg pushes the latents in this direction.
    """)
    return


@app.cell
def _(metrics, mo):
    mo.ui.table(
        [
            {
                "run": _l,
                "epoch": _epoch,
                "|mean|": f"{_m['mean_norm']:.3g}",
                "spread": f"{_m['spread']:.3g}",
                "spread / |mean|": f"{_m['spread'] / _m['mean_norm']:.3g}",
                "smallest dim std": f"{_m['dim_std'].min():.2e}",
                "largest dim std": f"{_m['dim_std'].max():.2e}",
            }
            for (_l, _epoch), _m in metrics.items()
        ],
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plot below shows the standard deviation of each of the 192
    dimensions, sorted from the largest to the smallest.

    **How to read it.** The y-axis is logarithmic: each grid line is
    10 times the line below it. A model with healthy latents gives a flat
    line near 1. A collapsed model gives a line that is many powers of
    10 lower. A line that falls steeply on the right side shows dimensions
    that the model does not use.
    """)
    return


@app.cell
def _(COLORS, epoch_select, metrics, plt, runs_at, space):
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    for _l in runs_at(epoch_select.value):
        _ax.plot(sorted(metrics[(_l, epoch_select.value)]["dim_std"], reverse=True), color=COLORS[_l], lw=2, label=_l)
    _ax.axhline(1, color="0.4", lw=1, ls="--", label="N(0, I): std = 1")
    _ax.set_yscale("log")
    _ax.set_xlabel("dimension (sorted by std)")
    _ax.set_ylabel("std of the dimension")
    _ax.set_title(f"Std of each dimension, {space.value}, epoch {epoch_select.value}")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Are the points all in the same direction?

    The **cosine similarity** of two vectors is 1 if they point in the same
    direction, 0 if they are perpendicular, and −1 if they point in opposite
    directions. We take 5,000 random pairs of frames and calculate the
    cosine similarity of their latents.

    The plot shows **1 − cosine similarity** on a logarithmic x-axis.
    Without the log scale, a collapsed model and a model with
    a small mean would both look like "close to 1".

    **How to read it.** For N(0, I), two random vectors are almost
    perpendicular, so 1 − cos is near 1. If the latents collapsed, every
    pair points in the same direction, and 1 − cos is very close to 0
    (far to the left).
    """)
    return


@app.cell
def _(COLORS, epoch_select, latents, np, plt, runs_at, space):
    _rng = np.random.default_rng(0)
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _all = []
    for _l in runs_at(epoch_select.value):
        _x = latents(_l, epoch_select.value)
        _i, _j = _rng.integers(0, len(_x), (2, 5000))
        _keep = _i != _j
        _xn = _x / np.linalg.norm(_x, axis=1, keepdims=True)
        _all.append((_l, 1 - (_xn[_i[_keep]] * _xn[_j[_keep]]).sum(1)))
    _lo = max(min(v[v > 0].min() for _, v in _all), 1e-16)
    _bins = np.logspace(np.log10(_lo), np.log10(2), 80)
    for _l, _v in _all:
        _ax.hist(_v.clip(min=_lo), bins=_bins, color=COLORS[_l], alpha=0.6, label=f"{_l} (median {np.median(_v):.1e})")
    _ax.set_xscale("log")
    _ax.set_xlabel("1 − cosine similarity of two random frames")
    _ax.set_ylabel("number of pairs")
    _ax.set_title(f"Direction of the latents, {space.value}, epoch {epoch_select.value}")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Eigenvalue spectrum: how many directions are used?

    The **covariance matrix** of the latents describes the shape of the
    cloud of points. Its **eigenvalues** are the variances along the main
    axes of the cloud (the principal components). We sort them from the
    largest to the smallest.

    - **Left plot: raw eigenvalues.** This shows the absolute size of
      each direction. For N(0, I), all 192 eigenvalues are 1.
    - **Right plot: eigenvalues divided by their sum.** This removes the
      total size and shows only the **shape** of the cloud. A flat line
      means all directions are used equally. A steep fall means that a few
      directions have most of the variance (dimensional collapse).

    Two numbers summarize the shape (the table under the plots):

    - **Effective rank**: exp(entropy of the normalized eigenvalues). It is
      the number of directions that the cloud "really" uses. The maximum
      is 192.
    - **Participation ratio**: (Σλ)² / Σλ². It is a second estimate of the
      same thing. It gives more weight to the large eigenvalues.

    **Be careful with a complete collapse.** The right plot and the
    effective rank only describe the shape. If the cloud is 10⁻⁶ wide, its
    shape is not important, because the cloud is almost one point.
    So always read the left plot (or section 1) first.
    Also, eigenvalues below about 10⁻¹⁴ can be float32 rounding noise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Effective rank:

    $$
    \begin{aligned}
    p_i &= \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}, \qquad i = 1, \dots, d \\[6pt]
    H(p) &= -\sum_{i=1}^{d} p_i \log p_i \\[6pt]
    \operatorname{erank}(\Sigma) &= \exp\big(H(p)\big)
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(COLORS, epoch_select, metrics, mo, plt, runs_at, space):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for _l in runs_at(epoch_select.value):
        _ev = metrics[(_l, epoch_select.value)]["eigenvalues"]
        _a1.plot(_ev, color=COLORS[_l], lw=2, label=_l)
        _a2.plot(_ev / _ev.sum(), color=COLORS[_l], lw=2, label=_l)
    _a1.axhline(1, color="0.4", lw=1, ls="--", label="N(0, I)")
    _a2.axhline(1 / 192, color="0.4", lw=1, ls="--", label="N(0, I): 1/192")
    for _a, _t in [(_a1, "Raw eigenvalues"), (_a2, "Eigenvalues / sum")]:
        _a.set_yscale("log")
        _a.set_xlabel("principal component (sorted)")
        _a.set_title(f"{_t}, {space.value}, epoch {epoch_select.value}")
        _a.legend()
    _a1.set_ylabel("variance along the component")
    _a2.set_ylabel("fraction of total variance")
    _fig.tight_layout()
    mo.vstack(
        [
            _fig,
            mo.ui.table(
                [
                    {
                        "run": _l,
                        "epoch": _epoch,
                        "effective rank": f"{_m['erank']:.1f}",
                        "participation ratio": f"{_m['pr']:.1f}",
                        "top component share": f"{_m['eigenvalues'][0] / _m['total_var']:.1%}",
                        "components for 90% of variance": _m["n90"],
                        "components for 99% of variance": _m["n99"],
                    }
                    for (_l, _epoch), _m in metrics.items()
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(gauss_ref, mo):
    mo.md(rf"""
    ## 4. Is the distribution close to N(0, I)?

    SIGReg looks at the latents along many random directions. Along each
    direction, it compares the distribution of the projected values with
    the standard normal distribution N(0, 1). The SIGReg loss is large when
    the two distributions are different.

    ### 4.1 The SIGReg statistic

    We calculate the SIGReg loss as in training: random batches of 128
    frames, 1024 random directions, averaged over the batches. All models
    use the same batches and the same random directions (fixed seed).

    **The batches must be random.** The frames in the files are sorted by
    dataset row, and nearby episodes look alike. A batch of neighboring
    rows covers only a small part of the scenes, so it looks much less
    like N(0, I) than a random batch. (An earlier version of this notebook
    did not shuffle. It gave about 14 for run A at epoch 3, compared with
    about 1.5 with random batches.)

    **How the value depends on the batch size B.** The value is about

    > SIGReg ≈ c + B × D

    - **c ≈ 1** is the noise floor. Even real N(0, I) samples give it,
      because a finite batch is never exactly Gaussian. It does not depend
      on B.
    - **D** is the real distance between the latent distribution and
      N(0, I). It does not depend on B either.

    So only the part above the floor tells you how far the latents are
    from N(0, I). A larger batch makes this part (and its gradient) larger,
    while the prediction loss does not change.

    We give two values:

    - **SIGReg (raw)**: on the latents as they are. This is the value that
      the training loss sees. Compare it with `fit/sigreg_loss` in W&B.
    - **SIGReg (standardized)**: after we shift and scale each dimension to
      mean 0 and std 1. This removes the scale problem and shows only the
      **shape** problem (for example, a few dimensions that move together).

    **Reference:** real samples from N(0, I) give **{gauss_ref:.2f}**.
    This is the lowest value that you can expect with batches of 128,
    because a finite sample is never exactly Gaussian.
    """)
    return


@app.cell
def _(gauss_ref, metrics, mo):
    mo.ui.table(
        [
            {
                "run": _l,
                "epoch": _epoch,
                "SIGReg (raw)": f"{_m['sigreg']:.3g}",
                "SIGReg (standardized)": f"{_m['sigreg_z']:.3g}",
                "N(0, I) reference": f"{gauss_ref:.3g}",
            }
            for (_l, _epoch), _m in metrics.items()
        ],
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Histograms along random directions

    We take 4 random unit directions (the same 4 for all runs) and project
    every latent onto them. Each histogram shows the projected values. The
    gray curve is the N(0, 1) density, which is what SIGReg wants.

    **All plots use the same x-axis, from −4 to 4.** This is on purpose. If
    the latents collapsed, all projected values are almost the same number,
    so the histogram is one thin spike. If we let each plot choose its own
    x-axis, the spike would be stretched and could look like a normal bell
    curve. The text in each plot gives the real std of the values.
    """)
    return


@app.cell
def _(COLORS, epoch_select, latents, np, plt, runs_at, space):
    _rng = np.random.default_rng(1)
    _dirs = _rng.standard_normal((192, 4))
    _dirs /= np.linalg.norm(_dirs, axis=0)
    _grid = np.linspace(-4, 4, 200)
    _runs = runs_at(epoch_select.value)
    _fig, _axes = plt.subplots(len(_runs), 4, figsize=(11, 2.6 * len(_runs)), sharex=True, squeeze=False)
    for _r, _l in enumerate(_runs):
        _proj = latents(_l, epoch_select.value) @ _dirs
        for _c in range(4):
            _a = _axes[_r, _c]
            _a.hist(
                _proj[:, _c], bins=np.linspace(-4, 4, 81), density=True,
                color=COLORS[_l], edgecolor=COLORS[_l], lw=1, alpha=0.8, zorder=3,
            )
            _a.plot(_grid, np.exp(-_grid**2 / 2) / np.sqrt(2 * np.pi), color="0.3", lw=1.5, zorder=4)
            _a.set_xlim(-4, 4)
            _a.text(
                0.02, 0.95, f"mean {_proj[:, _c].mean():.3f}\nstd {_proj[:, _c].std():.1e}",
                transform=_a.transAxes, va="top", fontsize=8,
            )
            if _c == 0:
                _a.set_ylabel(f"{_l}\ndensity")
            if _r == 0:
                _a.set_title(f"direction {_c + 1}")
    _fig.suptitle(f"Latents projected on random directions, {space.value}, epoch {epoch_select.value}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How to read it.** For a SIGReg run, the bars should follow the gray
    curve. For a collapsed run, all values fall into one bar, so the
    histogram is one tall, thin bar. Its height is 10, because the bar is
    0.1 wide and holds all the frames. The gray N(0, 1) curve is then small
    in comparison, because each row has its own y-axis.

    ## 5. Linear probe: is the information about the scene still there?

    Sections 1–4 show **how** the latents are distributed. They do not show
    if the latents still describe the scene. A **linear probe** answers
    this question. We fit a linear regression (ridge) from the latent to
    the true state of the scene:

    - agent x, agent y, block x, block y,
    - cos and sin of the block angle (the angle wraps around at 2π, so we
      use cos and sin instead of the angle).

    Before the regression, we **standardize** each latent dimension (mean
    0, std 1). This makes the probe blind to the scale. So even a cloud that
    is 10⁻⁶ wide can give a good result if its small differences still
    follow the state. We train on 80% of the episodes and test on the
    other 20%.

    **How to read it.** R² = 1 means a perfect prediction. R² = 0 means the
    probe is no better than the mean value.

    **Note for planning.** A high R² after standardization does not mean
    that the latent is useful for the planner. The planner compares raw
    latents with a squared distance and does no standardization. If all
    distances are 10⁻¹⁰, the cost differences between action plans are
    almost zero.
    """)
    return


@app.cell
def _(COLORS, PROBE_TARGETS, epoch_select, metrics, np, plt, runs_at, space):
    _runs = runs_at(epoch_select.value)
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _w = 0.8 / len(_runs)
    _xs = np.arange(len(PROBE_TARGETS))
    for _k, _l in enumerate(_runs):
        _r2 = metrics[(_l, epoch_select.value)]["r2"]
        _bars = _ax.bar(_xs + (_k - (len(_runs) - 1) / 2) * _w, _r2, width=_w - 0.02, color=COLORS[_l], label=_l)
        _ax.bar_label(_bars, fmt="%.2f", fontsize=7, padding=2)
    _ax.set_xticks(_xs, PROBE_TARGETS)
    _ax.set_ylim(min(0, min(metrics[(_l, epoch_select.value)]["r2"].min() for _l in _runs)) - 0.05, 1.1)
    _ax.axhline(0, color="0.3", lw=1)
    _ax.set_ylabel("test R²")
    _ax.set_title(f"Linear probe from the standardized latent, {space.value}, epoch {epoch_select.value}")
    _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(_runs), frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. PCA picture of the latents

    PCA finds the two directions with the largest variance. We project
    every latent onto these two directions and draw it as a point. We fit
    one PCA for each run, so each plot has its own axes.

    - **Top row:** the color shows the block angle. The color map is cyclic,
      because 0 and 2π are the same angle.
    - **Bottom row:** the color shows the x position of the agent.
    - **Black lines:** 3 full episodes, frame by frame.

    **How to read it.** Smooth color changes mean that the two main
    directions follow the state of the scene. **Always read the numbers on
    the axes.** PCA can make a very small cloud look as big as a large one.
    The axis labels give the share of the total variance of each
    direction.
    """)
    return


@app.cell
def _(data, epoch_select, latents, np, plt, runs_at, space):
    _runs = runs_at(epoch_select.value)
    _fig, _axes = plt.subplots(2, len(_runs), figsize=(5.2 * len(_runs), 8.4), squeeze=False)
    for _k, _l in enumerate(_runs):
        _x = latents(_l, epoch_select.value)
        _mu = _x.mean(0)
        _, _s, _vt = np.linalg.svd(_x - _mu, full_matrices=False)
        _share = _s**2 / (_s**2).sum()
        _pc = (_x - _mu) @ _vt[:2].T
        _d = data[(_l, epoch_select.value)]
        _state = _d["state"][_d["group"] == 0]
        _episode_x = latents(_l, epoch_select.value, group=1)
        _episode_id = _d["episode"][_d["group"] == 1]
        for _r, (_c, _cmap, _name) in enumerate([(_state[:, 4] % (2 * np.pi), "twilight", "block angle (rad)"), (_state[:, 0], "viridis", "agent x")]):
            _a = _axes[_r, _k]
            _sc = _a.scatter(_pc[:, 0], _pc[:, 1], c=_c, cmap=_cmap, s=3, alpha=0.6, linewidths=0)
            for _episode in np.unique(_episode_id)[:3]:
                _t = (_episode_x[_episode_id == _episode] - _mu) @ _vt[:2].T
                _a.plot(_t[:, 0], _t[:, 1], color="black", lw=0.8, alpha=0.8)
            _a.set_xlabel(f"PC1 ({_share[0]:.0%} of variance)")
            _a.set_ylabel(f"PC2 ({_share[1]:.0%} of variance)")
            _a.set_title(f"{_l}")
            _a.ticklabel_format(style="sci", scilimits=(-3, 3))
            _fig.colorbar(_sc, ax=_a, label=_name)
    _fig.suptitle(f"PCA of the latents, {space.value}, epoch {epoch_select.value}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Over epochs: when does the collapse happen?

    These plots show the main numbers for every epoch of every run, for
    the selected latent. Runs A and B have 3 epochs, and the img224 run has
    10. Each plot has one measure, so each plot has one y-axis.

    - **Total variance** (log scale): the size of the cloud (section 1).
    - **Effective rank**: the number of directions used (section 3).
    - **Directions for 99% of the variance**: a second count of the
      directions used (section 3).
    - **SIGReg (raw)** (log scale): the distance from N(0, I) (section 4).
      The dashed line is the N(0, I) reference.
    - **Mean probe R²**: the mean over the 6 targets (section 5).

    A checkpoint is saved only at the end of each epoch. So these plots
    cannot show when the collapse started inside epoch 1. Section 8 uses
    the W&B log, which has a value for each training step.
    """)
    return


@app.cell
def _(COLORS, epochs, gauss_ref, labels, metrics, plt, run_epochs, space):
    _fig, _axes = plt.subplots(1, 5, figsize=(17, 3.4))
    _panels = [
        ("total_var", "Total variance", True),
        ("erank", "Effective rank", False),
        ("n99", "Directions for 99% of variance", False),
        ("sigreg", "SIGReg (raw)", True),
        ("r2", "Mean probe R²", False),
    ]
    for _a, (_key, _title, _log) in zip(_axes, _panels):
        for _l in labels:
            _eps = run_epochs(_l)
            _vals = [metrics[(_l, _epoch)][_key] for _epoch in _eps]
            if _key == "r2":
                _vals = [v.mean() for v in _vals]
            _a.plot(_eps, _vals, marker="o", ms=8, lw=2, color=COLORS[_l], label=_l)
        if _key == "sigreg":
            _a.axhline(gauss_ref, color="0.4", lw=1, ls="--", label="N(0, I)")
        if _log:
            _a.set_yscale("log")
        _a.set_xticks(epochs)
        _a.set_xlabel("epoch")
        _a.set_title(_title)
    _axes[0].legend()
    _fig.suptitle(f"Change over epochs, {space.value}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Training curves from W&B (optional)

    During training, `fit/sigreg_loss` was logged also for the run without
    SIGReg (its weight was 0, so it had no effect on training). It is a
    collapse monitor with one value for each step. A value that goes up
    shows when the latents move away from N(0, I).

    **The value for a complete collapse is about 51.46.** If all 128
    points of a batch are at the same place near 0, the SIGReg loss is
    128 × Σₖ wₖ (1 − φ(tₖ))² ≈ 51.46. (If that place is far from 0, the
    value is higher.) So a curve that stays at about 51.4 means a complete
    collapse near 0.

    Also look at `fit/pred_loss`. Without SIGReg, a prediction loss that
    goes to almost zero is a sign of the "easy" collapsed solution.

    The 10-epoch img224 run was trained with W&B turned off, so this
    section shows only runs A and B.

    This cell reads the two curves with the W&B API. You must be logged in
    to the W&B server of the `librakevin-workday` entity. If the cell
    cannot connect, it shows the error and the rest of the notebook still
    works. You can also look at the curves on the W&B web page.
    """)
    return


@app.cell
def _(COLORS, RUNS, WANDB_RUNS, labels, mo, plt):
    def _load_wandb() -> dict:
        import wandb

        api = wandb.Api(timeout=30)
        keys = ["trainer/global_step", "fit/sigreg_loss", "fit/pred_loss"]
        return {
            _l: api.run(f"librakevin-workday/lewm/{RUNS[_l]}").history(keys=keys, samples=2000)
            for _l in WANDB_RUNS
            if _l in labels
        }

    try:
        _hist = _load_wandb()
    except Exception as _err:
        _out = mo.callout(mo.md(f"Could not read W&B: `{type(_err).__name__}: {str(_err)[:300]}`"), kind="warn")
    else:
        _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.4))
        for _a, _key in zip(_axes, ["fit/sigreg_loss", "fit/pred_loss"]):
            for _l in _hist:
                _h = _hist[_l].dropna(subset=[_key])
                _x = _h["trainer/global_step"] if "trainer/global_step" in _h else _h["_step"]
                _a.plot(_x, _h[_key], color=COLORS[_l], lw=1.5, label=_l)
            _a.set_yscale("log")
            _a.set_xlabel("training step")
            _a.set_title(_key)
            _a.legend()
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Summary

    The table puts the main numbers side by side for the selected epoch and
    latent, for each run that has this epoch. If runs A and B both have the
    epoch, the last column divides the "no SIGReg" value by the "SIGReg"
    value.
    """)
    return


@app.cell
def _(epoch_select, gauss_ref, metrics, mo, runs_at, space):
    _runs = runs_at(epoch_select.value)
    _measures = [
        ("spread (√ total variance)", lambda m: m["spread"]),
        ("|mean|", lambda m: m["mean_norm"]),
        ("effective rank (max 192)", lambda m: m["erank"]),
        ("participation ratio", lambda m: m["pr"]),
        ("directions for 99% of variance", lambda m: m["n99"]),
        (f"SIGReg raw (N(0, I): {gauss_ref:.2f})", lambda m: m["sigreg"]),
        ("SIGReg standardized", lambda m: m["sigreg_z"]),
        ("mean probe R²", lambda m: m["r2"].mean()),
    ]
    _rows = []
    for _name, _get in _measures:
        _row = {"measure": _name}
        for _l in _runs:
            _row[_l] = f"{_get(metrics[(_l, epoch_select.value)]):.3g}"
        if "SIGReg" in _runs and "no SIGReg" in _runs:
            _va, _vb = _get(metrics[("SIGReg", epoch_select.value)]), _get(metrics[("no SIGReg", epoch_select.value)])
            _row["no SIGReg / SIGReg"] = f"{_vb / _va:.2g}"
        _rows.append(_row)
    mo.vstack([mo.md(f"**{space.value}, epoch {epoch_select.value}**"), mo.ui.table(_rows, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Longer training: does the effective rank keep going up?

    With SIGReg, run A has no complete collapse, but its effective rank is
    only about 69 of 192 at epoch 3 (section 3). Is this because the
    training is short? The 10-epoch img224 run helps to answer this. The
    table below gives its numbers for each epoch, for the selected latent.
    The plots in section 7 show the same numbers as lines.

    **What to look for:**

    - **Effective rank and directions for 99% of the variance.** If they
      keep going up, the low rank of run A is partly caused by the short
      training.
    - **A plateau.** If they stop going up, more epochs do not help
      (at least with this setup).
    - **SIGReg (raw).** If it goes down while the rank goes up, the SIGReg
      loss "sees" the low rank and pushes against it. Compare it with the
      N(0, I) reference (about 1.05): only the part above the reference is
      a real distance from N(0, I) (section 4.1).

    **Results for `emb` (what this notebook showed on 2026-10-16):** the
    effective rank goes up from about 51 at epoch 1 to about 98 at epoch
    10. It stops at about epoch 8. The directions for 99% of the variance
    go from 55 to 97. The SIGReg statistic goes down at the same time, from
    about 2.3 to about 1.33, so the SIGReg loss follows the rank. At epoch
    10, the part above the N(0, I) reference is only about 0.3. The spread
    stays at about 13.8 (√192 ≈ 13.9), so there is no complete collapse at
    any epoch. The `cls` latent has a lower rank (about 36 at epoch 10).

    **Why the rank does not go higher.** A separate test on run A at epoch
    3 (not in this notebook) whitened the latents: it made all eigenvalues
    equal to 1 and kept the rest of the shape. This removed about 90% of
    the distance above the floor. So most of what is left of the SIGReg
    loss is this anisotropy. But with B = 128, it adds only about 0.45 to
    the loss, which is less than the noise floor of about 1.05. So the
    signal against the remaining anisotropy is weak. A larger batch (or a
    larger SIGReg weight) makes this signal stronger. The scene also has
    only about 5 degrees of freedom (agent x and y, block x and y, block
    angle), which can limit how many directions the encoder can fill.

    **Be careful with these comparisons:**

    - **Learning-rate schedule.** The scheduler is a linear warmup and a
      cosine decay over **all** training steps, to 0 at the end
      (`LinearWarmupCosineAnnealingLR`, stepped once per step). So in run A,
      the learning rate is almost 0 at the end of epoch 3. In the 10-epoch
      run, it is still about 80% of the peak at the end of epoch 3. This
      is one reason why run A stops at about 69. It also means that the
      plateau at epochs 8–10 can come from the learning rate that goes to
      0, and not from a real limit.
    - **Image size.** The 10-epoch run uses 224-pixel images, and run A
      uses 112. The image size and the training length change together.
    - **One seed.** Each run is one training run.
    """)
    return


@app.cell
def _(COLORS, gauss_ref, metrics, mo, run_epochs):
    _l = "SIGReg 10 epochs (img224)"
    if (_l, 1) not in metrics:
        _out = mo.callout(mo.md("No embeddings for the 10-epoch run. Run the second command in the introduction."), kind="warn")
    else:
        _out = mo.vstack(
            [
                mo.md(f"<span style='color:{COLORS[_l]}'>■</span> **{_l}**, N(0, I) SIGReg reference {gauss_ref:.2f}"),
                mo.ui.table(
                    [
                        {
                            "epoch": _epoch,
                            "spread": f"{metrics[(_l, _epoch)]['spread']:.3g}",
                            "effective rank": f"{metrics[(_l, _epoch)]['erank']:.1f}",
                            "participation ratio": f"{metrics[(_l, _epoch)]['pr']:.1f}",
                            "directions for 99%": metrics[(_l, _epoch)]["n99"],
                            "SIGReg (raw)": f"{metrics[(_l, _epoch)]['sigreg']:.3g}",
                            "mean probe R²": f"{metrics[(_l, _epoch)]['r2'].mean():.3f}",
                        }
                        for _epoch in run_epochs(_l)
                    ],
                    selection=None,
                    pagination=False,
                ),
            ]
        )
    _out
    return


if __name__ == "__main__":
    app.run()
