"""Print the shape of every tensor in one LeWM training forward pass.

Builds the model from config/train/model/lewm.yaml with random weights, runs the
same steps as lejepa_forward (train.py) on a small random batch on CPU, and
prints one line per module output. Used to check the shapes in lewm_training.html.

    python docs/viz/trace_shapes.py [batch_size]
"""

import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra import compose, initialize_config_dir
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jepa import JEPA  # noqa: E402
from module import SIGReg  # noqa: E402

ACTION_DIM = 2  # PushT raw action size


def shape_of(x: Any) -> str:
    if torch.is_tensor(x):
        return str(tuple(x.shape))
    if isinstance(x, (tuple, list)):
        return ", ".join(shape_of(v) for v in x)
    if hasattr(x, "last_hidden_state"):
        return f"last_hidden_state={shape_of(x.last_hidden_state)}"
    return type(x).__name__


def add_hooks(model: nn.Module, rows: list[tuple[str, str, str]]) -> None:
    """Record input/output shapes of every module (first call only)."""
    seen: set[str] = set()

    def make_hook(name: str):
        def hook(mod: nn.Module, args: tuple[Any, ...], out: Any) -> None:
            if name in seen:
                return
            seen.add(name)
            rows.append((name, shape_of(args), shape_of(out)))

        return hook

    for name, mod in model.named_modules():
        # skip repeated layers after the first one to keep the table short
        parts = name.split(".")
        if any(p.isdigit() and p != "0" and parts[i - 1] in ("layer", "layers") for i, p in enumerate(parts)):
            continue
        mod.register_forward_hook(make_hook(name or "<model>"))


def main() -> None:
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with initialize_config_dir(config_dir=str(ROOT / "config/train"), version_base=None):
        cfg = compose(config_name="lewm", overrides=["model.action_encoder.input_dim=10"])
    frameskip = cfg.data.dataset.frameskip
    t = cfg.history_size + cfg.num_preds
    img = cfg.img_size

    model: JEPA = hydra.utils.instantiate(cfg.model)
    model.eval()
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs)
    rows: list[tuple[str, str, str]] = []
    add_hooks(model, rows)

    batch = {
        "pixels": torch.randn(b, t, 3, img, img),
        "action": torch.randn(b, t, frameskip * ACTION_DIM),
    }
    with torch.no_grad():
        out = model.encode(batch)
        emb, act_emb = out["emb"], out["act_emb"]
        ctx_emb, ctx_act = emb[:, : cfg.history_size], act_emb[:, : cfg.history_size]
        tgt_emb = emb[:, cfg.num_preds :]
        pred_emb = model.predict(ctx_emb, ctx_act)
        pred_loss = (pred_emb - tgt_emb).pow(2).mean()

        proj = emb.transpose(0, 1)
        a = torch.randn(proj.size(-1), sigreg.num_proj)
        x_t = (proj @ a).unsqueeze(-1) * sigreg.t
        err = (x_t.cos().mean(-3) - sigreg.phi).square() + x_t.sin().mean(-3).square()
        stat = (err @ sigreg.weights) * proj.size(-2)
        sigreg_loss = sigreg(proj)

    print(f"{'module':60s} {'input':40s} output")
    for name, i, o in rows:
        print(f"{name:60s} {i:40s} {o}")

    print("\n-- lejepa_forward / SIGReg tensors --")
    named: dict[str, torch.Tensor] = {
        "batch.pixels": batch["pixels"],
        "batch.action": batch["action"],
        "emb": emb,
        "act_emb": act_emb,
        "ctx_emb": ctx_emb,
        "ctx_act": ctx_act,
        "tgt_emb": tgt_emb,
        "pred_emb": pred_emb,
        "pred_loss": pred_loss,
        "sigreg.proj (emb.T)": proj,
        "sigreg.A": a,
        "sigreg.x_t": x_t,
        "sigreg.err": err,
        "sigreg.statistic": stat,
        "sigreg_loss": sigreg_loss,
    }
    for k, v in named.items():
        print(f"{k:30s} {tuple(v.shape)}")


if __name__ == "__main__":
    main()
