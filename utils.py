import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_pretraining import data as dt
import lightning as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig

def get_img_preprocessor(source: str, target: str, img_size: int = 224) -> dt.transforms.Compose:
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset: Any, source: str, target: str) -> dt.transforms.WrapTorchTransform:
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name: str, cfg: DictConfig, epoch_interval: int = 1) -> None:
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model: torch.nn.Module, epoch: int) -> None:
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch:03d}.pt',
        )


class ResumeCkptCallback(Callback):
    """Save the full training state (optimizer, scheduler, loop counters) to a fixed path after each epoch.

    stable-pretraining's Manager redirects every ModelCheckpoint into its own cache dir, so this writes
    the file train.py resumes from directly, keeping the whole run in one folder."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = str(path)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # called on all ranks; Lightning writes from rank zero only
        trainer.save_checkpoint(self.path)


class BucketSyncCallback(Callback):
    """Mirror the local run folder to a HF bucket every N epochs (in the background) and when training stops."""

    def __init__(self, local_dir: str | Path, bucket_uri: str, every_n_epochs: int = 5) -> None:
        super().__init__()
        self.local_dir = str(local_dir)
        self.bucket_uri = bucket_uri
        self.every_n_epochs = every_n_epochs
        self._thread: threading.Thread | None = None

    def _sync(self) -> None:
        from huggingface_hub import sync_bucket
        try:
            # *.tmp: half-written checkpoints from the atomic-save plugin
            sync_bucket(self.local_dir, self.bucket_uri, exclude=["*.tmp"], quiet=True)
            logging.info(f"Synced {self.local_dir} to {self.bucket_uri}")
        except Exception as e:
            logging.warning(f"Bucket sync to {self.bucket_uri} failed: {e}")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # ModelCheckpoint runs last at epoch end, so the previous epoch's files are complete by now
        epoch = trainer.current_epoch
        if not trainer.is_global_zero or epoch == 0 or epoch % self.every_n_epochs:
            return
        if self._thread is not None and self._thread.is_alive():
            logging.warning("Previous bucket sync still running, skipping this one")
            return
        self._thread = threading.Thread(target=self._sync, daemon=True)
        self._thread.start()

    def _final_sync(self, trainer: pl.Trainer) -> None:
        if not trainer.is_global_zero:
            return
        if self._thread is not None:
            self._thread.join()
        self._sync()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._final_sync(trainer)

    def on_exception(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException
    ) -> None:
        self._final_sync(trainer)
