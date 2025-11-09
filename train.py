import os
from pathlib import Path
from typing import Literal

import chz
import torch
from loguru import logger

from cs336_basics.optim import AdamW, CosineAnnealingLRScheduler
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.utils import load_checkpoint

Dataset = Literal["tinystories", "owt"]

DATASET_PATHS: dict[Dataset, Path] = {
    "tinystories": Path("data/TinyStoriesV2-GPT4-train.txt"),
    "owt": Path("data/owt_train.txt"),
}

CHECKPOINT_ROOT = Path(".checkpoints")


@chz.chz(typecheck=True)
class ModelConfig:
    vocab_size: int = 32000
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 1344
    theta: float = 10000.0
    num_layers: int = 4
    num_heads: int = 16


@chz.chz(typecheck=True)
class TrainingConfig:
    name: str
    dataset: Dataset = "tinystories"
    epochs: int = 10
    batch_size: int = 128
    lr_max: float = 5e-2
    lr_min: float = 5e-4
    warmup_t: int = 5000
    cosine_cycle_t: int = 50000
    model: ModelConfig

    @chz.init_property
    def dataset_path(self) -> Path:
        return DATASET_PATHS[self.dataset]

    @chz.init_property
    def checkpoint_dir(self) -> Path:
        return CHECKPOINT_ROOT / self.name

    @chz.init_property
    def checkpoint_data_path(self) -> Path | None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        ckpt_files = list(self.checkpoint_dir.glob("*.pt"))
        if not ckpt_files:
            return None

        numeric_ckpts = []
        for f in ckpt_files:
            try:
                numeric_ckpts.append((int(f.stem), f))
            except ValueError:
                logger.warning(f"Ignoring non-numeric checkpoint file: {f.name}")
                continue

        if not numeric_ckpts:
            logger.warning("No valid numeric checkpoint files found")
            return None

        ckpt_file = max(numeric_ckpts, key=lambda x: x[0])[1]
        logger.info(f"Loading checkpoint from {ckpt_file.name}")
        return ckpt_file


# run: wandb.Run = wandb.init(
#     entity="yoasobyin-n-a",
#     project="cs336",
#     config={
#         "learning_rate": 0.02,
#         "architecture": "CNN",
#         "dataset": "CIFAR-100",
#         "epochs": 10,
#     },
# )

# # Simulate training.
# epochs = 10
# offset = random.random() / 5
# for epoch in range(2, epochs):
#     acc = 1 - 2**-epoch - random.random() / epoch - offset
#     loss = 2**-epoch + random.random() / epoch + offset

#     run.log({"acc": acc, "loss": loss})

# run.finish()


def main(config: TrainingConfig):
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = TransformerLM(
        vocab_size=config.model.vocab_size,
        context_length=config.model.context_length,
        num_layers=config.model.num_layers,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        theta=config.model.theta,
    )
    model.to(device)

    optimizer = AdamW(model.parameters())

    if config.checkpoint_data_path is not None:
        logger.info(f"Loading checkpoint from {config.checkpoint_data_path.name}")
        t0 = load_checkpoint(config.checkpoint_data_path, model, optimizer)
    else:
        t0 = -1
        logger.info("No checkpoint found, starting from scratch")

    lr_scheduler = CosineAnnealingLRScheduler(
        optimizer,
        t_0=t0,
        lr_max=config.lr_max,
        lr_min=config.lr_min,
        warmup_t=config.warmup_t,
        cosine_cycle_t=config.cosine_cycle_t,
    )

    logger.info(f"Total model size: {sum(p.numel() for p in model.parameters()) / 1024**2:.2f} MB")
    logger.info(f"Initial learning rate: {lr_scheduler.get_last_lr()}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
