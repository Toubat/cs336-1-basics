import hashlib
import os
from pathlib import Path
from typing import Literal

import chz
import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from tqdm import tqdm

import wandb
from cs336_basics.loss import CrossEntropyLoss
from cs336_basics.nn.modules.utils import softmax
from cs336_basics.optim import AdamW, CosineAnnealingLRScheduler
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.utils import get_batch, load_checkpoint

Dataset = Literal["tinystories", "owt"]

DATASET_TRAIN_PATHS: dict[Dataset, str] = {
    "tinystories": "data/tinystories_gpt4_train.npy",
    "owt": "data/owt_train.npy",
}

DATASET_VALID_PATHS: dict[Dataset, str] = {
    "tinystories": "data/tinystories_gpt4_valid.npy",
    "owt": "data/owt_valid.npy",
}

TOTAL_TOKENS: dict[Dataset, int] = {
    "tinystories": 541000000,
    "owt": -1,
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
    epochs: int = 5000
    batch_size: int = 128
    lr_max: float = 5e-2
    lr_min: float = 5e-4
    warmup_t: int = 5000
    cosine_cycle_t: int = 50000
    model: ModelConfig
    valid_interval: int = 100
    valid_steps: int = 10

    @chz.init_property
    def wandb_id(self) -> str:
        return hashlib.sha256(self.name.encode()).hexdigest()

    @chz.init_property
    def wandb_config(self) -> dict:
        return chz.asdict(self)

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


def get_dataset(config: TrainingConfig, mode: Literal["train", "valid"]) -> npt.NDArray:
    path = DATASET_TRAIN_PATHS[config.dataset] if mode == "train" else DATASET_VALID_PATHS[config.dataset]
    return np.lib.format.open_memmap(path, mode="r", dtype=np.uint16)


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

    train, valid = get_dataset(config, "train"), get_dataset(config, "valid")

    logger.info(f"Total model size: {sum(p.numel() for p in model.parameters()) / 1024**2:.2f} MB")
    logger.info(f"Initial learning rate: {lr_scheduler.get_last_lr()}")
    logger.info(f"Train dataset size: {train.shape[0]}")
    logger.info(f"Valid dataset size: {valid.shape[0]}")

    criterion = CrossEntropyLoss()
    with wandb.init(
        entity="yoasobyin-n-a",
        project="cs336",
        id=config.wandb_id,
        name=config.name,
        config=config.wandb_config,
    ) as run:
        run.watch(model, log="all", log_freq=100)

        model.train()
        for step in tqdm(range(t0 + 1, config.epochs), desc="Steps", total=config.epochs - t0 - 1):
            optimizer.zero_grad()

            batch_X, batch_y = get_batch(train, config.batch_size, config.model.context_length, device)
            if step == t0 + 1:
                logger.info(f"Batch size: {batch_X.shape}")

            logits: torch.Tensor = model(batch_X)  # (bs, seq_len, vocab_size)
            loss: torch.Tensor = criterion(logits, batch_y)
            train_accuracy = compute_accuracy(logits, batch_y)
            run.log(
                {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0], "train_accuracy (%)": train_accuracy},
                step=step,
            )

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if step % config.valid_interval == 0:
                valid_loss, valid_accuracy = run_evaluation(model, valid, config, device)
                run.log({"valid_loss": valid_loss, "valid_accuracy (%)": valid_accuracy}, step=step)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    actual = softmax(logits, dim=-1).argmax(dim=-1)
    correct = (actual == targets).sum().item()
    return correct / targets.numel()


def run_evaluation(
    model: TransformerLM, valid: npt.NDArray, config: TrainingConfig, device: str
) -> tuple[float, float]:
    """Validate the model on the validation dataset."""
    model.eval()

    with torch.no_grad():
        valid_loss = 0.0
        valid_accuracy = 0.0

        for _ in range(config.valid_steps):
            batch_X, batch_y = get_batch(
                valid, config.batch_size, config.model.context_length, device
            )  # both are (bs, seq_len)
            valid_logits: torch.Tensor = model(batch_X)  # (bs, seq_len, vocab_size)
            valid_loss += CrossEntropyLoss()(valid_logits, batch_y).item()
            valid_accuracy += compute_accuracy(valid_logits, batch_y)

    model.train()
    return valid_loss / config.valid_steps, valid_accuracy / config.valid_steps


if __name__ == "__main__":
    chz.nested_entrypoint(main)
