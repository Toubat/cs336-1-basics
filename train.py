import chz
import torch
from loguru import logger

from cs336_basics.optim import AdamW, CosineAnnealingLRScheduler
from cs336_basics.transformer_lm import TransformerLM


@chz.chz(typecheck=True)
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 1344
    theta: float = 10000.0
    num_layers: int = 4
    num_heads: int = 16


@chz.chz(typecheck=True)
class TrainingConfig:
    checkpoint_dir: str = ".checkpoints"
    lr_max: float = 0.002
    lr_min: float = 0.0002
    warmup_t: int = 1000
    cosine_cycle_t: int = 10000
    epochs: int = 10
    batch_size: int = 128
    model: ModelConfig


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
    ).to(device)

    # log total model size
    logger.info(f"Total model size: {sum(p.numel() for p in model.parameters()) / 1024**2:.2f} MB")

    optimizer = AdamW(model.parameters())

    lr_scheduler = CosineAnnealingLRScheduler(
        optimizer,
        t_0=-1,
        lr_max=config.lr_max,
        lr_min=config.lr_min,
        warmup_t=config.warmup_t,
        cosine_cycle_t=config.cosine_cycle_t,
    )

    logger.info(f"Initial learning rate: {lr_scheduler.get_last_lr()}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
