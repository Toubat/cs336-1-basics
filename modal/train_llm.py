import modal

app = modal.App("train-llm")


image = modal.Image.debian_slim().uv_sync().add_local_python_source("cs336_basics")


source_volume = modal.Volume.from_name(
    "cs336",
    version=2,
    create_if_missing=True,
)


with image.imports():
    from pathlib import Path

    from cs336_basics.train import ModelConfig, TrainingConfig
    from cs336_basics.train import train as train_llm


@app.function(
    image=image,
    volumes={"/source": source_volume},
    gpu="H100",
    cpu=32,
    secrets=[modal.Secret.from_name("wandb")],
)
def train():
    config = TrainingConfig(
        name="owt-cuda",
        volume_path=Path("/source/cs336-1-basics/"),
        dataset="owt",
        model=ModelConfig(),
    )
    train_llm(config)


@app.local_entrypoint()
def main():
    train.remote()
