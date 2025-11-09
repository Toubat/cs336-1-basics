import os
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    N, M = dataset.shape[0], context_length

    starts = np.random.randint(0, N - M, batch_size)
    indices = np.zeros((batch_size, M + 1), dtype=np.int32)

    for row in range(batch_size):
        indices[row, :] = np.arange(starts[row], starts[row] + M + 1)

    batch = torch.from_numpy(dataset[indices]).to(device)

    return batch[:, :-1], batch[:, 1:]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    return torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
