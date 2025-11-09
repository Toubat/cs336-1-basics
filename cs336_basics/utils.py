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
