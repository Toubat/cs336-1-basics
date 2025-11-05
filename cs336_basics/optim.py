from collections.abc import Callable
from typing import cast

import torch
from torch.optim.optimizer import ParamsT


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults=defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = cast(float, group["lr"])
            beta1, beta2 = cast(tuple[float, float], group["betas"])
            eps = cast(float, group["eps"])
            weight_decay = cast(float, group["weight_decay"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                m = self.state[p].get("m", 0)
                v = self.state[p].get("v", 0)
                t = self.state[p].get("t", 1)

                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2

                lr_t = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                p.data -= lr_t * m / ((v**0.5) + eps)
                p.data -= lr * weight_decay * p.data

                self.state[p]["m"] = m
                self.state[p]["v"] = v
                self.state[p]["t"] = t + 1

        return loss
