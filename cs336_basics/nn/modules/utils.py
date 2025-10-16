from einops import reduce
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    x = x.swapdims(-1, dim)
    x_max = reduce(x, "... d -> ... 1", reduction="max")

    x = (x - x_max).exp()
    x_sum = reduce(x, "... d -> ... 1", reduction="sum")
    x = x / x_sum

    return x.swapdims(-1, dim)
