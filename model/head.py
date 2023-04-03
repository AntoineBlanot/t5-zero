import torch
from torch import Tensor, nn


class NLIHead(nn.Module):

    def __init__(self, n_class: int, d_model: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_class = n_class
        self.layer = nn.Linear(d_model, n_class)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)