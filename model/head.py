import torch
from torch import Tensor, nn


class ClassificationHead(nn.Module):

    def __init__(self, hidden_size: int, out_size: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, out_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense(self.dropout1(x))
        x = self.tanh(x)
        x = self.out_proj(self.dropout2(x))
        return x