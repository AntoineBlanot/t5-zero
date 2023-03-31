import torch
from torch import Tensor, nn

from transformers import T5EncoderModel


class T5Encoder(nn.Module):

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.model = T5EncoderModel.from_pretrained(name)

    def forward(self, *args, **kwargs) -> Tensor:
        last_hidden_state = self.model.forward(*args, **kwargs).last_hidden_state
        return last_hidden_state


class NLIHead(nn.Module):

    def __init__(self, n_class: int, d_model: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_class = n_class
        self.layer = nn.Linear(d_model, n_class)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)
