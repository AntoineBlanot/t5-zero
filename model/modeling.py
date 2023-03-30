import inspect

import torch
from torch import Tensor, nn

from model.encoder import T5Encoder, NLIHead

class MyModel(nn.Module):

    def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = T5Encoder(name=name)
        self.head = NLIHead(n_class=n_class, d_model=self.encoder.model.config.d_model)

    def forward(self, *args, **kwargs) -> dict:
        encoder_outputs, pooled_outputs = self.encoder(*args, **kwargs)
        head_outputs = self.head(pooled_outputs)

        outputs_dict = dict(encoder_outputs=encoder_outputs, pooled_outputs=pooled_outputs, head_outputs=head_outputs)
        return outputs_dict