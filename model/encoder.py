from torch import Tensor, nn

from transformers import T5EncoderModel, BertModel


class T5Encoder(nn.Module):

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.module = T5EncoderModel.from_pretrained(name)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.module(*args, **kwargs).last_hidden_state


class BertEncoder(nn.Module):

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        BertModel._keys_to_ignore_on_load_unexpected = ["cls.*"]
        self.module = BertModel.from_pretrained(name)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.module(*args, **kwargs).last_hidden_state