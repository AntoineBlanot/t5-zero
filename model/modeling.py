import torch
from torch import Tensor, nn

from model.encoder import T5Encoder, BertEncoder
from model.head import NLIHead
from model.pooling import Pooling

class MyModel(nn.Module):

    def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = T5Encoder(name=name)
        self.pooling = Pooling(self.encoder.model.config.d_model, pooling_mode_mean_tokens=True)
        self.head = NLIHead(n_class=n_class, d_model=self.encoder.model.config.d_model)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_outputs = self.pooling(dict(token_embeddings=encoder_outputs, attention_mask=attention_mask))['sentence_embedding']
        head_outputs = self.head(pooled_outputs)

        outputs_dict = dict(encoder_outputs=encoder_outputs, pooled_outputs=pooled_outputs, head_outputs=head_outputs)
        return outputs_dict
    

class BERTClassification(nn.Module):

    def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = BertEncoder(name=name)
        self.head = NLIHead(n_class=n_class, d_model=self.encoder.model.config.hidden_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_outputs = encoder_outputs[:, 0, :]   # [CLS] token
        head_outputs = self.head(pooled_outputs)

        outputs_dict = dict(encoder_outputs=encoder_outputs, pooled_outputs=pooled_outputs, head_outputs=head_outputs)
        return outputs_dict