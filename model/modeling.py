from torch import Tensor, nn
from transformers import BartConfig, BartForSequenceClassification

from model.encoder import T5Encoder, BertEncoder
from model.pooling import Pooling


class PretrainedBARTClassif(nn.Module):

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder_decoder = BartForSequenceClassification.from_pretrained(name)

    def forward(self, *args, **kwargs):
        outputs = self.encoder_decoder(*args, **kwargs)
        
        return dict(
            logits=outputs.logits,
            hidden_states=outputs.decoder_hidden_states
        )
    

class T5Classif(nn.Module):

    def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = T5Encoder(name=name)
        self.pool_layer = Pooling(self.encoder.module.config.d_model, pooling_mode_mean_tokens=True)
        self.head = nn.Linear(self.encoder.module.config.d_model, n_class)

    def forward(self, *args, **kwargs) -> dict:
        # Encoder
        encoder_outputs = self.encoder(*args, **kwargs)
        # Pooling layer (no [CLS] token)
        pool_inputs = dict(token_embeddings=encoder_outputs, *args, **kwargs)
        pooled_outputs = self.pool_layer(pool_inputs)['sentence_embedding']
        # Head
        head_outputs = self.head(pooled_outputs)

        return dict(
            logits=head_outputs,
            hidden_states=encoder_outputs
        )
    

class BERTClassif(nn.Module):

    def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = BertEncoder(name=name)
        self.head = nn.Linear(self.encoder.module.config.hidden_size, n_class)

    def forward(self, *args, **kwargs) -> dict:
        # Encoder
        encoder_outputs = self.encoder(*args, **kwargs)
        # Poolinng embeddings ([CLS] token)
        pooled_outputs = encoder_outputs[:, 0, :]
        head_outputs = self.head(pooled_outputs)

        return dict(
            logits=head_outputs,
            hidden_states=encoder_outputs
        )


class BinaryBERTClassif(nn.Module):

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = BertEncoder(name=name)
        self.head = nn.Linear(self.encoder.module.config.d_model, 1)

    def forward(self, *args, **kwargs) -> dict:
        # Encoder
        encoder_outputs = self.encoder(*args, **kwargs)
        # Poolinng embeddings ([CLS] token)
        pooled_outputs = encoder_outputs[:, 0, :]
        head_outputs = self.head(pooled_outputs)

        return dict(
            logits=head_outputs,
            hidden_states=encoder_outputs
        )
    

class BARTClassif(nn.Module):

    def __init__(self, name: str, id2label: dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        label2id = {v: int(k) for k,v in id2label.items()}
        config = BartConfig(id2label=id2label, label2id=label2id, num_labels=len(id2label))
        self.encoder_decoder = BartForSequenceClassification.from_pretrained(name, config=config)

    def forward(self, *args, **kwargs):
        outputs = self.encoder_decoder(*args, **kwargs)
        
        return dict(
            logits=outputs.logits,
            hidden_states=outputs.decoder_hidden_states
        )
