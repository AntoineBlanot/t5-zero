import torch
from torch import Tensor, nn
from transformers import AutoConfig, BartConfig, BartForSequenceClassification, AutoModelForSeq2SeqLM, BertForSequenceClassification, RobertaForSequenceClassification

from model.encoder import T5Encoder, BertEncoder, RoBertaEncoder
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
    

class PretrainedUniEvalClassif(nn.Module):

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder_decoder = AutoModelForSeq2SeqLM.from_pretrained(name)

    def forward(self, *args, **kwargs):
        B, L = kwargs['input_ids'].shape
        decoder_input_ids = torch.full((B, 1), self.encoder_decoder.config.decoder_start_token_id).long().to(self.encoder_decoder.device)
        outputs = self.encoder_decoder(*args, decoder_input_ids=decoder_input_ids, **kwargs)

        return dict(
            logits=outputs.logits[:, -1, :],        # last element of sequence
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
    

# class BERTClassif(nn.Module):

#     def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.encoder = BertEncoder(name=name)
#         self.head = nn.Linear(self.encoder.module.config.hidden_size, n_class)

#     def forward(self, *args, **kwargs) -> dict:
#         # Encoder
#         encoder_outputs = self.encoder(*args, **kwargs)
#         # Poolinng embeddings ([CLS] token)
#         pooled_outputs = encoder_outputs[:, 0, :]
#         head_outputs = self.head(pooled_outputs)

#         return dict(
#             logits=head_outputs,
#             hidden_states=encoder_outputs
#         )


class BERTClassif(nn.Module):

    def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = AutoConfig.from_pretrained(name)
        self.__update_config(base_config=config, n_class=n_class)

        BertForSequenceClassification._keys_to_ignore_on_load_unexpected = ["cls.*"]
        self.model = BertForSequenceClassification.from_pretrained(name, config=config)

    def __update_config(self, base_config, n_class):
        num_labels = n_class
        label2id = { 'entailment': 0, 'neutral': 1, 'contradiction': 2}
        id2label = {v: k for k,v in label2id.items()}
        
        base_config.update(dict(
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label
        ))

    def forward(self, *args, **kwargs) -> dict:
        outputs = self.model(*args, **kwargs)

        return dict(
            logits=outputs.logits,
            hidden_states=outputs.hidden_states
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


class RoBertaClassif(nn.Module):

    def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = AutoConfig.from_pretrained(name)
        self.__update_config(base_config=config, n_class=n_class)

        BertForSequenceClassification._keys_to_ignore_on_load_unexpected = ["cls.*"]
        self.model = RobertaForSequenceClassification.from_pretrained(name)

    def __update_config(self, base_config, n_class):
        num_labels = n_class
        label2id = { 'entailment': 0, 'neutral': 1, 'contradiction': 2}
        id2label = {v: k for k,v in label2id.items()}
        
        base_config.update(dict(
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label
        ))

    def forward(self, *args, **kwargs) -> dict:
        outputs = self.model(*args, **kwargs)

        return dict(
            logits=outputs.logits,
            hidden_states=outputs.hidden_states
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
