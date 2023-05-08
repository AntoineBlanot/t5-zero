from typing import Dict
import torch
from torch import Tensor, nn
from transformers import (
    AutoConfig, AutoModelForSeq2SeqLM,
    T5Model,
    BertForSequenceClassification, RobertaForSequenceClassification, BartForSequenceClassification
)

from model.encoder import T5Encoder
from model.pooling import Pooling
from model.head import ClassificationHead


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

class PretrainedRobertaClassif(nn.Module):

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder_decoder = RobertaForSequenceClassification.from_pretrained(name)

    def forward(self, *args, **kwargs):
        outputs = self.encoder_decoder(*args, **kwargs)
        
        return dict(
            logits=outputs.logits,
            hidden_states=outputs.hidden_states
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
        self.head = ClassificationHead(
            hidden_size=self.encoder.module.config.d_model,
            out_size=n_class,
            dropout=self.encoder.module.config.dropout_rate
        )

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
        config = AutoConfig.from_pretrained(name)
        self.__update_config(base_config=config, n_class=n_class)

        BertForSequenceClassification._keys_to_ignore_on_load_unexpected = ["cls.*"]
        self.model = BertForSequenceClassification.from_pretrained(name, config=config)

    def __update_config(self, base_config, n_class):
        num_labels = n_class
        label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
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


class RoBertaClassif(nn.Module):

    def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = AutoConfig.from_pretrained(name)
        self.__update_config(base_config=config, n_class=n_class)

        RobertaForSequenceClassification._keys_to_ignore_on_load_unexpected = ["lm_head.*"]
        self.model = RobertaForSequenceClassification.from_pretrained(name, config=config)

    def __update_config(self, base_config, n_class):
        num_labels = n_class
        label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
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

class RoBertaBinaryClassif(nn.Module):

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = AutoConfig.from_pretrained(name)
        self.__update_config(base_config=config, n_class=1)

        RobertaForSequenceClassification._keys_to_ignore_on_load_unexpected = ["lm_head.*"]
        self.model = RobertaForSequenceClassification.from_pretrained(name, config=config)

    def __update_config(self, base_config, n_class):
        num_labels = n_class
        
        base_config.update(dict(
            num_labels=num_labels
        ))

    def forward(self, *args, **kwargs) -> dict:
        outputs = self.model(*args, **kwargs)

        return dict(
            logits=outputs.logits.squeeze(),
            hidden_states=outputs.hidden_states
        )
      

class BARTClassif(nn.Module):

    def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = AutoConfig.from_pretrained(name)
        self.__update_config(base_config=config, n_class=n_class)

        self.model = BartForSequenceClassification.from_pretrained(name, config=config)
    
    def __update_config(self, base_config, n_class):
        num_labels = n_class
        label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
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
            hidden_states=outputs.decoder_hidden_states
        )

class BARTBinaryClassif(nn.Module):

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = AutoConfig.from_pretrained(name)
        self.__update_config(base_config=config, n_class=1)

        self.model = BartForSequenceClassification.from_pretrained(name, config=config)
    
    def __update_config(self, base_config, n_class):
        num_labels = n_class

        base_config.update(dict(
            num_labels=num_labels
        ))

    def forward(self, *args, **kwargs) -> dict:
        outputs = self.model(*args, **kwargs)

        return dict(
            logits=outputs.logits.squeeze(),
            hidden_states=outputs.decoder_hidden_states
        )
    
class T5ModelForClassification(nn.Module):

    def __init__(self, config_dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.t5 = T5Model.from_pretrained(**config_dict)
        self.head = nn.Linear(self.t5.config.d_model, 3)

        print('Model parameters: {}M ({}M trainable)'.format(
            int(self.__count_parameters()/1e6), int(self.__count_parameters(True)/1e6)
        ))

    def __count_parameters(self, trainable: bool = False):
        if trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def forward(self,
        input_ids: Tensor, attention_mask: Tensor,
        decoder_input_ids: Tensor, decoder_attention_mask: Tensor,
        labels: Tensor = None
    ) -> Dict[str, Tensor]:
        
        last_hidden_state = self.t5(
            input_ids=input_ids, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask
        ).last_hidden_state
        outputs = self.head(last_hidden_state[:, 0, :])

        return dict(
            logits=outputs
        )