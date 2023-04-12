import torch
from torch import Tensor, nn

from transformers import (
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    BartConfig, BartForSequenceClassification
)

from model.encoder import T5Encoder, BertEncoder
from model.head import NLIHead
from model.pooling import Pooling, AttentionPooling


class T5Classification(nn.Module):

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


class T5ClassificationAttentionPooling(nn.Module):

    def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = T5Encoder(name=name)
        self.pooling = AttentionPooling(n_query=n_class, d_model=self.encoder.model.config.d_model, nhead=self.encoder.model.config.num_heads, dim_feedforward=self.encoder.model.config.d_ff, dropout=self.encoder.model.config.dropout_rate)
        self.linear = nn.Linear(self.encoder.model.config.d_model, 1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_outputs = self.pooling(**dict(x=encoder_outputs, mask=~attention_mask.bool()))
        head_outputs = self.linear(pooled_outputs).squeeze()
        # print(encoder_outputs.shape, pooled_outputs.shape, head_outputs.shape)

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


class BinaryBERTClassification(nn.Module):

    def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = BertEncoder(name=name)
        self.head = NLIHead(n_class=n_class, d_model=self.encoder.model.config.hidden_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_outputs = encoder_outputs[:, 0, :]   # [CLS] token
        head_outputs = self.head(pooled_outputs).squeeze()

        outputs_dict = dict(encoder_outputs=encoder_outputs, pooled_outputs=pooled_outputs, head_outputs=head_outputs)
        return outputs_dict
    

class BARTClassification(nn.Module):

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        id2label = {'0': 'entailment', '1': 'neutral', '2': 'contradiction'}
        label2id = {v: int(k) for k,v in id2label.items()}
        num_labels = len(id2label)
        config = BartConfig(id2label=id2label, label2id=label2id, num_labels=num_labels)
        self.encoder_decoder = BartForSequenceClassification.from_pretrained(name, config=config)

    def forward(self, *args, **kwargs):
        outputs = self.encoder_decoder(*args, **kwargs).logits
        outputs_dict = dict(head_outputs=outputs)
        return outputs_dict


class ZeroShotModel(nn.Module):

    def __init__(self, model: nn.Module, true_id: int, false_id: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.true_id = true_id
        self.false_id = false_id

    def forward(self, input_ids: Tensor, attention_mask: Tensor, **kwargs) -> Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)['head_outputs'].squeeze()
        outputs = outputs[:, [self.true_id, self.false_id]]

        outputs_dict = dict(head_outputs=outputs)
        return outputs_dict


class BARTMNLI(ZeroShotModel):

    def __init__(self, name: str, *args, **kwargs) -> None:
        model = AutoModelForSequenceClassification.from_pretrained(name)
        contra_id, neutral_id, entail_id = 0, 1, 2
        super().__init__(model=model, true_id=entail_id, false_id=contra_id, *args, **kwargs)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        model_outputs = super().forward(input_ids, attention_mask)
        output_dict = dict(head_outputs=model_outputs)
        return output_dict


class UniEval(ZeroShotModel):

    def __init__(self, name: str, *args, **kwargs) -> None:
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        yes_id, no_id = 2163, 465
        super().__init__(model=model, true_id=yes_id, false_id=no_id, *args, **kwargs)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        B, L = input_ids.shape
        decoder_input_ids = torch.full((B, 1), self.model.config.pad_token_id).long().to(self.model.device)
        
        model_outputs = super().forward(input_ids, attention_mask, decoder_input_ids=decoder_input_ids)
        output_dict = dict(head_outputs=model_outputs)
        return output_dict
    

class BERTMNLI(ZeroShotModel):

    def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
        model = BERTClassification(name=name, n_class=n_class, *args, **kwargs)
        contra_id, neutral_id, entail_id = 2, 1, 0
        super().__init__(model=model, true_id=entail_id, false_id=contra_id, *args, **kwargs)


class T5MNLI(ZeroShotModel):

    def __init__(self, name: str, n_class: int, *args, **kwargs) -> None:
        model = T5Classification(name=name, n_class=n_class, *args, **kwargs)
        contra_id, neutral_id, entail_id = 2, 1, 0
        super().__init__(model=model, true_id=entail_id, false_id=contra_id, *args, **kwargs)
