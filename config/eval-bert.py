from pathlib import Path

import torch.nn as nn
from transformers import AutoTokenizer

import model.modeling as models
import data.dataset as datasets
import data.preprocess as preprocesses


config = dict(
    name='eval-bert-base-mnli',
    model_cfg=dict(
        cls=models.BERTClassification,
        name='bert-base-uncased',
        n_class=3,
        save_path=Path('exp/bert-base-mnli/model-step-24000.pt')
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='bert-base-uncased',
        model_max_length=128
    ),
    data_cfg=dict(
        cls=datasets.MultiNLIDataset,
        do_tokenize=True,
        do_prompt=False
    ),
    collator_cfg=dict(
        cls=preprocesses.PaddingCollator
    ),
    engine_cfg=dict(
        criterion=dict(
            cls=nn.CrossEntropyLoss
        ),
        output_dir='.',
        eval_batch_size=8,
        device='cpu'
    )
)