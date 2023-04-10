from pathlib import Path

import torch.nn as nn
from transformers import AutoTokenizer

import model.modeling as models
import data.dataset as datasets
import data.preprocess as preprocesses


config = dict(
    name='eval-t5-base-mnli-sched',
    model_cfg=dict(
        cls=models.T5Classification,
        name='t5-base',
        n_class=3,
        save_path=Path('exp/t5-base-mnli-sched/model-step-32000.pt')
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='t5-base',
        model_max_length=200
    ),
    data_cfg=dict(
        cls=datasets.MultiNLIDataset,
        do_tokenize=True,
        do_prompt=True
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