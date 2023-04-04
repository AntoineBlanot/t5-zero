from pathlib import Path

import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from transformers import AutoTokenizer

from model.modeling import T5Classification
from data.dataset import MultiNLIDataset
from data.preprocess import PaddingCollator


config = dict(
    name='t5-base-mnli-sched-dec',
    model_cfg=dict(
        cls=T5Classification,
        name='t5-base',
        n_class=3,
        save_path=Path('exp/t5-base-mnli-sched-dec/model-step-36000.pt')
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='t5-base',
        model_max_length=200
    ),
    data_cfg=dict(
        cls=MultiNLIDataset,
        do_tokenize=True,
        do_prompt=True
    ),
    collator_cfg=dict(
        cls=PaddingCollator
    ),
    train_cfg=dict(
        criterion=dict(
            cls=nn.CrossEntropyLoss
        ),
        scheduler=dict(
            cls=lr_scheduler.StepLR,
            step_size=12272,
            gamma=0.1
        ),
        output_dir='exp/t5-base-mnli-sched-dec',
        eval_batch_size=32,
        device='cuda',
    )
)