import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from transformers import AutoTokenizer

import model.modeling as models
import data.dataset as datasets
import data.preprocess as preprocesses


config = dict(
    name='bart-large-mnli',
    model_cfg=dict(
        cls=models.BARTClassification,
        name='facebook/bart-large'
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='facebook/bart-large',
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
        optimizer=dict(
            cls=optim.Adam,
            lr=2e-5
        ),
        scheduler=dict(
            cls=lr_scheduler.LambdaLR,
            lr_lambda=lambda epoch: 1,
        ),
        output_dir='exp/bart-large-mnli',
        train_batch_size=4,
        eval_batch_size=4,
        device='cpu',
        max_train_steps=12272*3,
        eval_steps= 4000,
        save_steps= 4000,
        log_steps= 4000,
        # logger=dict(
        #     project='t5-zero',
        #     name='bart-large-mnli'
        # )
    )
)