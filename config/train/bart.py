import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from transformers import AutoTokenizer

import model.modeling as models
import data.dataset as datasets
import data.preprocess as preprocesses
import data.prompt as prompts

NAME = 'bart-base-mnli'

config = dict(
    name=NAME,
    model_cfg=dict(
        cls=models.BERTClassif,
        name='facebook/bart-base-mnli',
        n_class=3
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='facebook/bart-base-mnli',
        model_max_length=128
    ),
    data_cfg=dict(
        cls=datasets.MNLIDataset,
        prompt=prompts.BARTNLIPrompt(),
        to_binary=False
    ),
    collator_cfg=dict(
        cls=preprocesses.TokenizeAndPad
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
        output_dir=f'exp/{NAME}',
        train_batch_size=32,
        eval_batch_size=32,
        device='cuda',
        max_train_steps=12272*3,
        eval_steps=1000,
        save_steps=1000,
        log_steps=1000,
        logger=dict(
            project='t5-zero',
            name=NAME
        )
    )
)