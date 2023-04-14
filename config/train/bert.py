import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from transformers import AutoTokenizer

import model.modeling as models
import data.dataset as datasets
import data.preprocess as preprocesses
import data.prompt as prompts

NAME = 'bert-base-mnli'

config = dict(
    name=NAME,
    model_cfg=dict(
        cls=models.BERTClassif,
        name='bert-base-uncased',
        n_class=3
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='bert-base-uncased',
        model_max_length=128
    ),
    data_cfg=dict(
        cls=datasets.MNLIDataset,
        prompt=prompts.BERTNLIPrompt(),
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
        eval_steps= 500,
        save_steps= 500,
        log_steps= 500,
        logger=dict(
            project='t5-zero',
            name=NAME
        )
    )
)