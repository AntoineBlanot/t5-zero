import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from transformers import AutoTokenizer, get_polynomial_decay_schedule_with_warmup

import model.modeling as models
import data.dataset as datasets
import data.preprocess as preprocesses
import data.prompt as prompts

NAME = 'roberta-large-mnli-fairseq'

config = dict(
    name=NAME,
    model_cfg=dict(
        cls=models.RoBertaClassif,
        name='roberta-large',
        n_class=3
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='roberta-large',
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
            lr=1e-6,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.01
        ),
        scheduler=dict(
            cls=get_polynomial_decay_schedule_with_warmup,
            num_warmup_steps=7432,
            num_training_steps=123873,
            lr_end=0
        ),
        output_dir=f'exp/{NAME}',
        train_batch_size=32,
        eval_batch_size=32,
        grad_acc=1,
        fp16=True,
        device='cuda',
        max_train_steps=123873,
        eval_steps=1000,
        save_steps=1000,
        log_steps=1000,
        logger=dict(
            project='t5-zero',
            name=NAME
        )
    )
)