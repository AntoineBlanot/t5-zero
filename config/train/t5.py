import torch.nn as nn
import torch. optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from transformers import AutoTokenizer, Adafactor

import model.modeling as models
import data.dataset as datasets
import data.preprocess as preprocesses
import data.prompt as prompts

NAME = 't5-base-mnli'

config = dict(
    name=NAME,
    model_cfg=dict(
        cls=models.T5Classif,
        name='t5-base',
        n_class=3
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='t5-base',
        model_max_length=200
    ),
    data_cfg=dict(
        cls=datasets.MNLIDataset,
        prompt=prompts.T5NLIPrompt(),
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
            cls=Adafactor,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=1e-3
        ),
        scheduler=dict(
            cls=lr_scheduler.StepLR,
            step_size=12272,
            gamma=0.1
        ),
        output_dir=f'exp/{NAME}',
        train_batch_size=8,
        eval_batch_size=8,
        device='cpu',
        max_train_steps=12272*3,
        eval_steps= 4000,
        save_steps= 4000,
        log_steps= 4000,
        # logger=dict(
        #     project='t5-zero',
        #     name=NAME
        # )
    )
)