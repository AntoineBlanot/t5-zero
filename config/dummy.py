import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from transformers import AutoTokenizer, Adafactor, DataCollatorForSeq2Seq

import model.modeling as models
import data.dataset as datasets
import data.preprocess as preprocesses

NAME = 'test'


config = dict(
    name=NAME,
    model_cfg=dict(
        cls=models.T5Classif,
        name='t5-large',
        n_class=3
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='t5-large',
        model_max_length=200
    ),
    data_cfg=dict(
        cls=datasets.DummyDataset,
        n=100,
        l=200
    ),
    collator_cfg=dict(
        cls=DataCollatorForSeq2Seq
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
        output_dir=NAME,
        train_batch_size=32,
        eval_batch_size=32,
        device='cuda',
        max_train_steps=20,
        eval_steps= 10,
        save_steps= 10,
        log_steps= 10
    )
)