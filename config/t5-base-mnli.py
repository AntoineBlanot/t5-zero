import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from transformers import Adafactor, AutoTokenizer

from model.modeling import T5Classification
from data.dataset import MultiNLIDataset
from data.preprocess import PaddingCollator


config = dict(
    name='t5-base-mnli-sched-dec',
    model_cfg=dict(
        cls=T5Classification,
        name='t5-base',
        n_class=3
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
        optimizer=dict(
            cls=Adafactor,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=1e-3,
            weight_decay=1e-3
        ),
        scheduler=dict(
            cls=lr_scheduler.StepLR,
            step_size=12272,
            gamma=0.1
        ),
        output_dir='exp/t5-base-mnli-sched-dec',
        train_batch_size=32,
        eval_batch_size=32,
        device='cuda',
        max_train_steps=12272*3,
        eval_steps= 4000,
        save_steps= 4000,
        log_steps= 4000,
        logger=dict(
            project='t5-zero',
            name='t5-base-mnli-sched-dec'
        )
    )
)