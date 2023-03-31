from torch.nn import CrossEntropyLoss
from transformers import Adafactor, AutoTokenizer

from model.modeling import MyModel
from data.dataset import DummyDataset
from data.preprocess import PaddingCollator


config = dict(
    name='t5-1',
    model_cfg=dict(
        cls=MyModel,
        name='t5-base',
        n_class=3
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='t5-base',
        model_max_length=200
    ),
    data_cfg=dict(
        cls=DummyDataset,
        n=100,
        l=200
    ),
    collator_cfg=dict(
        cls=PaddingCollator
    ),
    train_cfg=dict(
        criterion=dict(
            cls=CrossEntropyLoss
        ),
        optimizer=dict(
            cls=Adafactor,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=1e-3
        ),
        output_dir='test',
        train_batch_size=32,
        eval_batch_size=32,
        device='cuda',
        max_train_steps=20,
        eval_steps= 10,
        save_steps= 10,
        log_steps= 10
    )
)