from torch.nn import CrossEntropyLoss
from transformers import Adafactor, AutoTokenizer

from model.modeling import MyModel
from data.dataset import DummyDataset
from data.preprocess import PaddingCollator


config = dict(
    name='t5-1',
    model_cfg=dict(
        cls=MyModel,
        name='t5-large',
        n_class=3
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='t5-large'
    ),
    data_cfg=dict(
        cls=DummyDataset,
        n=1000,
        l=512
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
        train_batch_size=8,
        eval_batch_size=8,
        device='cuda',
        max_train_steps=100,
        eval_steps= 20,
        save_steps= 20
    )
)