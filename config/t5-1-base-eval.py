from pathlib import Path

from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer

from model.modeling import MyModel
from data.dataset import MultiNLIDataset
from data.preprocess import PaddingCollator


config = dict(
    name='t5-1',
    model_cfg=dict(
        cls=MyModel,
        name='t5-base',
        n_class=3,
        save_path=Path('t5-1-base/model-step-100000.pt')
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='t5-base',
        model_max_length=512
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
            cls=CrossEntropyLoss
        ),
        output_dir='t5-1-base',
        eval_batch_size=8,
        device='cuda'
    )
)