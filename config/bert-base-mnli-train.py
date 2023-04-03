from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoTokenizer

from model.modeling import BERTClassification
from data.dataset import MultiNLIDataset
from data.preprocess import PaddingCollator


config = dict(
    name='bert-base-mnli',
    model_cfg=dict(
        cls=BERTClassification,
        name='bert-base-uncased',
        n_class=3
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='bert-base-uncased',
        model_max_length=200
    ),
    data_cfg=dict(
        cls=MultiNLIDataset,
        do_tokenize=True,
        do_prompt=False
    ),
    collator_cfg=dict(
        cls=PaddingCollator
    ),
    train_cfg=dict(
        criterion=dict(
            cls=CrossEntropyLoss
        ),
        optimizer=dict(
            cls=Adam,
            lr=2e-5
        ),
        output_dir='exp/bert-base-mnli',
        train_batch_size=32,
        eval_batch_size=32,
        device='cpu',
        max_train_steps=12272*3,
        eval_steps= 4000,
        save_steps= 4000,
        log_steps= 4000,
        logger=dict(
            project='t5-zero',
            name='bert-base-mnli'
        )
    )
)