from pathlib import Path

from transformers import AutoTokenizer

from model.modeling import T5Classification
from data.dataset import ZeroShotDataset
from data.preprocess import ZeroShotPaddingCollator


config = dict(
    name='t5-base-mnli-sched-dec',
    model_cfg=dict(
        cls=T5Classification,
        name='t5-base',
        n_class=3,
        save_path=Path('exp/t5-base-mnli-sched-dec/model-step-36000.pt')
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='t5-base',
        model_max_length=200
    ),
    data_cfg=dict(
        cls=ZeroShotDataset,
        do_tokenize=True,
        do_prompt=True,
        files=[str(x) for x in Path('/home/chikara/data/zero-shot-intent/hri-forms/').glob('*/*v2.json')]
    ),
    collator_cfg=dict(
        cls=ZeroShotPaddingCollator,
        metadata_col=['ref_list', 'group']
    ),
    train_cfg=dict(
        eval_batch_size=8,
        device='cpu'
    )
)