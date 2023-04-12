from pathlib import Path

from transformers import AutoTokenizer

import model.modeling as models
import data.dataset as datasets
import data.preprocess as preprocesses


config = dict(
    name='pred-t5-base-mnli',
    model_cfg=dict(
        cls=models.T5MNLI,
        name='t5-base',
        n_class=3,
        save_path=Path('exp/t5-base-mnli-sched/model-step-32000.pt')

    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='t5-base',
        model_max_length=512
    ),
    data_cfg=dict(
        cls=datasets.ZeroShotDataset,
        do_tokenize=True,
        do_prompt=True,
        files=[str(x) for x in Path('/home/chikara/data/zero-shot-intent/demos/').glob('hospital_collected_clean.json')]
    ),
    collator_cfg=dict(
        cls=preprocesses.ZeroShotPaddingCollator,
        metadata_col=['ref_list', 'group']
    ),
    engine_cfg=dict(
        eval_batch_size=8,
        device='cpu'
    )
)