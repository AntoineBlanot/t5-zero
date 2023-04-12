from pathlib import Path

from transformers import AutoTokenizer

import model.modeling as models
import data.dataset as datasets
import data.preprocess as preprocesses


config = dict(
    name='pred-bart-mnli',
    model_cfg=dict(
        cls=models.BERTMNLI,
        name='bert-base-uncased',
        n_class=3,
        save_path=Path('exp/bert-base-mnli/model-step-24000.pt')

    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='bert-base-uncased',
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