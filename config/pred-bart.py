from pathlib import Path

from transformers import AutoTokenizer

import model.modeling as models
import model.zero as zero
import data.dataset as datasets
import data.preprocess as preprocesses


config = dict(
    name='pred-bart',
    model_cfg=dict(
        cls=zero.MultiClassZeroShot,
        module_cfg=dict(
            cls=models.BARTClassif,
            name='facebook/bart-large',
            id2label={'0': 'entaillment', '1': 'neutral', '2': 'contradiction'}
        ),
        save_path=Path('exp/bart-large-mnli-sched/model-step-28000.pt'),
        true_id=0,
        false_id=2
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='facebook/bart-large',
        model_max_length=512
    ),
    data_cfg=dict(
        cls=datasets.ZeroShotDataset,
        do_tokenize=True,
        do_prompt=True,
        files=[str(x) for x in Path('/home/chikara/data/').glob('hospital_collected_clean.json')]
    ),
    collator_cfg=dict(
        cls=preprocesses.ZeroShotPaddingCollator,
        metadata_col=['ref_list', 'group']
    ),
    engine_cfg=dict(
        eval_batch_size=32,
        device='cuda'
    )
)