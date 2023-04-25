from pathlib import Path

from transformers import AutoTokenizer

import model.modeling as models
import model.zero as zero
import data.dataset as datasets
import data.preprocess as preprocesses
import data.prompt as prompts

NAME = 'pred-pretrained-bart'

config = dict(
    name=NAME,
    model_cfg=dict(
        cls=zero.SingleClassZeroShot,
        module_cfg=dict(
            cls=models.PretrainedBARTClassif,
            name='/home/chikara/models/bart-3way/'
        ),
        true_id=2,
        # false_id=0
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='facebook/bart-large',
        model_max_length=512
    ),
    data_cfg=dict(
        cls=datasets.ZeroDataset,
        files=[str(x) for x in Path('/home/chikara/data/zero-shot-intent/godel-generated/').glob('*/*.json')],
        prompt=prompts.BARTZeroIntentPrompt()
    ),
    collator_cfg=dict(
        cls=preprocesses.ZeroCollator,
        metadata_columns=['group']
    ),
    engine_cfg=dict(
        eval_batch_size=8,
        device='cuda'
    )
)