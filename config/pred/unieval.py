from pathlib import Path

from transformers import AutoTokenizer

import model.modeling as models
import model.zero as zero
import data.dataset as datasets
import data.preprocess as preprocesses
import data.prompt as prompts

NAME = 'pred-pretrained-unieval'

config = dict(
    name=NAME,
    model_cfg=dict(
        cls=zero.SingleClassZeroShot,
        module_cfg=dict(
            cls=models.PretrainedUniEvalClassif,
            name='MingZhong/unieval-intermediate'
        ),
        true_id=2163,
        # false_id=465
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='MingZhong/unieval-intermediate',
        model_max_length=512
    ),
    data_cfg=dict(
        cls=datasets.ZeroDataset,
        files=[str(x) for x in Path('/home/chikara/data/zero-shot-intent/tung-yesno/').glob('*/data.json')],
        prompt=prompts.UniEvalZeroYesNoPrompt(),
        to_binary=False
    ),
    collator_cfg=dict(
        cls=preprocesses.ZeroCollator,
        metadata_columns=['group']
    ),
    engine_cfg=dict(
        eval_batch_size=8,
        device='cpu'
    )
)