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
        cls=zero.MultiClassZeroShot,
        module_cfg=dict(
            cls=models.PretrainedBARTClassif,
            name='facebook/bart-large-mnli'
        ),
        true_id=2,
        false_id=0
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='facebook/bart-large-mnli',
        model_max_length=512
    ),
    data_cfg=dict(
        cls=datasets.ZeroDataset,
        files=[str(x) for x in Path('/home/chikara/data/zero-shot-intent/demos/').glob('sentiment_collected_clean.json')],
        prompt=prompts.BARTZeroSentimentPrompt(),
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