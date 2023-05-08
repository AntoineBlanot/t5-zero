from pathlib import Path

from transformers import AutoTokenizer

import model.modeling as models
import model.zero as zero
import datas.dataset as datasets
import datas.preprocess as preprocesses
import datas.prompt as prompts

NAME = 'pred-pretrained-roberta'

config = dict(
    name=NAME,
    model_cfg=dict(
        cls=zero.SingleClassZeroShot,
        module_cfg=dict(
            cls=models.T5ModelForClassification,
            config_dict=dict(
                pretrained_model_name_or_path='google/t5-v1_1-large',
                num_decoder_layers=1
            )
        ),
        save_path=Path('/home/chikara/ws/checkpoint-24000/pytorch_model.bin'),
        true_id=0,
        # false_id=2
    ),
    tokenizer_cfg=dict(
        cls=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='google/t5-v1_1-large',
        model_max_length=512
    ),
    data_cfg=dict(
        cls=datasets.ZeroDataset,
        files=[str(x) for x in Path('/home/chikara/data/zero-shot-intent/godel-generated/').glob('*/*.json')],
        prompt=prompts.T5ZeroIntentPrompt()
    ),
    collator_cfg=dict(
        cls=preprocesses.T5ZeroCollator,
        metadata_columns=['group']
    ),
    engine_cfg=dict(
        eval_batch_size=8,
        device='cuda'
    )
)