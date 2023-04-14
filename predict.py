import argparse
import importlib

import torch
from torch.utils.data import DataLoader

from zero import ZeroShotPredictor


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str)
args = parser.parse_args()

config = importlib.import_module('config.' + args.config_file).config
model_cfg, tokenizer_cfg, data_cfg, collator_cfg, engine_cfg = config['model_cfg'], config['tokenizer_cfg'], config['data_cfg'], config['collator_cfg'], config['engine_cfg']

# Model
module_cfg = model_cfg.pop('module_cfg')
module = module_cfg.pop('cls')(**module_cfg)
if 'save_path' in model_cfg.keys():
    print('Loading from checkpoint')
    save_path = model_cfg.pop('save_path')
    model = model_cfg.pop('cls')(model=module, **model_cfg)
    model.model.load_state_dict(torch.load(save_path, map_location=engine_cfg['device']))
else:
    print('Loading from pretrained')
    model = model_cfg.pop('cls')(model=module, **model_cfg)
# Tokenizer
tokenizer = tokenizer_cfg.pop('cls')(**tokenizer_cfg)
# Data collator
collator = collator_cfg.pop('cls')(tokenizer=tokenizer, **collator_cfg)
# Datasets: training and evaluation
dataset_cls = data_cfg.pop('cls')
eval_data = dataset_cls(split='validation_matched', tokenizer=tokenizer, **data_cfg)
# Data loaders
eval_loader = DataLoader(eval_data, batch_size=engine_cfg.pop('eval_batch_size'), collate_fn=collator)

# Predictor
predictor = ZeroShotPredictor(
    model=model, data_loader=eval_loader,
    **engine_cfg
)

metrics = predictor.predict()
print('Prediction is over')
print(metrics)