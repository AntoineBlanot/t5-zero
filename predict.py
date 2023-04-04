import argparse
import importlib

from evaluate import load
import torch
from torch.utils.data import DataLoader

from zero import ZeroShotPredictor


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str)
args = parser.parse_args()

config = importlib.import_module('config.' + args.config_file).config
model_cfg, tokenizer_cfg, data_cfg, collator_cfg, train_cfg = config['model_cfg'], config['tokenizer_cfg'], config['data_cfg'], config['collator_cfg'], config['train_cfg']

# Model
save_path = model_cfg.pop('save_path')
model = model_cfg.pop('cls')(**model_cfg)
model.load_state_dict(torch.load(save_path))
# Tokenizer
tokenizer = tokenizer_cfg.pop('cls')(**tokenizer_cfg)
# Data collator
collator = collator_cfg.pop('cls')(tokenizer=tokenizer, **collator_cfg)
# Datasets: training and evaluation
dataset_cls = data_cfg.pop('cls')
eval_data = dataset_cls(split='validation_matched', tokenizer=tokenizer, **data_cfg)
# Data loaders
eval_loader = DataLoader(eval_data, batch_size=train_cfg.pop('eval_batch_size'), collate_fn=collator)


def compute_metrics(outputs_dict: dict) -> dict:
    accuracy_metric = load("accuracy")
    recall_metric = load("recall")
    precision_metric = load("precision")
    f1_metric = load("f1")

    outputs = outputs_dict['outputs']
    groups = outputs_dict['group']
    group_count = [groups.count(g) for g in set(groups)]
    
    grouped_outputs = torch.split(outputs, group_count)
    probs = [x[:, 0].softmax(-1) for x in grouped_outputs]
    predictions = [x.argmax() for x in probs]
    labels = outputs_dict['labels']
    labels = torch.split(labels, group_count)
    labels = [x[0].item() for x in labels]

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    rec = recall_metric.compute(predictions=predictions, references=labels, average='macro')
    prec = precision_metric.compute(predictions=predictions, references=labels, average='macro')
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')

    return {**acc, **rec, **prec, **f1}


# Trainer
predictor = ZeroShotPredictor(
    model=model, data_loader=eval_loader,
    compute_metrics=compute_metrics,
    **train_cfg
)

metrics = predictor.predict()
print('Prediction is over')
print(metrics)