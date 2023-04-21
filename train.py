import argparse
import importlib

from evaluate import load
import torch
from torch.utils.data import DataLoader

from engine import Engine


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str)
args = parser.parse_args()

config = importlib.import_module('config.' + args.config_file).config
model_cfg, tokenizer_cfg, data_cfg, collator_cfg, engine_cfg = config['model_cfg'], config['tokenizer_cfg'], config['data_cfg'], config['collator_cfg'], config['engine_cfg']

# Model
model = model_cfg.pop('cls')(**model_cfg)
# Tokenizer
tokenizer = tokenizer_cfg.pop('cls')(**tokenizer_cfg)
# Data collator
collator = collator_cfg.pop('cls')(tokenizer=tokenizer, **collator_cfg)
# Datasets: training and evaluation
dataset_cls = data_cfg.pop('cls')
train_data = dataset_cls(split='train', tokenizer=tokenizer, **data_cfg)
eval_data = dataset_cls(split='validation_matched', tokenizer=tokenizer, **data_cfg)
# Criterion
criterion_cfg = engine_cfg.pop('criterion')
criterion = criterion_cfg.pop('cls')(**criterion_cfg)
# Optimizer
optimizer_cfg = engine_cfg.pop('optimizer')
optimizer = optimizer_cfg.pop('cls')(model.parameters(), **optimizer_cfg)
# LR Scheduler
scheduler_cfg = engine_cfg.pop('scheduler')
scheduler = scheduler_cfg.pop('cls')(optimizer, **scheduler_cfg)
# Data loaders
train_loader = DataLoader(train_data, batch_size=engine_cfg.pop('train_batch_size'), collate_fn=collator)
eval_loader = DataLoader(eval_data, batch_size=engine_cfg.pop('eval_batch_size'), collate_fn=collator)
# Metrics
accuracy_metric = load("accuracy")
recall_metric = load("recall")
precision_metric = load("precision")
f1_metric = load("f1")

def compute_metrics(outputs_dict: dict) -> dict:
    outputs = outputs_dict['outputs']
    predictions = outputs.argmax(-1)
    labels = outputs_dict['labels']

    loss = outputs_dict['loss'].mean(-1).item()
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    rec = recall_metric.compute(predictions=predictions, references=labels, average='macro')
    prec = precision_metric.compute(predictions=predictions, references=labels, average='macro')
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')

    return {**dict(loss=loss), **acc, **rec, **prec, **f1}

def compute_binary_metrics(outputs_dict: dict) -> dict:
    outputs = outputs_dict['outputs']
    predictions = torch.where(outputs.sigmoid() > 0.5, 1.0, 0.0)
    labels = outputs_dict['labels']

    loss = outputs_dict['loss'].mean(-1).item()
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    rec = recall_metric.compute(predictions=predictions, references=labels, average='macro')
    prec = precision_metric.compute(predictions=predictions, references=labels, average='macro')
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')

    return {**dict(loss=loss), **acc, **rec, **prec, **f1}

# Engine
engine =  Engine(
    model=model, train_loader=train_loader, eval_loader=eval_loader,
    criterion=criterion, optimizer=optimizer, scheduler=scheduler, compute_metrics=compute_metrics,
    **engine_cfg
)

metrics = engine.train()
print('Training is over')
print(metrics)