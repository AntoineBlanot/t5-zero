import argparse
import importlib

from evaluate import load
import torch
from torch.utils.data import DataLoader

from trainer import MyTrainer


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
# Criterion
criterion_cfg = train_cfg.pop('criterion')
criterion = criterion_cfg.pop('cls')(**criterion_cfg)
# Data loaders
eval_loader = DataLoader(eval_data, batch_size=train_cfg.pop('eval_batch_size'), collate_fn=collator)


def compute_metrics(outputs_dict: dict) -> dict:
    accuracy_metric = load("accuracy")
    recall_metric = load("recall")
    precision_metric = load("precision")
    f1_metric = load("f1")

    outputs = outputs_dict['outputs']
    predictions = outputs.argmax(-1)
    labels = outputs_dict['labels']

    loss = outputs_dict['loss'].mean(-1).item()
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    rec = recall_metric.compute(predictions=predictions, references=labels, average='macro')
    prec = precision_metric.compute(predictions=predictions, references=labels, average='macro')
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')

    return {**dict(loss=loss), **acc, **rec, **prec, **f1}


# Trainer
trainer =  MyTrainer(
    model=model, eval_loader=eval_loader,
    criterion=criterion, compute_metrics=compute_metrics,
    **train_cfg
)

metrics = trainer.evaluate()
print('Evaluation is over')
print(metrics)