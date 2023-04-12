import argparse
import importlib

from sklearn.metrics import confusion_matrix
from evaluate import load
import torch
from torch.utils.data import DataLoader

from zero import ZeroShotPredictor


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str)
args = parser.parse_args()

config = importlib.import_module('config.' + args.config_file).config
model_cfg, tokenizer_cfg, data_cfg, collator_cfg, engine_cfg = config['model_cfg'], config['tokenizer_cfg'], config['data_cfg'], config['collator_cfg'], config['engine_cfg']

# Model
if 'save_path' in model_cfg.keys():
    print('Loading from checkpoint')
    save_path = model_cfg.pop('save_path')
    model = model_cfg.pop('cls')(**model_cfg)
    model.model.load_state_dict(torch.load(save_path, map_location=engine_cfg['device']))
else:
    print('Loading from pretrained')
    model = model_cfg.pop('cls')(**model_cfg)
# Tokenizer
tokenizer = tokenizer_cfg.pop('cls')(**tokenizer_cfg)
# Data collator
collator = collator_cfg.pop('cls')(tokenizer=tokenizer, **collator_cfg)
# Datasets: training and evaluation
dataset_cls = data_cfg.pop('cls')
eval_data = dataset_cls(split='validation_matched', tokenizer=tokenizer, **data_cfg)
# Data loaders
eval_loader = DataLoader(eval_data, batch_size=engine_cfg.pop('eval_batch_size'), collate_fn=collator)
# Metrics
accuracy_metric = load("accuracy")
recall_metric = load("recall")
precision_metric = load("precision")
f1_metric = load("f1")


def compute_metrics(outputs_dict: dict, softmax_dim: int = 0) -> dict:
    """
    Compute metrics
    Args:
        - outputs_dict: models outputs format as dict
        - softmax_dim: dimension to perform softmax (0 for only True class, 1 for True and False classes)
    Returns:
        - dict of metrics
    """
    outputs = outputs_dict['outputs']
    groups = outputs_dict['group']

    group_count = [groups.count(g) for g in set(groups)]
    grouped_outputs = torch.split(outputs, group_count)

    probs = [x.softmax(softmax_dim)[:, 0] for x in grouped_outputs]
    predictions = complex_rules(probs)

    labels = outputs_dict['labels']
    labels = torch.split(labels, group_count)
    labels = [x[0].item() for x in labels]

    print([(l, p, x) for l, p, x in zip(labels, predictions, probs)])
    print(confusion_matrix(y_true=labels, y_pred=predictions))

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    rec = recall_metric.compute(predictions=predictions, references=labels, average='macro')
    prec = precision_metric.compute(predictions=predictions, references=labels, average='macro')
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')

    return {**acc, **rec, **prec, **f1}

def simple_rules(probs: list, threshold: float = 0.8) -> list:
    """
    Compute simple prediction rules
    Args:
        - probs: list of probabilities
        - threshold: if probabiliy is under, considered as fallback
    Return:
        - list of predictions
    """
    results = []
    for x in probs:
        if torch.max(x) >= threshold:
            results.append(torch.argmax(x).item())
        else:
            results.append(-1)
    return results

def complex_rules(probs: list) -> list:
    """
    Compute complex predictions rules
    Args:
        - probs: list of probabilities
    Return:
        - list of predictions
    """

    results = []

    for x in probs:
        simultaneous_threshold = 1 / (len(x)+1)
        single_threshhold = 1 / len(x)
        simultaneous_confidence = 0.95

        if torch.all(x >= simultaneous_threshold).item():
            if torch.max(x) >= simultaneous_confidence:
                i = torch.argmax(x).item()
            else:
                i = -1
        elif torch.any(x >= single_threshhold).item():
            i = torch.argmax(x).item()
        else:
            i = -1

        results.append(i)

    return results
        

# Predictor
predictor = ZeroShotPredictor(
    model=model, data_loader=eval_loader,
    compute_metrics=compute_metrics,
    **engine_cfg
)

metrics = predictor.predict()
print('Prediction is over')
print(metrics)