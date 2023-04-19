import torch
from torch import Tensor, nn
from evaluate import load
from sklearn.metrics import confusion_matrix

accuracy_metric = load("accuracy")
recall_metric = load("recall")
precision_metric = load("precision")
f1_metric = load("f1")

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

class MultiClassZeroShot(nn.Module):
    """
    Make use of True and False class for zero-shot predictions (in NLI `entaillment` and `contradiction`)
    """

    def __init__(self, model: nn.Module, true_id: int, false_id: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.true_id = true_id
        self.false_id = false_id

    def forward(self, *args, **kwargs) -> Tensor:
        # Model logits
        logits = self.model(*args, **kwargs)['logits']
        # Outputs (True and False class)
        logits = logits[:, [self.true_id, self.false_id]]
        assert logits.shape[1] == 2

        return dict(
            logits=logits
        )
    
    def compute_metrics(self, outputs_dict: dict, average: str = 'weighted') -> dict:
        outputs = outputs_dict['outputs']
        groups = outputs_dict['group']

        group_count = [groups.count(g) for g in set(groups)]
        grouped_outputs = torch.split(outputs, group_count)

        probs = [x.softmax(1)[:, 0] for x in grouped_outputs]
        predictions = complex_rules(probs)

        labels = outputs_dict['labels']
        labels = torch.split(labels, group_count)
        labels = [x[0].item() for x in labels]

        # print([(l, p, x) for l, p, x in zip(labels, predictions, probs)])
        print(confusion_matrix(y_true=labels, y_pred=predictions))

        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        rec = recall_metric.compute(predictions=predictions, references=labels, average=average)
        prec = precision_metric.compute(predictions=predictions, references=labels, average=average)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average=average)

        return {**acc, **rec, **prec, **f1}



class SingleClassZeroShot(nn.Module):
    """
    Make use of only True class for zero-shot predictions (in NLI `entaillment`). Can also be used if the model is binary (single class)
    """

    def __init__(self, model: nn.Module, true_id: int = None , *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.true_id = true_id

    def forward(self, *args, **kwargs) -> Tensor:
        # Model logits
        logits = self.model(*args, **kwargs)['logits']
        # Outputs (True class only if the model was originally multiclass)
        assert not ((len(logits.shape) > 1) and (self.true_id is None)), 'true_id must be defined if the model was originally multiclass'
        logits = logits[:, [self.true_id]] if self.true_id is not None else logits.unsqueeze(1)

        return dict(
            logits=logits
        )
    
    def compute_metrics(self, outputs_dict: dict, average: str = 'weighted') -> dict:
        outputs = outputs_dict['outputs']
        groups = outputs_dict['group']

        group_count = [groups.count(g) for g in set(groups)]
        grouped_outputs = torch.split(outputs, group_count)

        probs = [x.softmax(0)[:, 0] for x in grouped_outputs]
        predictions = complex_rules(probs)

        labels = outputs_dict['labels']
        labels = torch.split(labels, group_count)
        labels = [x[0].item() for x in labels]

        # print([(l, p, x) for l, p, x in zip(labels, predictions, probs)])
        print(confusion_matrix(y_true=labels, y_pred=predictions))

        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        rec = recall_metric.compute(predictions=predictions, references=labels, average=average)
        prec = precision_metric.compute(predictions=predictions, references=labels, average=average)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average=average)

        return {**acc, **rec, **prec, **f1}