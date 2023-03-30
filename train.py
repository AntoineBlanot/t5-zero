from transformers import (
    AutoTokenizer,
    Trainer, TrainingArguments,
    Adafactor
)
from evaluate import load
import torch
from torch.utils.data import DataLoader

from model.modeling import MyModel
from data.dataset import MultiNLIDataset
from data.preprocess import PaddingCollator
from trainer import MyTrainer

name = "t5-large"
tokenizer = AutoTokenizer.from_pretrained(name)
model = MyModel(name, 3)
print('Model params: {} M'.format(int(sum(p.numel() for p in model.parameters())/1e6)))
print('Trainable params: {} M'.format(int(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)))


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

    return {**dict(eval_loss=loss), **acc, **rec, **prec, **f1}

train_dataset = MultiNLIDataset(split='train', tokenizer=tokenizer)
eval_dataset = MultiNLIDataset(split='validation_matched', tokenizer=tokenizer)

collator = PaddingCollator(tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collator)
eval_loader = DataLoader(eval_dataset, batch_size=4, collate_fn=collator)

criterion = torch.nn.CrossEntropyLoss()
optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)

trainer = MyTrainer(
    model=model,
    train_loader=train_loader,
    eval_loader=eval_loader,
    criterion=criterion,
    optimizer=optimizer,
    compute_metrics=compute_metrics,
    output_dir='t5-zero-1',
    device='cuda',
    max_train_steps=len(train_loader)*2,
    eval_steps=1000,
    save_steps=1000
)

out = trainer.train()

print(out)