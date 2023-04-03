from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm

class MyTrainer():

    def __init__(self,
        model: nn.Module,
        train_loader: DataLoader = None, eval_loader: DataLoader = None,
        criterion: nn.Module = None , optimizer: optim = None, scheduler: optim = None, compute_metrics: callable = None,
        output_dir: str = '.', device: str = 'cpu',
        max_train_steps: int = None, eval_steps : int = None, save_steps: int = None, log_steps: int = None,
        logger: dict = None
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compute_metrics = compute_metrics

        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.device = device

        default_steps = len(train_loader) if train_loader is not None else 0
        self.max_train_steps = max_train_steps if max_train_steps is not None else default_steps
        self.eval_steps = eval_steps if eval_steps is not None else max_train_steps
        self.save_steps = save_steps if save_steps is not None else max_train_steps
        self.log_steps = log_steps if log_steps is not None else max_train_steps

        self.logger = logger


    def train(self) -> dict:
        """
        Training entry point.
        """

        self.model.train()
        train_pbar = tqdm(range(self.max_train_steps), desc='Training')
        self.train_step = 0
        self.all_train_metrics, self.all_eval_metrics = {}, {}

        if self.logger is not None:
            import wandb
            wandb.init(**self.logger)
            wandb.watch(self.model, log_freq=self.log_steps)

        while not self.__should_stop_train():
            for i, inputs in enumerate(self.train_loader):
                self.train_step += 1
                train_pbar.update()
                inputs = {k: v.to(self.device) for k,v in inputs.items()}
                labels = inputs.pop('label')
                
                self.optimizer.zero_grad()

                outputs = self.model(**inputs)

                train_loss = self.criterion(outputs['head_outputs'], labels)
                train_loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                train_metrics = dict(train_epoch=round(self.train_step/len(self.train_loader), 4), train_loss=train_loss.item(), train_lr=self.scheduler.get_last_lr()[0])
                self.all_train_metrics.update({k: self.all_train_metrics.get(k, []) + [v] for k,v in train_metrics.items()})

                if self.__should_eval():
                    eval_metrics = self.evaluate()
                    self.all_eval_metrics.update(eval_metrics)
                    self.model.train()

                if self.__should_log():
                    log_metrics = {
                        **{k: np.asarray(v).mean() if k not in ['train_epoch', 'train_lr'] else v[-1] for k,v in self.all_train_metrics.items()},
                        **self.all_eval_metrics
                    }
                    train_pbar.set_postfix(log_metrics)

                    if self.logger is not None:
                        wandb.log({
                            **{'train/step': self.train_step},
                            **{k.replace('_', '/'): v for k,v in log_metrics.items()}
                        })

                    self.all_train_metrics = {}      # reset train metrics

                if self.__should_save():
                    save_path = '{}/model-step-{}.pt'.format(self.output_dir, self.train_step)
                    torch.save(self.model.state_dict(), save_path)

                if self.__should_stop_train():
                    break
        
        if self.logger is not None:
            wandb.finish()

        return log_metrics

    def evaluate(self) -> dict:
        """
        Evaluation entry point.
        """
        self.model.eval()
        eval_pbar = tqdm(self.eval_loader, desc='Evaluation', leave=False)

        all_eval_outputs, all_eval_labels = [], []
        all_eval_loss = []
        for i, inputs in enumerate(self.eval_loader):
            with torch.no_grad():
                eval_pbar.update()
                inputs = {k: v.to(self.device) for k,v in inputs.items()}
                labels = inputs.pop('label')
                
                outputs = self.model(**inputs)

                eval_loss = self.criterion(outputs['head_outputs'], labels)

                all_eval_outputs.append(outputs['head_outputs'].detach())
                all_eval_labels.append(labels)
                all_eval_loss.append(eval_loss.detach())

        eval_dict = dict(outputs=all_eval_outputs, labels=all_eval_labels, loss=all_eval_loss)
        eval_dict = {k: torch.cat(v) if v[0].dim() > 0 else torch.stack(v) for k,v in eval_dict.items()}

        if self.compute_metrics is not None:
            eval_metrics = self.compute_metrics(eval_dict)
            eval_metrics = {'eval_' + k: v for k,v in eval_metrics.items()}
        else:
            eval_metrics = {}

        return eval_metrics

    def __should_stop_train(self) -> bool:
        if self.train_step >= self.max_train_steps:
            return True
        return False

    def __should_eval(self) -> bool:
        if (self.train_step % self.eval_steps) == 0:
            return True
        return False
    
    def __should_save(self) -> bool:
        if (self.train_step % self.save_steps) == 0:
            return True
        return False 
    
    def __should_log(self) -> bool:
        if (self.train_step % self.log_steps) == 0:
            return True
        return False
