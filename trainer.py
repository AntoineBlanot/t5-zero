from pathlib import Path

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm

class MyTrainer():

    def __init__(self,
        model: nn.Module,
        train_loader: DataLoader, eval_loader: DataLoader,
        criterion: nn.Module, optimizer: optim, compute_metrics: callable = None,
        output_dir: str = '.', device: str = 'cpu',
        max_train_steps: int = None, eval_steps : int = None, save_steps: int = None
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.compute_metrics = compute_metrics

        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.device = device

        default_steps = len(train_loader)
        self.max_train_steps = max_train_steps if max_train_steps is not None else default_steps
        self.eval_steps = eval_steps if eval_steps is not None else default_steps
        self.save_steps = save_steps if save_steps is not None else default_steps

    def train(self) -> dict:
        """
        Training entry point.
        """
        self.model.train()
        train_pbar = tqdm(range(self.max_train_steps), desc='Training', leave=True)
        self.train_step = 0
        self.train_metrics = {}

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
                self.train_metrics.update(dict(epoch_step=self.train_step/len(self.train_loader), train_loss=train_loss.item()))
                train_pbar.set_postfix(self.train_metrics)

                if self.__should_eval():
                    eval_res = self.evaluate()
                    self.train_metrics.update(eval_res)
                    train_pbar.set_postfix(self.train_metrics)
                    self.model.train()

                if self.__should_save():
                    save_path = '{}/model-step-{}.pt'.format(self.output_dir, self.train_step)
                    torch.save(self.model.state_dict(), save_path)

                if self.__should_stop_train():
                    break

    def evaluate(self) -> dict:
        """
        Evaluation entry point.
        """
        self.model.eval()
        eval_pbar = tqdm(self.eval_loader, desc='Evaluation')

        all_eval_outputs, all_eval_labels = [], []
        all_eval_loss = []
        for i, inputs in enumerate(self.eval_loader):
            with torch.no_grad():
                eval_pbar.update()
                inputs = {k: v.to(self.device) for k,v in inputs.items()}
                labels = inputs.pop('label')
                
                outputs = self.model(**inputs)

                eval_loss = self.criterion(outputs['head_outputs'], labels)

                eval_pbar.set_postfix(dict(eval_loss=eval_loss.item()))

                all_eval_outputs.append(outputs['head_outputs'].detach())
                all_eval_labels.append(labels)
                all_eval_loss.append(eval_loss.detach())

        eval_dict = dict(outputs=all_eval_outputs, labels=all_eval_labels, loss=all_eval_loss)
        eval_dict = {k: torch.cat(v) if v[0].dim() > 0 else torch.stack(v) for k,v in eval_dict.items()}
        
        eval_metrics = self.compute_metrics(eval_dict)
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