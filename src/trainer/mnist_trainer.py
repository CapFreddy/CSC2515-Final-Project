import torch

from ..utils.util import get_lr
from .trainer import BaseTrainer, TrainingState
import torch.nn as nn
from ..utils.config import Config
from torch.utils.data import Dataset, DataLoader
from typing import Union, Callable
from ..constant import ROOT
import wandb


class Trainer(BaseTrainer):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: object, criterion: Union[torch.nn.Module, Callable], config: Config, train_dataset: Dataset, train_dataloader: DataLoader, test_dataloader: DataLoader) -> None:
        super().__init__(model, optimizer, scheduler, criterion, config, train_dataset)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.best_valid_loss = float('inf')

    def _train_epoch(self, epoch: int) -> TrainingState:
        self.model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.progress_bar.update(len(data))
            self.progress_bar.set_postfix(train_L=loss.item(),
                                          val_L=(self.valid_losses[-1] if len(self.valid_losses) != 0 else None),
                                          lr=get_lr(self.optimizer),
                                          bs=len(data))
        self.train_loss = total_loss / len(self.train_dataloader.dataset)
        self.train_losses.append(self.train_loss)
        self.valid_loss = self._eval_epoch(epoch)
        if self.valid_loss < self.best_valid_loss:
            # new best weights
            self.logger.debug(
                f"Best validation loss so far. validation loss={self.valid_loss}. Save state at epoch={epoch}")
            self.best_valid_loss = self.valid_loss
            super(Trainer, self)._save_checkpoint(epoch, is_best=True)
        self.valid_losses.append(self.valid_loss)
        self.learning_rates.append(get_lr(self.optimizer))
        if self.scheduler is not None:
            self.scheduler.step()
        return TrainingState(epoch, get_lr(self.optimizer), self.train_loss, self.valid_loss)

    def _eval_epoch(self, epoch: int) -> float:
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                test_loss += self.criterion(output,
                                            target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_dataloader.dataset)

        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(self.test_dataloader.dataset),
        #     100. * correct / len(self.test_dataloader.dataset)))
        return test_loss
