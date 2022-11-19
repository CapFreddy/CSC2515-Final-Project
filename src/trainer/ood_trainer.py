import torch

from ..utils.util import get_lr
from .trainer import BaseTrainer, TrainingState
import torch.nn as nn
from ..utils.config import Config
from torch.utils.data import Dataset, DataLoader
from typing import Union, Callable
from ..constant import ROOT
import wandb
from dataclasses import dataclass
from torch.optim import Optimizer


@dataclass
class OODTrainingState(TrainingState):
    valid_acc: float


class OODTrainer(BaseTrainer):
    def __init__(self, model: nn.Module,
                 optimizer: Optimizer, optimizer_cls: Optimizer, optimizer_domain: Optimizer,
                 scheduler: object, scheduler_cls: object, scheduler_domain: object,
                 criterion: Union[torch.nn.Module, Callable], config: Config,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader, test_dataloader: DataLoader,
                 cls_classifier: nn.Module, domain_classifier: nn.Module,
                 latent_dim: int
                 ) -> None:
        super().__init__(model, optimizer, scheduler,
                         criterion, config, train_dataloader.dataset)
        self.latent_dim = latent_dim
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls
        self.scheduler_domain = scheduler_domain
        self.optimizer_domain = optimizer_domain
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.cls_classifier = cls_classifier
        self.domain_classifier = domain_classifier
        self.best_valid_loss = float('inf')
        self.backbone = self.model

    def train(self):
        self.backbone = self.backbone.to(self.device)
        self.cls_classifier = self.cls_classifier.to(self.device)
        self.domain_classifier = self.domain_classifier.to(self.device)
        super().train()

    def _train_epoch(self, epoch: int):
        self.backbone.train()
        self.cls_classifier.train()
        self.domain_classifier.train()
        cuda = torch.cuda.is_available()
        total_loss = 0
        for it, (batch, domain) in enumerate(self.train_dataloader):
            x = batch[0]
            x = x.reshape(self.config.batch_size, -1)
            y = batch[1]
            if cuda:
                x = x.cuda()
                y = y.cuda()
                domain = domain.cuda()
            feature = self.backbone(x)
            if self.domain_classifier is not None:
                feature_domain = feature[:, self.latent_dim:]
                feature = feature[:, :self.latent_dim]
                scores_cls = self.cls_classifier(feature)
                loss_cls = self.criterion(scores_cls, y)
                scores_domain = self.domain_classifier(feature_domain)
                loss_domain = self.criterion(scores_domain, domain)
                loss = loss_cls + 0.5 * loss_domain
            else:
                feature = feature[:, :self.latent_dim]
                scores = self.cls_classifier(feature)
                loss_cls = self.criterion(scores, y)
                loss = loss_cls
            self.progress_bar.set_postfix(train_L=loss.item(),
                                          val_L=(
                                              self.valid_losses[-1] if len(self.valid_losses) != 0 else None),
                                          lr=get_lr(self.optimizer),
                                          batch_size=len(batch))
            self.progress_bar.update(len(batch[1]))
            total_loss += loss.item()
            self.optimizer.zero_grad()
            self.optimizer_cls.zero_grad()
            if self.domain_classifier is not None:
                self.optimizer_domain.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_cls.step()
            if self.domain_classifier is not None:
                self.optimizer_domain.step()
        self.scheduler.step()
        self.scheduler_cls.step()
        if self.domain_classifier is not None:
            self.scheduler_domain.step()

        self.train_loss = total_loss / \
            len(self.train_dataloader.dataset)  # compute average loss
        self.train_losses.append(self.train_loss)
        valid_acc, self.valid_loss = self._eval(epoch, self.valid_dataloader)
        if self.valid_loss < self.best_valid_loss:
            # new best weights
            self.logger.debug(
                f"Best validation loss so far. validation loss={self.valid_loss}. Save state at epoch={epoch}")
            self.best_valid_loss = self.valid_loss
            super()._save_checkpoint(epoch, is_best=True)
        self.valid_losses.append(self.valid_loss)
        self.learning_rates.append(get_lr(self.optimizer))
        return OODTrainingState(epoch, get_lr(self.optimizer), self.train_loss, self.valid_loss, valid_acc)

    def _eval(self, epoch: int, dataloader: DataLoader):
        if not dataloader:
            # TODO: this is temporary before I set up test image
            return 0
        total = 0
        correct, total_loss = 0, 0
        for it, (batch, domain) in enumerate(dataloader):
            x = batch[0].to(self.device)
            x = x.reshape(self.config.batch_size, -1)
            y = batch[1].to(self.device)
            feature = self.backbone(x)
            feature = feature[:, :self.latent_dim]
            scores = self.cls_classifier(feature)
            _, pred = scores.max(dim=1)
            correct += torch.sum(pred.eq(y)).item()
            loss_cls = self.criterion(scores, y)
            total_loss += loss_cls.item()
            total += x.shape[0]
        return correct / total, total_loss / total
