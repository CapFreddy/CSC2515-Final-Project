import os
import time
import wandb
import torch
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
from pathlib2 import Path
from ..utils.config import Config
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from typing import List, Dict, Callable, Union
from ..logger.memory_profile import MemoryProfiler
from ..utils.util import get_divider_str, save_checkpoint_state, load_checkpoint_state
from ..constant import MSG_DIVIDER_LEN

from dataclasses import dataclass, asdict


@dataclass
class TrainingState:
    epoch: int
    lr: float
    train_loss: float
    valid_loss: float


class BaseTrainer(ABC):
    """Base Trainer Template
    Design a general pytorch neural network training template using OOP design
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: object, criterion: Union[torch.nn.Module, Callable], config: Config, train_dataset: Dataset) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.epochs = self.config.epochs
        self.device = self.config.device
        self.train_dataset = train_dataset
        self.state: TrainingState
        self.save_period = config.save_period
        self.start_epoch = config.start_epoch
        self.model_weights_dir = self.config.checkpoint_dir / 'weights'
        self.valid_results = self.config.checkpoint_dir / 'validation'
        self.log_path = self.config.checkpoint_dir / 'log.log'
        self.cur_epoch = 0
        self.logger = logging.getLogger(os.path.basename(
            Path(config.checkpoint_dir).absolute()))
        if self.start_epoch == 0:
            raise ValueError("start_epoch must start from at least 1")
        self.progress_bar: tqdm = None
        self.memory_profiler = MemoryProfiler(self.logger)
        self.train_loss, self.valid_loss = 0, 0
        ################## load valid_loss and train_loss if this is not starting from the beginning ##################
        if self.start_epoch != 1 and not self.config.checkpoint_dir.exists():
            raise ValueError(
                "Start Epoch is not 1 but checkpoint directory doesn't exist. Verify your Configurations.")
        if self.start_epoch != 1 and self.config.checkpoint_dir.exists():
            print(self.start_epoch)
            self.logger.info("loading history csv with pandas")
            history_df = pd.read_csv(self.history_csv_path)
            self.train_losses = history_df['train_loss'].to_list()
            self.valid_losses = history_df['valid_loss'].to_list()
            self.learning_rates = history_df['learning_rate'].to_list()
            if len(self.train_losses) < self.start_epoch or len(self.valid_losses) < self.start_epoch:
                raise ValueError(
                    f'There is not enough loss in previous loss files.\n'
                    f'Start Epoch={self.start_epoch}, train_loss length={len(self.train_losses)}, '
                    f'valid_loss length={len(self.valid_losses)}')
            else:
                self.train_losses = list(self.train_losses[:self.start_epoch])
                self.valid_losses = list(self.valid_losses[:self.start_epoch])
                self.learning_rates = list(
                    self.learning_rates[:self.start_epoch])
            self.logger.info(
                f'loaded training loss from previous train (length={len(self.train_losses)}):')
            self.logger.debug(self.train_losses)
            self.logger.info(
                f'loaded validation loss from previous train (length={len(self.valid_losses)}):')
            self.logger.debug(self.valid_losses)
        else:
            self.train_losses: List[float] = []
            self.valid_losses: List[float] = []
            self.learning_rates: List[float] = []
        ################################# load checkpoint weights if needed #################################
        if not (self.start_epoch <= 1 and self.config.checkpoint_dir.exists()):
            # load weights
            self.logger.info(
                f"'start_epoch' is not 1, looking for epoch{self.start_epoch}.pth")
            weights_files = os.listdir(self.model_weights_dir)
            if f'epoch{self.start_epoch}.pth' in weights_files:
                self.logger.info(
                    f"epoch{self.start_epoch}.pth found, load model state. Will start training epoch {self.start_epoch + 1}")
                state = load_checkpoint_state(
                    self.model_weights_dir / f'epoch{self.start_epoch}.pth')
                self.model.load_state_dict(state['state_dict'])
                self.optimizer.load_state_dict(state['optimizer'])
                # self.model.load_state_dict(torch.load(
                #     self.model_weights_dir / f'epoch{self.start_epoch}.pth'))
                self.model.eval()
            else:
                raise ValueError(
                    f"Weight file not found, the start epoch is {self.start_epoch}, epoch{self.start_epoch}.pth doesn't exist in {self.model_weights_dir}")
        for path in [self.config.checkpoint_dir, self.model_weights_dir, self.valid_results]:
            path.mkdir(parents=True, exist_ok=True)

        # start from next epoch as start_epoch has been trained already
        if self.start_epoch != 1:
            self.start_epoch += 1

    @property
    def history_csv_path(self) -> Path:
        return self.config.checkpoint_dir / 'history.csv'

    @abstractmethod
    def _train_epoch(self, epoch: int):
        """Training logic for an epoch
        :param epoch: Current epoch number
        :type epoch: int
        :raises NotImplementedError: Abstract method not implemented
        """
        raise NotImplementedError

    def wandb_epoch_log(self) -> None:
        if self.state:
            wandb.log(asdict(self.state))

    def wandb_config_log(self, data: Dict) -> None:
        d = {}  # default parameters, and update it with data dict
        d.update(data)
        wandb.config = d

    def train(self):
        """Sample train function
        Iterate epochs and call self._train_epoch in every iteration, while also do stuff like logging to simplify self._train_epoch
        so that self._train_epoch just need to focus on the training logic
        """
        self.model = self.model.to(self.config.device)
        if self.config.wandb is not None:
            wandb.init(self.config.wandb.project, name=self.config.exp_name)
            self.wandb_config_log(
                {"batch_size": self.config.batch_size, "epochs": self.config.epochs})
            if self.config.wandb.watch_model:
                wandb.watch(self.model)
        self.logger.info(get_divider_str('Training Started', MSG_DIVIDER_LEN))
        start_time = time.time()
        with tqdm(total=len(self.train_dataset) * (self.epochs - self.start_epoch + 1)) as progress_bar:
            # with tqdm(range(self.start_epoch, self.epochs + 1), total=self.epochs, file=sys.stdout) as progress_bar:
            self.progress_bar = progress_bar
            for epoch in range(self.start_epoch, self.epochs + 1):
                self.cur_epoch = epoch
                self.progress_bar.set_description(
                    'epoch: {}/{}'.format(epoch, self.epochs))
                self.state = self._train_epoch(epoch) # !this is the main training logic, the rest are all sumplementary
                if epoch % self.save_period == 0 or epoch == self.epochs:
                    self._save_checkpoint(epoch)
                self._update_loss_plot()
                self.memory_profiler.update_n_log(epoch)
                if self.config.wandb is not None:
                    self.wandb_epoch_log()
        self.progress_bar.close()
        self.logger.info(get_divider_str('Training Finished', MSG_DIVIDER_LEN))
        self.memory_profiler.log_final_message()
        elapsed_time = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - start_time))
        self.logger.info(f"Total Training Time: {elapsed_time}")
        self.logger.info(get_divider_str(
            'Saving Final Checkpoint', MSG_DIVIDER_LEN))

    def _save_checkpoint(self, epoch, is_best=False):
        # save model state
        checkpoint_state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        save_checkpoint_state(state=checkpoint_state, epoch=epoch, is_best=is_best, checkpoint_dir=os.path.join(
            self.model_weights_dir))

    def _update_loss_plot(self):
        """Plot current loss plot
        TODO: Use seaborn + pandas to plot the loss plot
        """
        # training loss plot
        assert len(self.valid_losses) == len(self.train_losses) == len(
            self.learning_rates), f'valid_loss, train_loss, learning_rates should have the same length'
        history_df = pd.DataFrame(data={
            "epoch": list(range(1, len(self.valid_losses) + 1)),
            "valid_loss": self.valid_losses,
            "train_loss": self.train_losses,
            "learning_rate": self.learning_rates
        })
        history_df.to_csv(self.history_csv_path)

        if len(self.train_losses) != 0 and len(self.valid_losses) != 0:
            self.logger.debug("plot training and validation loss")
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
            sns.lineplot(data=history_df, x='epoch', y='train_loss', ax=axes[0])
            sns.lineplot(data=history_df, x='epoch', y='valid_loss', ax=axes[1])
            axes[0].set_ylabel("loss")
            axes[0].set_title("Training Loss")
            axes[1].set_ylabel("loss")
            axes[1].set_title("Validation Loss")
            fig.savefig(self.config.checkpoint_dir / 'loss.png')
            plt.close()
        else:
            # training loss or validation loss is missing, cannot save both loss in to the same image
            if len(self.train_losses) != 0:
                self.logger.debug("plot training loss")
                plt.figure()
                plt.plot(list(range(1, len(self.train_losses) + 1)),
                         self.train_losses)
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.title("Training Loss")
                plt.savefig(self.config.checkpoint_dir / 'train_loss.png')
                plt.close()
            else:
                self.logger.error("error: no training loss")
            if len(self.valid_losses) != 0:
                self.logger.debug("plot validation loss")
                plt.figure()
                plt.plot(list(range(1, len(self.valid_losses) + 1)),
                         self.valid_losses)
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.title("Validation Loss")
                plt.savefig(self.config.checkpoint_dir / 'valid_loss.png')
                plt.close()
            else:
                self.logger.error("error: no validation loss")
        if len(self.learning_rates) != 0:
            plt.figure()
            plt.plot(list(range(1, len(self.learning_rates) + 1)),
                     self.learning_rates)
            plt.xlabel("epoch")
            plt.ylabel("learning rates")
            plt.title("Learning Rates")
            plt.close()
