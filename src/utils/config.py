import torch
from pathlib2 import Path
from typing import Union, Optional, Callable
from dataclasses import dataclass

@dataclass
class WandbConfig:
    project: str
    watch_model: bool

@dataclass
class Config:
    """Config is a dataclass acting as a container of config variables"""
    epochs: int
    batch_size: int
    save_period: int # frequency to save a checkpoint
    checkpoint_dir: Path # where to save checkpoint model weights
    # criterion: Union[torch.nn.Module, Callable]
    # optimizer: torch.optim.Optimizer
    num_worker: int # If any error occurs related to multiprocessing, try change this parameter to 0.
    exp_name: str = None
    wandb: Optional[WandbConfig] = None
    start_epoch: int = 1
    train_set_percentage: float = 0.9 # percentage of dataset used for training, the rest is used for testing
    test_only: bool = False # Run test only (entire dataset used for testing)
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
