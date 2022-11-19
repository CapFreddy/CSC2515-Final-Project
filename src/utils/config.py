from pathlib2 import Path
from typing import Union


class Config:
    def __init__(self, epochs: int, checkpoint_dir: Union[str, Path]) -> None:
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir).absolute()

        