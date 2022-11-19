import os
import pathlib2

ROOT = pathlib2.Path(__file__).parent.absolute()

DIV2K_DATASET_PATH = (ROOT.parent / 'datasets' / 'DIV2K').absolute()
TEXT_DATASET_PATH = (ROOT.parent / 'datasets' / 'TEXT').absolute()


MSG_DIVIDER_LEN = 100


