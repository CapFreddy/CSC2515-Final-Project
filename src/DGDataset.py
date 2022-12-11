import os
import bisect

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DGDataset(Dataset):

    def __init__(self, datasets, mode, root=os.path.join(root_path, 'digits_dg')):
        super(DGDataset, self).__init__()
        assert len(datasets) > 0

        self.cached = False
        self.cached_samples = []

        self.datasets = []
        tfs = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if mode == 'test':
            assert len(datasets) == 1
            for split in ['train', 'val']:
                self.datasets.append(ImageFolder(
                    root=os.path.join(root, datasets[0], split),
                    transform=tfs
                ))
        else:
            assert mode in ['train', 'val']
            for dataset in datasets:
                self.datasets.append(ImageFolder(
                    root=os.path.join(root, dataset, mode),
                    transform=tfs
                ))
        self.cumulative_sizes = np.cumsum(list(map(len, self.datasets))).tolist()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if self.cached:
            return self.cached_samples[idx]

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        (x, y_object), y_domain = self.datasets[dataset_idx][sample_idx], dataset_idx
        return x, y_object, y_domain

    def cache_samples(self):
        assert not self.cached
        for i in tqdm(range(len(self)), desc='Caching dataset'):
            self.cached_samples.append(self[i])
        self.cached = True
