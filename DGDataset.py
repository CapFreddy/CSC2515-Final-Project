import bisect
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DGDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, dataroots, mode="train"):
        super(DGDataset, self).__init__()
        assert len(dataroots) > 0, 'datasets should not be an empty iterable'
        self.datasets = []
        for dataroot in dataroots:
            self.datasets.append(datasets.ImageFolder(
                root="../DataSets/digits_dg/" + dataroot + "/" + mode,
                transform=transforms.Compose(
                    [transforms.Resize(32), transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )))
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        test = self.datasets[dataset_idx][sample_idx]
        return self.datasets[dataset_idx][sample_idx], dataset_idx