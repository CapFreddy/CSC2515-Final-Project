import bisect
from typing import List
from torchvision.datasets import ImageFolder
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
    def cumsum(sequence: List[ImageFolder]) -> List[int]:
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, dataroots: List[str], mode: str = "train"):
        super(DGDataset, self).__init__()
        assert len(dataroots) > 0, 'datasets should not be an empty iterable'
        self.datasets = [ImageFolder(
            root=f"../DataSets/digits_dg/{dataroot}/{mode}",
            transform=transforms.Compose(
                [transforms.Resize(32), transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            ))
            for dataroot in dataroots]
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self) -> int:
        """Get the last number in cummulative sizes, which should be the number of images in all datasets

        :return: size of this ensemble dataset (number of images)
        :rtype: int
        """
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx - (0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1])
        return self.datasets[dataset_idx][sample_idx], dataset_idx
