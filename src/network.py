import torch
import torch.nn as nn
from typing import List

def mlp(sizes: List[int], activation: nn.Module=nn.ReLU) -> nn.Module:
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def cls_classifier(size: int) -> nn.Module:
    # Build a feedforward neural network.
    layers = []
    layers += [nn.Linear(size, 10), nn.Softmax()]
    return nn.Sequential(*layers)


def domain_classifier(size: int, n_domains: int) -> nn.Module:
    # Build a feedforward neural network.
    layers = []
    layers += [nn.Linear(size, n_domains), nn.Softmax()]
    return nn.Sequential(*layers)