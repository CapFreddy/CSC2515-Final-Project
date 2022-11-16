import torch
import torch.nn as nn


def mlp(sizes, activation=nn.ReLU):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def cls_classifier(sizes):
    # Build a feedforward neural network.
    layers = []
    layers += [nn.Linear(sizes, 10), nn.Softmax()]
    return nn.Sequential(*layers)


def domain_classifier(size, n_domains):
    # Build a feedforward neural network.
    layers = []
    layers += [nn.Linear(size, n_domains), nn.Softmax()]
    return nn.Sequential(*layers)