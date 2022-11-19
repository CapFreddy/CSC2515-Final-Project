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


class MLP_Swap(nn.Module):
    def __init__(self, sizes, class_dim=128, domain_dim=128):
        super(MLP_Swap, self).__init__()
        assert sizes[-2] == class_dim + domain_dim
        self.class_dim = class_dim
        self.domain_dim = domain_dim
        self.mlp1 = mlp(sizes[:-1])
        self.mlp2 = mlp(sizes[-2:])

    def forward(self, x, mode="train"):
        feature = self.mlp1(x)
        if mode == "train":
            batch_size = feature.shape[0]
            feature_swaped1 = torch.cat(
                (feature[0:batch_size//2, 0:self.class_dim], feature[batch_size//2:, self.class_dim:]), dim=1)
            feature_swaped2 = torch.cat(
                (feature[batch_size // 2:, 0:self.class_dim], feature[:batch_size // 2, self.class_dim:]), dim=1)
            feature = torch.cat((feature, feature_swaped1), dim=0)
            feature = torch.cat((feature, feature_swaped2), dim=0)

            return self.mlp2(feature), feature[:batch_size, 0:self.class_dim]
        else:
            return self.mlp2(feature), feature[:, 0:self.class_dim]
