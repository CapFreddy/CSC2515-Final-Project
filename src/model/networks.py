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
    # !Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument
    # TODO: I set dim to 1, verify this
    layers += [nn.Linear(sizes, 10), nn.Softmax(dim=1)]
    return nn.Sequential(*layers)


def domain_classifier(size, n_domains):
    # Build a feedforward neural network.
    layers = []
    layers += [nn.Linear(size, n_domains), nn.Softmax(dim=1)]
    return nn.Sequential(*layers)


class MLP_Swap(nn.Module):
    def __init__(self, sizes):
        super(MLP_Swap, self).__init__()

        self.mlp1 = mlp(sizes[:-1])
        self.mlp2 = mlp(sizes[-2:])

    def forward(self, x, mode="train"):
        feature = self.mlp1(x)
        batch_szie = feature.shape[0]
        latent_dim = feature.shape[1]
        if mode == "train":
            feature_swaped1 = torch.cat(
                (feature[0:batch_szie//2, 0:latent_dim//2], feature[batch_szie//2:, latent_dim//2:]), dim=1)
            feature_swaped2 = torch.cat(
                (feature[batch_szie // 2:, 0:latent_dim // 2], feature[:batch_szie // 2, latent_dim // 2:]), dim=1)
            feature = torch.cat((feature, feature_swaped1), dim=0)
            feature = torch.cat((feature, feature_swaped2), dim=0)

            return self.mlp2(feature), feature[:batch_szie, 0:latent_dim//2]
        else:
            return self.mlp2(feature), feature[:, 0:latent_dim//2]
