import os
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser()

    # Model Architecture
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 128, 64])
    parser.add_argument('--domain_dim', type=int, required=True)
    parser.add_argument('--disentangle_layer', type=int, required=True)

    # Training
    datasets = ['mnist', 'mnist_m', 'svhn', 'syn']
    parser.add_argument('--target_domain', choices=datasets, default='syn')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_latent_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_lr', type=float, default=1e-5)

    # Reproducibility
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    if args.disentangle_layer == -1:
        args.disentangle_layer = len(args.hidden_dims)
    assert 0 < args.disentangle_layer <= len(args.hidden_dims)

    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    args.logdir = os.path.join('result', args.logdir)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    return args
