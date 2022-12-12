import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from torchmetrics.functional.classification import multiclass_accuracy
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from argparser import parse_args
from DGDataset import DGDataset
from model import DomainAwareEncoder, ObjectDomainClassifier, LatentObjectClassifier


def train_object_domain_model(model, train_loader, val_loader, args):
    # Method 1/2/3-1
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    best_val_acc = 0.
    train_losses, val_object_accs, val_domain_accs = [], [], []
    for epoch in tqdm(range(args.num_epochs), desc='Training'):
        train_loss = train_epoch(model, criterion, train_loader, optimizer, args.alpha, args.device)
        val_object_acc, val_domain_acc = evaluate(model, val_loader, args.device)
        if val_object_acc > best_val_acc:
            best_val_model = copy.deepcopy(model)

        tqdm.write(f'Epoch {epoch} | '
                   f'Loss: {train_loss} | '
                   f'Val Object Acc {val_object_acc} | '
                   f'Val Domain Acc {val_domain_acc}')

        train_losses.append(train_loss)
        val_object_accs.append(val_object_acc)
        val_domain_accs.append(val_domain_acc)

        scheduler.step()

    result = {
        'train_loss': np.array(train_losses),
        'val_object_acc': np.array(val_object_accs),
        'val_domain_acc': np.array(val_domain_accs)
    }
    return best_val_model, result
    

def train_latent_object_model(model, train_loader, val_loader, args):
    # Method 3-2
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    if args.finetune:
        optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr': args.finetune_lr},
                                {'params': model.object_clf.parameters()}],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(model.object_clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    best_val_acc = 0.
    train_losses, val_object_accs = [], []
    for epoch in tqdm(range(args.num_latent_epochs), desc='Training'):
        train_loss = train_epoch(model, criterion, train_loader, optimizer, args.alpha, args.device)
        val_object_acc, _ = evaluate(model, val_loader, args.device)
        if val_object_acc > best_val_acc:
            best_val_model = copy.deepcopy(model)

        train_losses.append(train_loss)
        val_object_accs.append(val_object_acc)

        tqdm.write(f'Epoch {epoch} | '
                   f'Loss: {train_loss} | '
                   f'Val Object Acc {val_object_acc}')

        scheduler.step()

    result = {
        'train_loss': np.array(train_losses),
        'val_object_acc': np.array(val_object_accs)
    }
    return best_val_model, result


def train_epoch(model, criterion, train_loader, optimizer, alpha, device):
    model.train()  # Enables permutation
    train_losses = []
    for x, y_object, y_domain in tqdm(train_loader, desc='Training'):
        x = x.view(x.size(0), -1).to(device)
        y_object = y_object.to(device)
        y_domain = y_domain.to(device)
        perm = torch.randperm(x.size(0)).to(device)

        logits = model(x, perm=perm)
        if isinstance(logits, tuple):
            # Method 1/2/3-1
            logits_object, logits_domain = logits
            if logits_object.size(0) == 2 * x.size(0):
                # Method 2/3-1
                y_object = torch.cat((y_object, y_object))
                y_domain = torch.cat((y_domain, y_domain[perm]))

            loss_object = criterion(logits_object, y_object)
            loss_domain = criterion(logits_domain, y_domain)
            loss = loss_object + alpha * loss_domain
        else:
            # Method 3-2
            loss = criterion(logits, y_object)

        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum(train_losses) / len(train_losses)


def evaluate(model, eval_loader, device):
    model.to(device)
    model.eval()  # Disables permutation
    logits_object, logits_domain = [], []
    ys_object, ys_domain = [], []
    for x, y_object, y_domain in tqdm(eval_loader, desc='Evaluating'):
        x = x.view(x.size(0), -1).to(device)
        y_object = y_object.to(device)
        y_domain = y_domain.to(device)

        logits = model(x)
        if isinstance(logits, tuple):
            # Method 1/2/3-1
            ys_object.append(y_object)
            ys_domain.append(y_domain)
            logits_object.append(logits[0])
            logits_domain.append(logits[1])
        else:
            # Method 3-2
            ys_object.append(y_object)
            logits_object.append(logits)

    ys_object = torch.cat(ys_object)
    logits_object = torch.cat(logits_object)
    num_classes = logits_object.size(-1)
    object_acc = multiclass_accuracy(logits_object, ys_object, num_classes, average='micro').item()
    if len(ys_domain):
        ys_domain = torch.cat(ys_domain)
        logits_domain = torch.cat(logits_domain)
        num_classes = logits_domain.size(-1)
        domain_acc = multiclass_accuracy(logits_domain, ys_domain, num_classes, average='micro').item()
    else:
        domain_acc = None

    return object_acc, domain_acc


def build_dataloader(args):
    datasets = ['mnist', 'mnist_m', 'svhn', 'syn']
    datasets.remove(args.target_domain)
    train_dataset, val_dataset, test_dataset = DGDataset(datasets, mode='train'), \
                                               DGDataset(datasets, mode='val'), \
                                               DGDataset([args.target_domain], mode='test')

    train_dataset.cache_samples()
    val_dataset.cache_samples()
    test_dataset.cache_samples()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    return train_loader, val_loader, test_loader


def visualize_features(model, eval_loader, datasets, device, logdir):
    model.to(device)
    model.eval()  # Disables permutation
    feats_object, feats_domain = [], []
    ys_object, ys_domain = [], []
    for x, y_object, y_domain in tqdm(eval_loader, desc='Evaluating'):
        x = x.view(x.size(0), -1).to(device)
        y_object = y_object.to(device)
        y_domain = y_domain.to(device)
        feat_object, feat_domain = model(x, return_feature=True)
        ys_object.append(y_object)
        ys_domain.append(y_domain)
        feats_object.append(feat_object)
        feats_domain.append(feat_domain)

    ys_object = torch.cat(ys_object).cpu().numpy()
    ys_domain = torch.cat(ys_domain).cpu().numpy()
    ys_domain = [datasets[y_domain] for y_domain in ys_domain]
    feats_object = torch.cat(feats_object).detach().cpu().numpy()
    feats_domain = torch.cat(feats_domain).detach().cpu().numpy()

    plot_tnse(feats_object, ys_domain, os.path.join(logdir, 'object.png'))
    plot_tnse(feats_domain, ys_domain, os.path.join(logdir, 'domain.png'))


def plot_tnse(features, ys, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    features = tsne.fit_transform(features)
    ax = sns.scatterplot(data={'x': features[:, 0], 'y': features[:, 1]},
                         x='x', y='y', hue=ys, s=10,
                         palette=sns.color_palette('hls', len(set(ys))))
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    plt.legend(loc='upper right', title=None)
    plt.savefig(save_path, dpi=300)
    plt.clf()


def main():
    args = parse_args()
    seed_everything(args.seed)
    train_loader, val_loader, _ = build_dataloader(args)
    model = ObjectDomainClassifier(
        DomainAwareEncoder(3 * 32**2,
                           args.hidden_dims,
                           args.domain_dim,
                           args.disentangle_layer)
    )
    best_val_model, _ = train_object_domain_model(model, train_loader, val_loader, args)
    datasets = ['mnist', 'mnist_m', 'svhn', 'syn']
    datasets.remove(args.target_domain)
    visualize_features(best_val_model, val_loader, datasets, args.device, args.logdir)

    if args.disentangle_layer != len(args.hidden_dims):
        # Method 3-2
        model = LatentObjectClassifier(best_val_model.encoder)
        best_val_model, _ = train_latent_object_model(model, train_loader, val_loader, args)
        visualize_features(best_val_model, val_loader, datasets, args.device, args.logdir)


if __name__ == '__main__':
    main()
