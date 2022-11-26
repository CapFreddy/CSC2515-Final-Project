import os
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from torchmetrics.functional.classification import multiclass_accuracy
import numpy as np
from tqdm import tqdm

from DGDataset import DGDataset
from model import MLPClassifierReverse


def train(model, train_loader, val_loader, test_loader, args):
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    train_losses, val_accs, test_accs = [], [], []
    # for epoch in tqdm(range(args.num_epochs), desc='Training'):
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, args.alpha, criterion, train_loader, optimizer, args.device)
        val_acc = evaluate(model, val_loader, args.device)
        test_acc = evaluate(model, test_loader, args.device)

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        tqdm.write(f'Epoch {epoch} | '
                   f'Loss: {train_loss} | '
                   f'Val Acc {val_acc} | '
                   f'Test Acc {test_acc}')

        scheduler.step()

    result = {
        'train_loss': np.array(train_losses),
        'val_object_acc': np.array(val_accs),
        'test_acc': np.array(test_accs)
    }
    return result


def train_epoch(model, alpha, criterion, train_loader, optimizer, device):
    model.train()
    train_losses = []
    # for x, y_object, y_domain in tqdm(train_loader, desc='Training'):
    for it, (x, y_object, y_domain) in enumerate(train_loader):
        x = x.view(x.size(0), -1).to(device)
        y_object = y_object.to(device)
        y_domain = y_domain.to(device)

        logits_cls, logits_domain = model(x)
        loss = criterion(logits_cls, y_object) + alpha * criterion(logits_domain, y_domain)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum(train_losses) / len(train_losses)


def evaluate(model, eval_loader, device):
    model.to(device)
    model.eval()
    logits_object, ys_object = [], []
    # for x, y_object, _ in tqdm(eval_loader, desc='Evaluating'):
    for it, (x, y_object, _) in enumerate(eval_loader):
        x = x.view(x.size(0), -1).to(device)
        y_object = y_object.to(device)
        logits, _ = model(x)
        ys_object.append(y_object)
        logits_object.append(logits)

    ys_object = torch.cat(ys_object)
    logits_object = torch.cat(logits_object)
    num_classes = logits_object.size(-1)
    object_acc = multiclass_accuracy(logits_object, ys_object, num_classes, average='micro').item()
    return object_acc


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


def save_json(result, save_path):
    with open(save_path, 'w') as fout:
        best_val_epoch = result['val_object_acc'].argmax()
        best_val = result['val_object_acc'][best_val_epoch]
        best_val_test = result['test_acc'][best_val_epoch]
        best_test = result['test_acc'].max()
        result = {'best_val': best_val, 'best_val_test': best_val_test, 'best_test': best_test}
        tqdm.write(f'Best ValEpoch {best_val_epoch} | '
                   f'Best Val Object Acc {best_val} | '
                   f'Best Val Test Acc {best_val_test} | '
                   f'Best Test Acc {best_test}')
        json.dump(result, fout, indent=4)


def main(args):
    seed_everything(args.seed)
    train_loader, val_loader, test_loader = build_dataloader(args)
    model = MLPClassifierReverse(3 * 32 ** 2, args.hidden_dims)
    result = \
        train(model, train_loader, val_loader, test_loader, args)
    np.savez(os.path.join(args.logdir, 'phase1.npz'), **result)
    save_json(result, os.path.join(args.logdir, 'phase1.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 128, 64])
    datasets = ['mnist', 'mnist_m', 'svhn', 'syn']
    parser.add_argument('--target_domain', choices=datasets, default='svhn')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=5e-2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--logdir', type=str, default="../results/baseline_reverse/")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # args.logdir = os.path.join('result', args.logdir)
    args.logdir = os.path.join(args.logdir, "seed_" + str(args.seed))
    args.logdir = os.path.join(args.logdir, args.target_domain)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    main(args)
