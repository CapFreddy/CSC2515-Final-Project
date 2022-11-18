from DGDataset import DGDataset
import network
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import argparse
from torch.optim import Adam
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--target", type=str, default="syn", help="target domain")
parser.add_argument("--baseline", type=bool, default=False, help="baseline method")
parser.add_argument("--final_cls", type=str, default="latent", help="final classifier method")
opt = parser.parse_args()
print(opt)

Digits_DG = ["mnist", "mnist_m", "svhn", "syn"]
Digits_DG.remove(opt.target)
train_set = Digits_DG
train_loader = DataLoader(dataset=DGDataset(dataroots=train_set, mode="train"),
                          batch_size=opt.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=DGDataset(dataroots=train_set, mode="val"),
                          batch_size=opt.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=DGDataset([opt.target], mode="test"),
                          batch_size=opt.batch_size, shuffle=True, drop_last=True)

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
backbone = network.MLP_Swap([3 * opt.img_size ** 2, 512, opt.latent_dim * 4, opt.latent_dim]).to(device)
cls_classifier = network.cls_classifier(opt.latent_dim).to(device)
domain_classifier = None if opt.baseline else network.domain_classifier(opt.latent_dim, 3).to(device)
final_classifier = network.cls_classifier(opt.latent_dim).to(device) if opt.final_cls == "same" \
    else network.cls_classifier(opt.latent_dim * 2).to(device)

optimizer = Adam(backbone.parameters(), lr=opt.lr)
optimizer_cls = Adam(cls_classifier.parameters(), lr=opt.lr)
optimizer_final = Adam(final_classifier.parameters(), lr=opt.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
scheduler_cls = lr_scheduler.StepLR(optimizer_cls, step_size=100, gamma=0.1)
scheduler_final = lr_scheduler.StepLR(optimizer_final, step_size=50, gamma=0.1)
if domain_classifier is not None:
    optimizer_domain = Adam(domain_classifier.parameters(), lr=opt.lr)
    scheduler_domain = lr_scheduler.StepLR(optimizer_domain, step_size=100, gamma=0.1)
criterion = nn.CrossEntropyLoss()


def eval_accuracy(data_loader, final=False):
    total = 0
    correct = 0
    for it, (batch, domain) in enumerate(data_loader):
        x = batch[0].to(device)
        x = x.reshape(opt.batch_size, -1)
        y = batch[1].to(device)
        # y = torch.cat((y, y), dim=0)
        # feature = feature[:, :opt.latent_dim]
        if final == True:
            _, feature = backbone(x, mode="test")
            scores = final_classifier(feature)
        else:
            feature, _ = backbone(x, mode="test")
            scores = cls_classifier(feature)
        _, pred = scores.max(dim=1)
        correct += torch.sum(pred.eq(y)).item()
        total += y.shape[0]
    return correct / total


def domain_accuracy(data_loader):
    total = 0
    correct = 0
    for it, (batch, domain) in enumerate(data_loader):
        x = batch[0].to(device)
        x = x.reshape(opt.batch_size, -1)
        domain = domain.to(device)
        domain_swaped = torch.cat((domain[opt.batch_size // 2:], domain[opt.batch_size // 2:]), dim=0)
        domain = torch.cat((domain, domain_swaped), dim=0)
        feature, _ = backbone(x)
        scores = domain_classifier(feature)
        _, pred = scores.max(dim=1)
        correct += torch.sum(pred.eq(domain)).item()
        total += domain.shape[0]
    return correct / total

train_accs = []
val_accs = []
test_accs = []
domain_accs = []
domain_val_accs = []
for epoch in range(opt.n_epochs):
    for it, (batch, domain) in enumerate(train_loader):
        x = batch[0].to(device)
        x = x.reshape(opt.batch_size, -1)
        y = batch[1].to(device)
        y = torch.cat((y, y), dim=0)
        domain = domain.to(device)
        domain_swaped = torch.cat((domain[opt.batch_size // 2:], domain[opt.batch_size // 2:]), dim=0)
        domain = torch.cat((domain, domain_swaped), dim=0)

        # feature, _ = backbone(x)
        if domain_classifier is not None:
            feature, _ = backbone(x)
            scores_cls = cls_classifier(feature)
            loss_cls = criterion(scores_cls, y)
            scores_domain = domain_classifier(feature)
            loss_domain = criterion(scores_domain, domain)
            loss = loss_cls + 0.5 * loss_domain
        else:
            feature, _ = backbone(x, mode="baseline")
            scores = cls_classifier(feature)
            loss_cls = criterion(scores, y)
            loss = loss_cls
        optimizer.zero_grad()
        optimizer_cls.zero_grad()
        if domain_classifier is not None:
            optimizer_domain.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_cls.step()
        if domain_classifier is not None:
            optimizer_domain.step()
    scheduler.step()
    scheduler_cls.step()
    if domain_classifier is not None:
        scheduler_domain.step()
    train_acc = eval_accuracy(data_loader=train_loader)
    val_acc = eval_accuracy(data_loader=val_loader)
    test_acc = eval_accuracy(data_loader=test_loader)
    domain_acc = domain_accuracy(data_loader=train_loader)
    domain_val_acc = domain_accuracy(data_loader=val_loader)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)
    domain_accs.append(domain_acc)
    domain_val_accs.append(domain_val_acc)
    print("epoch : %d || loss : %f || train_acc : %f || val_acc : %f || test_acc : %f || domain_acc : % f || domain_val_acc : %f"
          % (epoch, loss, train_acc, val_acc, test_acc, domain_acc, domain_val_acc))
print("Best train_acc : %f and its test_acc : %f" % (max(train_accs), test_accs[train_accs.index(max(train_accs))]))
print("Best val_acc : %f and its test_acc : %f" % (max(val_accs), test_accs[val_accs.index(max(val_accs))]))
print("Best test_acc : %f" % max(test_accs))
print("Best domain_acc : %f" % max(domain_accs))
print("Best domain_val_acc : %f" % max(domain_val_accs))

if opt.final_cls == "latent":
    train_accs = []
    val_accs = []
    test_accs = []
    for epoch in range(opt.n_epochs // 2):
        for it, (batch, domain) in enumerate(train_loader):
            x = batch[0].to(device)
            x = x.reshape(opt.batch_size, -1)
            y = batch[1].to(device)

            if opt.final_cls == "same":
                feature, _ = backbone(x)
            else:
                _, feature = backbone(x)
            scores = final_classifier(feature)
            loss_cls = criterion(scores, y)
            loss = loss_cls
            optimizer_final.zero_grad()
            loss.backward()
            optimizer_final.step()
        scheduler_final.step()
        train_acc = eval_accuracy(data_loader=train_loader)
        val_acc = eval_accuracy(data_loader=val_loader)
        test_acc = eval_accuracy(data_loader=test_loader)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        print(
            "epoch : %d || loss : %f || train_acc : %f || val_acc : %f || test_acc : %f"
            % (epoch, loss, train_acc, val_acc, test_acc))
    print("Best train_acc : %f and its test_acc : %f" % (max(train_accs), test_accs[train_accs.index(max(train_accs))]))
    print("Best val_acc : %f and its test_acc : %f" % (max(val_accs), test_accs[val_accs.index(max(val_accs))]))
    print("Best test_acc : %f" % max(test_accs))


