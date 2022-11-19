
import argparse
from torchvision import datasets, transforms
from src.model.mnist_nets import MnistNet
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from src.utils.config import WandbConfig
from torch.utils.data import Dataset, DataLoader
from src.constant import ROOT
import torch
import torch.nn as nn
from src.utils.config import Config
from pathlib2 import Path
from src.trainer.mnist_trainer import Trainer
import seaborn as sns
from src.dataset.DGDataset import DGDataset
from src.constant import *
from src.model import networks
from torch.optim import Adam
from torch.optim import lr_scheduler
from src.trainer.ood_trainer import OODTrainer




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="Learning Rate", default=1.0)
    parser.add_argument("--epochs", type=int, help="Epochs", default=15)
    parser.add_argument("--batch_size", type=int,
                        help="Batch Size", default=256)
    parser.add_argument("-w", "--workspace",
                        help="workspace path", required=True)
    parser.add_argument(
        "-n", "--name", help="Experiment Name, will be used in wandb")
    parser.add_argument("--target", type=str,
                        default=SYN, help="target domain")
    parser.add_argument("--img_size", type=int, default=32,
                        help="size of each image dimension")
    parser.add_argument("--baseline", action="store_true",
                        help="baseline method")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="dimensionality of the latent space")
    parser.add_argument("--num_worker", type=int, default=5, help="Number of Worker for dataloading")

    opt = parser.parse_args()
    sns.set_style("darkgrid")

    Digits_DG = DOMAINS.copy()
    Digits_DG.remove(opt.target)
    train_set = Digits_DG
    dataset_root = ROOT / 'dataset'
    train_loader = DataLoader(dataset=DGDataset(root=dataset_root, domains=train_set, mode="train"),
                              batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_worker)
    val_loader = DataLoader(dataset=DGDataset(root=dataset_root, domains=train_set, mode="val"),
                            batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_worker)
    # test_loader = DataLoader(dataset=DGDataset(root=dataset_root, domains=[opt.target], mode="test"),
    #                         batch_size=opt.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    workspace = Path(opt.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    backbone = networks.mlp(
        [3 * opt.img_size ** 2, 512, opt.latent_dim * 2]).to(device)
    cls_classifier = networks.cls_classifier(opt.latent_dim).to(device)
    domain_classifier = None if opt.baseline else networks.domain_classifier(
        opt.latent_dim, 3).to(device)

    optimizer = Adam(backbone.parameters(), lr=opt.lr)
    optimizer_cls = Adam(cls_classifier.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    scheduler_cls = lr_scheduler.StepLR(
        optimizer_cls, step_size=100, gamma=0.1)
    wandb_config = WandbConfig("ood", True)
    optimizer_domain = Adam(domain_classifier.parameters(), lr=opt.lr)
    scheduler_domain = lr_scheduler.StepLR(optimizer_domain, step_size=100, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    config = Config(epochs=opt.epochs, batch_size=opt.batch_size,
                    save_period=5, checkpoint_dir=workspace, num_worker=5, wandb=wandb_config, exp_name=opt.name)
    trainer = OODTrainer(model=backbone, optimizer=optimizer, optimizer_cls=optimizer_cls, optimizer_domain=optimizer_domain,
                         scheduler=scheduler, scheduler_cls=scheduler_cls, scheduler_domain=scheduler_domain,
                         criterion=criterion, config=config,
                         train_dataloader=train_loader, valid_dataloader=val_loader, test_dataloader=None,
                         cls_classifier=cls_classifier, domain_classifier=domain_classifier,
                         latent_dim=opt.latent_dim)
    trainer.train()


    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    # model = MnistNet()
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    # train_dataset = datasets.MNIST(
    #     ROOT.parent / 'dataset', train=True, download=True, transform=transform)
    # # train_dataset = torch.utils.data.Subset(train_dataset, list(range(100))) # TODO: remove this line

    # test_dataset = datasets.MNIST(
    #     ROOT.parent / 'dataset', train=False, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=opt.batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=opt.batch_size)
    # workspace = Path(opt.workspace)
    # workspace.mkdir(parents=True, exist_ok=True)
    # criterion = F.nll_loss
    # wandb_config = WandbConfig("mnist", True)
    # config = Config(epochs=opt.epochs, batch_size=opt.batch_size,
    #                 save_period=5, checkpoint_dir=workspace, num_worker=5, wandb=wandb_config, exp_name=opt.name)

    # trainer = Trainer(model, optimizer, scheduler, criterion,
    #                   config, train_dataset, train_loader, test_loader)
    # trainer.train()
