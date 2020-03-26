import hydra
from omegaconf import DictConfig
import logging

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from utils import load_model, save_model, AverageMeter


logger = logging.getLogger(__name__)


class LogisticRegression(nn.Module):

    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)


def run_epoch(args, loader, simclr_model, model, criterion, optimizer=None):
    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('Acc')

    if optimizer:
        simclr_model.train()
    else:
        simclr_model.eval()

    for step, (x, y) in enumerate(loader):
        x = x.to(args.device)
        y = y.to(args.device)

        # get encoding
        with torch.no_grad():
            h, z = simclr_model(x)
            # h = 512
            # z = 64

        output = model(h)
        loss = criterion(output, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), x.size(0))
        acc = (output.argmax(dim=1) == y).float().mean().item()
        acc_meter.update(acc, x.size(0))
        if optimizer and step % 5 == 0:
            print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}")

    return loss_meter.avg, acc_meter.avg


@hydra.main(config_path='config.yaml')
def run(args: DictConfig) -> None:
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = "./datasets"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            root, split="train", download=True, transform=torchvision.transforms.ToTensor()
        )
        test_dataset = torchvision.datasets.STL10(
            root, split="test", download=True, transform=torchvision.transforms.ToTensor()
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root, train=False, download=True, transform=transform
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    simclr_model, _, _ = load_model(args)
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    ## Logistic Regression
    n_classes = 10  # stl-10
    model = LogisticRegression(simclr_model.n_features, n_classes).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        loss_epoch, accuracy_epoch = run_epoch(args, train_loader, simclr_model, model, criterion, optimizer)
        print(f"Epoch {epoch} Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}")

    # final testing
    loss_epoch, accuracy_epoch = run_epoch(args, test_loader, simclr_model, model, criterion)
    print(f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}")