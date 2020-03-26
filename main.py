import hydra
from omegaconf import DictConfig
import logging

import torch


from torchvision import datasets
from torch.utils.data import DataLoader
from nt_xent import NT_Xent
from simclr_transform import TransformsSimCLR
from utils import load_model, save_model, AverageMeter


logger = logging.getLogger(__name__)


def mask_correlated_samples(batch_size):
    mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


def train_epoch(args, train_loader, model, criterion, optimizer, scheduler=None):
    loss_meter = AverageMeter('loss')
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.to(args.device)
        x_j = x_j.to(args.device)

        # positive pair, with encoding
        h_i, z_i = model(x_i)
        h_j, z_j = model(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        loss_meter.update(loss.item(), x_i.size(0))
    return loss_meter.avg


@hydra.main(config_path='config.yaml')
def run(args: DictConfig) -> None:
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=args.data_dir, download=True, transform=TransformsSimCLR()
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    model, optimizer, scheduler = load_model(args)

    mask = mask_correlated_samples(args.batch_size)
    criterion = NT_Xent(args.batch_size, args.temperature, mask, args.device)

    for epoch in range(args.epochs):
        train_loss = train_epoch(args, train_loader, model, criterion, optimizer, scheduler=scheduler)

        logger.info('Epoch {}, train_loss: {:.4f}'.format(epoch, train_loss))
        if epoch % 10 == 9:
            save_model(args, model, epoch + 1)


if __name__ == '__main__':
    run()